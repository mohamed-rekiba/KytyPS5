#include "graphics/guest_gpu/capture/gpuPlayer.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/capture/captureFile.h"
#include "graphics/guest_gpu/capture/guestMemoryAccess.h"
#include "graphics/guest_gpu/capture/replayLogic.h"
#include "graphics/guest_gpu/capture/stateLayout.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/workCounters.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::Capture {

namespace {

namespace Memory = LibKernel::Memory;

constexpr uint64_t GuestPage = 0x4000;

void Report(const std::string& text) {
	Log::WriteToConsoleAndLog("GPU replay: " + text + "\n");
}

std::string Megabytes(uint64_t bytes) {
	return fmt::format("{:.1f} MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
}

bool BytesAre(uint64_t address, std::span<const uint8_t> wanted) {
	std::array<uint8_t, 16> current {};
	return wanted.size() <= current.size() &&
	       Memory::TryReadBacking(address, current.data(), wanted.size()) &&
	       std::memcmp(current.data(), wanted.data(), wanted.size()) == 0;
}

// The guest ranges the renderer sees as mapped. A range the capture unmaps keeps its memory in
// this process: only the renderer is told, which is all a replay needs, and the next pass can
// have the range back without mapping memory again.
class GuestMapping final {
public:
	explicit GuestMapping(RenderContext& renderer): m_renderer(renderer) {}

	// Makes `ranges` the mapped ranges. False when memory for a new range cannot be had.
	[[nodiscard]] bool Set(const std::vector<Range>& ranges) {
		// Guest mappings are made of 16 KiB pages; neighbours that touch after rounding become
		// one.
		RangeSet wanted;
		for (const auto& range: ranges) {
			const uint64_t begin = range.address & ~(GuestPage - 1);
			const uint64_t end   = (range.address + range.size + GuestPage - 1) & ~(GuestPage - 1);
			wanted.Add(begin, end - begin);
		}
		auto removed = m_mapped;
		auto added   = wanted;
		wanted.ForEach([&](uint64_t begin, uint64_t end) { removed.Subtract(begin, end - begin); });
		m_mapped.ForEach([&](uint64_t begin, uint64_t end) { added.Subtract(begin, end - begin); });
		removed.ForEach(
		    [&](uint64_t begin, uint64_t end) { m_renderer.UnmapMemory(begin, end - begin); });
		bool ok = true;
		added.ForEach([&](uint64_t begin, uint64_t end) { ok = ok && Map(begin, end); });
		m_mapped = std::move(wanted);
		return ok;
	}

private:
	[[nodiscard]] bool Map(uint64_t begin, uint64_t end) {
		// The part that has memory from an earlier mapping only goes back to the renderer.
		auto fresh = RangeSet {};
		fresh.Add(begin, end - begin);
		m_backed.ForEachInRange(begin, end - begin, [&](uint64_t from, uint64_t to) {
			m_renderer.MapMemory(from, to - from);
			fresh.Subtract(from, to - from);
		});
		bool ok = true;
		fresh.ForEach([&](uint64_t from, uint64_t to) {
			constexpr int ProtCpuGpuReadWrite = 0x33;
			constexpr int MapFixed            = 0x10;
			const auto    size                = to - from;
			int64_t       physical            = 0;
			void*         address             = reinterpret_cast<void*>(from);
			// Mapping it tells the renderer.
			if (Memory::KernelAllocateMainDirectMemory(size, GuestPage, 0, &physical) != 0 ||
			    Memory::KernelMapDirectMemory(&address, size, ProtCpuGpuReadWrite, MapFixed,
			                                  physical, GuestPage) != 0 ||
			    reinterpret_cast<uint64_t>(address) != from) {
				Report(fmt::format("cannot map 0x{:x} bytes at 0x{:016x}", size, from));
				ok = false;
				return;
			}
			m_backed.Add(from, size);
		});
		return ok;
	}

	RenderContext& m_renderer;
	RangeSet       m_mapped; // what the renderer sees
	RangeSet       m_backed; // what has memory in this process
};

// Adds the time it lives to a counter of clock ticks.
class WorkTimer final {
public:
	explicit WorkTimer(std::atomic<int64_t>& total)
	    : m_total(total), m_started(std::chrono::steady_clock::now()) {}
	~WorkTimer() { m_total.fetch_add((std::chrono::steady_clock::now() - m_started).count()); }
	WorkTimer(const WorkTimer&)            = delete;
	WorkTimer& operator=(const WorkTimer&) = delete;

private:
	std::atomic<int64_t>&                 m_total;
	std::chrono::steady_clock::time_point m_started;
};

// What the live run did to guest memory before one point of the stream: the ranges the caches
// were told about, and the pages that changed.
struct MemoryUpdate {
	// In the order they came. Each carries how many draws and dispatches after the point of the
	// stream before this one it came.
	std::vector<WriteRange> writes;
	// Ranges the renderer of the live run wrote itself, around this point: it copied what the
	// host GPU held to guest memory. Bytes that changed there are put in place without telling
	// the caches, as in the live run: telling them would make the replay load its image or
	// buffer from those bytes, which are older than what its GPU holds by then.
	RangeSet                           host_written;
	std::vector<const Reader::Record*> pages;
	// The ranges that are mapped from this point on, when the guest changed them.
	const std::vector<Range>* mapping = nullptr;
	// The update of the next point of the stream, in the order of the live run, and the place
	// of this one in that order.
	const MemoryUpdate* next  = nullptr;
	uint32_t            order = 0;
	// The live run stored this update when the renderer read memory that had changed, after
	// `position` draws and dispatches since the update before it. No record of the stream
	// brings the replay to it, so the GPU thread applies it when it has done as much work.
	bool     read_point = false;
	uint32_t position   = 0;
};

// The images the host GPU of the live run had written after `position` draws and dispatches.
struct ImageChecks {
	uint64_t                position = 0;
	std::vector<ImageCheck> images;
};

// What one frame of a pass showed and cost.
struct ReplayedFrame {
	Picture picture;
	// Only when the picture differs from the live run's: how far it is from it.
	PictureDifference difference;
	uint64_t          picture_size = 0;
	bool              flipped      = false;
	WorkCounts        work;
};

void FinishHostWork(RenderContext& renderer) {
	auto& scheduler = renderer.GetCommandScheduler();
	if (scheduler.Active()) {
		const auto tick = scheduler.CurrentTick();
		scheduler.Finish();
		scheduler.WaitPriorityOperations(tick);
	}
}

// Player side of the GPU thread hooks: puts the recorded guest memory in place where the live
// run had it, and tells the player thread what the GPU thread is doing.
class PlayerObserver final: public GuestGpuObserver {
public:
	// A wait on guest memory as the live run saw it pass.
	struct RecordedWait {
		uint64_t                 address = 0;
		std::span<const uint8_t> bytes;
		bool                     had_blocked = false;
		// What the live run did to guest memory before this wait passed.
		const MemoryUpdate* update = nullptr;
	};

	// A picture of the live run: its hash, and its bytes when the capture has them.
	struct LivePicture {
		Picture                  picture;
		std::span<const uint8_t> bytes;
	};

	PlayerObserver(RenderContext& renderer, GuestMapping& mapping, std::vector<RecordedWait> waits,
	               std::vector<GuestRange> live_buffers, std::vector<LivePicture> live_pictures,
	               std::vector<ImageChecks> checks)
	    : m_renderer(renderer), m_mapping(mapping), m_waits(std::move(waits)),
	      m_live_buffers(std::move(live_buffers)), m_live_pictures(std::move(live_pictures)),
	      m_checks(std::move(checks)),
	      m_frame_count(static_cast<uint32_t>(m_live_pictures.size())) {
		m_schedule_waits.reserve(m_waits.size());
		for (const auto& wait: m_waits) {
			m_schedule_waits.push_back({wait.address, wait.had_blocked});
		}
	}

	// Player thread: the memory update to apply when the next submission starts.
	void ExpectSubmission(const MemoryUpdate* update) {
		std::lock_guard lock(m_mutex);
		m_pending_update = update;
		m_started        = false;
	}

	void WaitUntilStarted() {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [this] { return m_started; });
	}

	// Player thread: the replay has reached the point where the live run saw wait `index` pass.
	void ReleaseWait(size_t index) {
		std::lock_guard lock(m_mutex);
		m_schedule.Release(index);
	}

	// Player thread: what the live run did to guest memory before the next frame end.
	void QueueFrameEnd(const MemoryUpdate* update) {
		std::lock_guard lock(m_mutex);
		m_frame_end_updates.push_back(update);
	}

	// Player thread: starts a pass.
	void BeginPass() {
		(void)g_work.Take();
		m_player_time.store(0);
		std::lock_guard lock(m_mutex);
		m_schedule.Arm(m_schedule_waits);
		m_frames_hashed = 0;
		m_frames.clear();
		m_frame_end_updates.clear();
		m_timed_from = std::chrono::steady_clock::now();
	}

	// Player thread: waits until the GPU thread has processed every frame end of the pass, then
	// returns its frames.
	[[nodiscard]] std::vector<ReplayedFrame> WaitForFrames() {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [this] { return m_frames.size() >= m_frame_count; });
		return m_frames;
	}

	// When the timed part of the pass began: after its first frame, when there is more than one.
	[[nodiscard]] std::chrono::steady_clock::time_point TimedFrom() {
		std::lock_guard lock(m_mutex);
		return m_timed_from;
	}

	// GPU thread time the timed part of the pass spent on the player's own work: putting
	// recorded guest memory in place and hashing pictures. A live run does not pay it here.
	[[nodiscard]] double PlayerMilliseconds() const {
		return std::chrono::duration<double, std::milli>(
		           std::chrono::steady_clock::duration(m_player_time.load()))
		    .count();
	}

	void OnSubmissionStart(SubmissionKind /*kind*/, uint32_t /*queue*/,
	                       std::span<const uint32_t> /*commands*/,
	                       std::span<const uint32_t> /*constant_commands*/) override {
		const MemoryUpdate* update = nullptr;
		{
			std::lock_guard lock(m_mutex);
			update = std::exchange(m_pending_update, nullptr);
		}
		m_stalled.Progress();
		EnsureLiveBuffers();
		Apply(update);
		m_last_progress = std::chrono::steady_clock::now();
		{
			std::lock_guard lock(m_mutex);
			m_started = true;
		}
		m_changed.notify_all();
	}

	void OnFlip(int handle, int index) override {
		m_flipped     = true;
		m_flip_handle = handle;
		m_flip_index  = index;
	}

	void OnSuspendPoint(uint32_t /*frame_id*/) override {
		m_stalled.Progress();
		const MemoryUpdate* update = nullptr;
		bool                first  = false;
		{
			std::lock_guard lock(m_mutex);
			if (!m_frame_end_updates.empty()) {
				update = m_frame_end_updates.front();
				m_frame_end_updates.pop_front();
			}
			first = m_frames.empty();
		}
		Apply(update);
		// The check of this frame end, and any check before it that the replay did not reach
		// by its count of draws.
		while (m_next_check < m_checks.size()) {
			const auto& check = m_checks[m_next_check++];
			if (check.position == FrameEndCheck) {
				CheckImages(check, true);
				break;
			}
		}
		ReplayedFrame frame;
		frame.flipped = std::exchange(m_flipped, false);
		if (frame.flipped) {
			const WorkTimer timer(m_player_time);
			frame.picture.handle = m_flip_handle;
			frame.picture.index  = m_flip_index;
			std::vector<uint8_t> pixels;
			frame.picture.available = HashDisplayBuffer(m_renderer, m_flip_handle, m_flip_index,
			                                            frame.picture.hash, &pixels)
			                              ? 1
			                              : 0;
			if (m_frames_hashed < m_live_pictures.size()) {
				const auto& live = m_live_pictures[m_frames_hashed];
				if (frame.picture.available != 0 && live.picture.available != 0 &&
				    live.picture.hash != frame.picture.hash && !live.bytes.empty()) {
					frame.difference   = ComparePictures(live.bytes, pixels);
					frame.picture_size = pixels.size();
				}
			}
		}
		m_frames_hashed++;
		frame.work = g_work.Take();
		// The first frame of a pass uploads every picture the GPU wrote before the capture. The
		// frames after it are the ones to time, so the host finishes that work first.
		const bool start_timing = first && m_frame_count > 1;
		if (start_timing) {
			FinishHostWork(m_renderer);
			m_player_time.store(0);
		}
		m_last_progress = std::chrono::steady_clock::now();
		{
			std::lock_guard lock(m_mutex);
			if (start_timing) {
				m_timed_from = m_last_progress;
			}
			m_frames.push_back(frame);
		}
		m_changed.notify_all();
	}

	// A wait that passed at once in the live run had its memory in place when it was reached.
	// Put it in place before the test, so the wait does not suspend the submission here either:
	// each suspension flushes the commands recorded so far.
	void OnBeforeWait(uint64_t address, uint32_t /*size*/) override {
		std::optional<size_t> wait;
		{
			std::lock_guard lock(m_mutex);
			wait = m_schedule.BeforeTest(address);
		}
		if (wait) {
			Apply(m_waits[*wait].update);
		}
	}

	// A wait failed its test. Once the replay is where the live run saw this wait pass, give it
	// what the live run had: the guest memory of that moment, and the awaited bytes. The bytes
	// are only written when nobody else will: the host GPU has finished (an end-of-pipe write
	// may be on its way), and no other queue of the GPU thread can run (one of them may be on
	// its way to the command that writes them).
	void OnWaitBlocked(uint64_t address, uint32_t /*size*/) override {
		std::optional<size_t> wait;
		{
			std::lock_guard lock(m_mutex);
			wait = m_schedule.Blocked(address);
		}
		if (!wait) {
			StopIfStuck(address);
			return;
		}
		Apply(m_waits[*wait].update);
		const auto bytes = m_waits[*wait].bytes;
		if (BytesAre(address, bytes)) {
			return;
		}
		FinishHostWork(m_renderer);
		if (BytesAre(address, bytes)) {
			return;
		}
		if (!m_stalled.NothingElseRan(*wait)) {
			return;
		}
		ApplyChangedBytes(address, bytes);
		m_last_progress = std::chrono::steady_clock::now();
	}

	void OnWaitPassed(uint64_t address, uint32_t /*size*/, bool /*had_blocked*/) override {
		std::optional<size_t> wait;
		{
			std::lock_guard lock(m_mutex);
			wait = m_schedule.Passed(address);
		}
		m_stalled.Progress();
		// The commands after the wait read what the guest wrote before it.
		if (wait) {
			Apply(m_waits[*wait].update);
		}
		m_last_progress = std::chrono::steady_clock::now();
	}

	// The live run told the caches about a guest write while the renderer was at work, and what
	// the next draws upload depends on it. So a recorded write is replayed after as many draws
	// and dispatches as the live run had made when it came, at the next point between two
	// commands: that is where the live run serves a write, too.
	void OnHostWork() override {
		m_stalled.Progress();
		m_work_since_update++;
		m_work_in_pass++;
	}

	void OnBetweenCommands() override {
		while (m_next_check < m_checks.size() &&
		       m_checks[m_next_check].position <= m_work_in_pass) {
			CheckImages(m_checks[m_next_check++], false);
		}
		EnsureLiveBuffers();
		while (m_upcoming != nullptr && m_upcoming->read_point &&
		       m_upcoming->position <= m_work_since_update) {
			ApplyUpcoming();
		}
		if (m_upcoming != nullptr) {
			ApplyWrites(*m_upcoming, m_work_since_update);
		}
	}

	void OnGuestRead(uint64_t /*address*/, uint64_t /*size*/) override {}

	void OnOtherBlock(const char* /*what*/) override {}
	void OnGuestWrite(WriteTarget /*target*/, uint64_t /*address*/,
	                  uint64_t /*size*/) noexcept override {}

	// GPU thread. Writes the bytes of `wanted` that differ from guest memory, as a CPU write.
	// Bytes inside `host_written` are written as the renderer's own copy from the host GPU: no
	// cache is told.
	void ApplyChangedBytes(uint64_t address, std::span<const uint8_t> wanted,
	                       const RangeSet* host_written = nullptr) {
		const WorkTimer timer(m_player_time);
		// Runs of differing bytes closer than this are written as one run.
		constexpr size_t MergeGap = 64;
		const auto       read     = [&] {
			m_current.resize(wanted.size());
			if (!Memory::TryReadBacking(address, m_current.data(), wanted.size())) {
				EXIT("GPU replay: page 0x%016" PRIx64 " is not mapped\n", address);
			}
		};
		read();
		if (std::memcmp(m_current.data(), wanted.data(), wanted.size()) == 0) {
			return;
		}
		// Telling the buffer cache about a write makes it read back bytes the host GPU holds,
		// which changes guest memory. Let that happen before the bytes are compared.
		if ((host_written == nullptr || !host_written->Intersects(address, wanted.size())) &&
		    m_renderer.GetBufferCache().IsRegionGpuModified(address, wanted.size())) {
			m_renderer.GetBufferCache().InvalidateMemory(address, wanted.size());
			read();
		}
		ForEachChangedRun(m_current, wanted, MergeGap, [&](size_t offset, size_t size) {
			// First let the caches mark the run CPU written, then put the bytes in place.
			uint64_t told = address + offset;
			if (host_written != nullptr) {
				host_written->ForEachInRange(told, size, [&](uint64_t begin, uint64_t end) {
					if (told < begin) {
						(void)m_renderer.InvalidateMemory(told, begin - told);
					}
					told = end;
				});
			}
			if (told < address + offset + size) {
				(void)m_renderer.InvalidateMemory(told, address + offset + size - told);
			}
			Memory::WriteBacking(address + offset, wanted.data() + offset, size);
		});
	}

	// GPU thread. Does to guest memory and to the caches what the live run did before one point
	// of the stream. The recorded writes go first: they are what the caches were told, and the
	// reason the live run uploaded more than the bytes that changed.
	//
	// Updates are applied in the order of the live run, each once in a pass. The replay can
	// reach two points of the stream in the other order, when they are in different queues. The
	// update of the later point then brings the earlier ones with it: an update holds whole
	// pages, so applying an older one after a newer one would bring back old bytes.
	void Apply(const MemoryUpdate* update) {
		while (update != nullptr && m_upcoming != nullptr && m_upcoming->order <= update->order) {
			ApplyUpcoming();
		}
	}

	// GPU thread, at the start of a pass: the first update is the upcoming one, and the buffers
	// of the live run are due.
	void StartUpdates(const MemoryUpdate* first, bool check_images) {
		m_stalled.Reset();
		m_next_check        = check_images ? 0 : m_checks.size();
		m_work_in_pass      = 0;
		m_live_buffers_due  = true;
		m_upcoming          = first;
		m_next_write        = 0;
		m_work_since_update = 0;
	}

private:
	// GPU thread. Takes the check the live run took at this point of its work, and names the
	// images whose bytes differ. Only the first checks that find a difference are reported: what
	// differs later follows from them.
	void CheckImages(const ImageChecks& live, bool frame_end) {
		constexpr uint32_t MaxReports = 3;
		constexpr size_t   MaxImages  = 40;
		// A check inside a frame can differ although nothing is wrong: the live run and the
		// replay need not have given the same share of the work to each queue by then. So the
		// checks inside a frame stop after the first reports; the ones at frame ends go on.
		if (!frame_end && m_check_reports >= MaxReports) {
			return;
		}
		const WorkTimer timer(m_player_time);
		const auto      now = m_renderer.GetTextureCache().HashGpuImages();
		std::unordered_map<uint64_t, const ImageCheck*> expected;
		for (const auto& image: live.images) {
			expected[image.address] = &image;
		}
		std::vector<std::string> lines;
		size_t                   compared = 0;
		for (const auto& image: now) {
			const auto found = expected.find(image.range.address);
			if (found == expected.end() || found->second->size != image.range.size) {
				continue;
			}
			compared++;
			if (found->second->hash != image.hash) {
				lines.push_back(fmt::format(
				    "image 0x{:x} {}x{}x{} {} ({})", image.range.address, image.width, image.height,
				    image.depth, vk::to_string(image.format), Megabytes(image.range.size)));
			}
		}
		if (frame_end) {
			Report(fmt::format("frame end {}: {} of the {} images the GPU wrote in both runs "
			                   "differ from the live run",
			                   ++m_frame_end_checks, lines.size(), compared));
		} else if (!lines.empty()) {
			m_check_reports++;
			Report(fmt::format("after {} draws and dispatches: {} of the {} images the GPU wrote "
			                   "in both runs differ from the live run{}",
			                   live.position, lines.size(), compared,
			                   m_check_reports == MaxReports
			                       ? "; later checks inside a frame are not shown"
			                       : ""));
		}
		for (size_t i = 0; i < std::min(lines.size(), MaxImages); i++) {
			Report("  " + lines[i]);
		}
	}

	// GPU thread. Applies the upcoming update and makes the one after it the upcoming one.
	void ApplyUpcoming() {
		const auto& update = *m_upcoming;
		if (update.mapping != nullptr && !m_mapping.Set(*update.mapping)) {
			EXIT("GPU replay: the ranges the capture maps cannot be mapped\n");
		}
		ApplyWrites(update, UINT32_MAX);
		const RangeSet* host_written = update.host_written.Empty() ? nullptr : &update.host_written;
		for (const auto* record: update.pages) {
			(void)Reader::ForEachPage(
			    *record, [this, host_written](uint64_t address, std::span<const uint8_t> bytes) {
				    ApplyChangedBytes(address, bytes, host_written);
			    });
		}
		m_upcoming          = update.next;
		m_next_write        = 0;
		m_work_since_update = 0;
		// Writes that came before the next draw are due now.
		if (m_upcoming != nullptr) {
			ApplyWrites(*m_upcoming, 0);
		}
	}

	// GPU thread. The live run had a buffer for each of these ranges at the capture start, and
	// kept them up to date whether the captured frames use them or not. A buffer can only be
	// made once the renderer records commands, which a new process does at its first draw.
	void EnsureLiveBuffers() {
		if (m_live_buffers_due && m_renderer.GetCommandScheduler().Active()) {
			m_live_buffers_due = false;
			m_renderer.GetBufferCache().EnsureBuffers(m_live_buffers);
		}
	}

	// GPU thread. Tells the caches about the writes of `update` that came up to `position`, and
	// have not been replayed yet.
	void ApplyWrites(const MemoryUpdate& update, uint32_t position) {
		if (m_next_write >= update.writes.size() ||
		    update.writes[m_next_write].position > position) {
			return;
		}
		const WorkTimer timer(m_player_time);
		for (; m_next_write < update.writes.size() &&
		       update.writes[m_next_write].position <= position;
		     m_next_write++) {
			const auto& write = update.writes[m_next_write];
			if (!m_renderer.IsMapped(write.address, write.size)) {
				continue;
			}
			if ((write.target & static_cast<uint32_t>(WriteTarget::Buffers)) != 0) {
				m_renderer.GetBufferCache().InvalidateMemory(write.address, write.size);
			}
			if ((write.target & static_cast<uint32_t>(WriteTarget::Images)) != 0) {
				m_renderer.GetTextureCache().InvalidateMemory(write.address, write.size);
			}
		}
	}

	// GPU thread, from a wait that keeps failing. A replay has no guest to release a wait the
	// capture does not account for, so it would block for ever: stop instead. Only time in which
	// the GPU thread did nothing but retry counts: retries come every fraction of a millisecond,
	// so a longer gap means it was busy with other work, a shader compile for example.
	void StopIfStuck(uint64_t address) {
		constexpr auto Limit   = std::chrono::seconds(30);
		constexpr auto BusyGap = std::chrono::seconds(1);
		const auto     now     = std::chrono::steady_clock::now();
		if (now - m_last_retry > BusyGap) {
			m_last_progress = now;
		}
		m_last_retry = now;
		if (now - m_last_progress < Limit) {
			return;
		}
		Report(fmt::format("stuck: nothing releases the wait on 0x{:016x}; stopping", address));
		std::quick_exit(1);
	}

	RenderContext&                  m_renderer;
	GuestMapping&                   m_mapping;
	const std::vector<RecordedWait> m_waits;
	std::vector<WaitSchedule::Wait> m_schedule_waits;
	const std::vector<GuestRange>   m_live_buffers;
	const std::vector<LivePicture>  m_live_pictures;
	const std::vector<ImageChecks>  m_checks;
	StalledWaits                    m_stalled; // GPU thread only
	// GPU thread only: the next check to take, the draws and dispatches of the pass so far, and
	// the checks that were reported.
	size_t                                m_next_check       = 0;
	uint64_t                              m_work_in_pass     = 0;
	uint32_t                              m_check_reports    = 0;
	uint32_t                              m_frame_end_checks = 0;
	size_t                                m_frames_hashed    = 0; // GPU thread only during a pass
	bool                                  m_live_buffers_due = false; // GPU thread only
	const uint32_t                        m_frame_count;
	std::chrono::steady_clock::time_point m_last_retry = std::chrono::steady_clock::now();
	// GPU thread only: the last time a submission started or a wait was served.
	std::chrono::steady_clock::time_point m_last_progress = std::chrono::steady_clock::now();
	std::atomic<int64_t>                  m_player_time {0};
	std::vector<uint8_t>                  m_current;
	// GPU thread only: the update whose writes the draws are replaying, the first of its
	// writes that is still due, and the draws and dispatches since the update before it.
	const MemoryUpdate* m_upcoming          = nullptr;
	size_t              m_next_write        = 0;
	uint32_t            m_work_since_update = 0;
	bool                m_flipped           = false;
	int                 m_flip_handle       = 0;
	int                 m_flip_index        = 0;

	std::mutex                            m_mutex;
	std::condition_variable               m_changed;
	WaitSchedule                          m_schedule;
	const MemoryUpdate*                   m_pending_update = nullptr;
	bool                                  m_started        = false;
	std::deque<const MemoryUpdate*>       m_frame_end_updates;
	std::vector<ReplayedFrame>            m_frames;
	std::chrono::steady_clock::time_point m_timed_from = std::chrono::steady_clock::now();
};

bool RestoreShaders(const Reader::Record& record) {
	if (record.payload.size() % sizeof(ShaderEntry) != 0) {
		return false;
	}
	for (size_t offset = 0; offset < record.payload.size(); offset += sizeof(ShaderEntry)) {
		ShaderEntry entry;
		std::memcpy(&entry, record.payload.data() + offset, sizeof(entry));
		ShaderMapRecord shader;
		shader.address        = entry.address;
		shader.hash           = entry.hash;
		shader.data.type      = static_cast<decltype(shader.data.type)>(entry.type);
		shader.data.user_data = reinterpret_cast<decltype(shader.data.user_data)>(entry.user_data);
		shader.data.input_semantics =
		    reinterpret_cast<decltype(shader.data.input_semantics)>(entry.input_semantics);
		shader.data.num_input_semantics = entry.num_input_semantics;
		shader.data.code_size_bytes     = entry.code_size_bytes;
		shader.data.scratch_size_dwords = entry.scratch_size_dwords;
		ShaderMapRestore(shader);
	}
	return true;
}

std::span<const uint32_t> GuestCommands(uint64_t address, uint64_t dwords) {
	return {reinterpret_cast<const uint32_t*>(address), static_cast<size_t>(dwords)};
}

const char* OpenFailure(Reader::OpenResult result) {
	switch (result) {
		case Reader::OpenResult::Missing: return "cannot be opened";
		case Reader::OpenResult::NotACapture: return "is not a capture of the guest GPU stream";
		case Reader::OpenResult::OtherVersion:
			return "is a capture in another file format version; record it again with this build";
		case Reader::OpenResult::OtherLayout:
			return "was recorded by a build whose command processor state has another layout; "
			       "record it again with this build";
		case Reader::OpenResult::Ok: break;
	}
	return "";
}

} // namespace

bool Play(RenderContext& renderer, const std::filesystem::path& path, uint32_t loops) {
	Reader reader;
	if (const auto result = reader.Open(path, CurrentStateLayout());
	    result != Reader::OpenResult::Ok) {
		Report(fmt::format("{} {}", path.string(), OpenFailure(result)));
		return false;
	}
	auto& gpu = renderer.GetGpu();
	// Never destroyed, like the observer that uses it.
	auto& mapping = *new GuestMapping(renderer);

	// State, up to the first frame.
	std::vector<GuestGpu::StartedSubmission>                                 started_submissions;
	std::vector<std::pair<uint32_t, std::unique_ptr<CommandProcessorState>>> processor_states;
	std::vector<TextureCache::SurfaceMeta>                                   surface_metas;
	std::vector<uint8_t>                                                     gds;
	std::vector<GuestRange>                                                  cached_buffers;
	// The guest memory of the capture start. Every pass starts from it.
	std::vector<Reader::Record> initial_pages;
	std::vector<Range>          ranges;
	bool                        mapped    = false;
	bool                        in_frames = false;
	Reader::Record              record;
	while (!in_frames && reader.Next(record)) {
		switch (record.type) {
			case RecordType::VideoOut: {
				VideoOutCall call;
				if (!record.As(call) || !VideoOut::VideoOutRestoreCall(call)) {
					Report("a video-out call could not be restored");
					return false;
				}
				break;
			}
			case RecordType::ProcessorState: {
				ProcessorHeader header;
				if (!record.As(header) || header.state_size != sizeof(CommandProcessorState) ||
				    record.payload.size() != sizeof(header) + sizeof(CommandProcessorState)) {
					Report("damaged command processor record");
					return false;
				}
				auto state = std::make_unique<CommandProcessorState>();
				std::memcpy(state.get(), record.payload.data() + sizeof(header), sizeof(*state));
				processor_states.emplace_back(header.queue_index, std::move(state));
				break;
			}
			case RecordType::StartedSubmission: {
				GuestGpu::StartedSubmission started;
				if (!record.As(started) || record.payload.size() != sizeof(started)) {
					Report("damaged record of a suspended submission");
					return false;
				}
				started_submissions.push_back(started);
				break;
			}
			case RecordType::Gds: gds = record.payload; break;
			case RecordType::SurfaceMetas:
				if (!Reader::ForEach<SurfaceMeta>(record, [&](const SurfaceMeta& meta) {
					    surface_metas.push_back({meta.address, meta.type, meta.clear_mask});
				    })) {
					Report("damaged surface metadata record");
					return false;
				}
				break;
			case RecordType::Shaders:
				if (!RestoreShaders(record)) {
					Report("damaged shader record");
					return false;
				}
				break;
			case RecordType::Buffers:
				if (!Reader::ForEach<Range>(record, [&](const Range& buffer) {
					    cached_buffers.push_back({buffer.address, buffer.size});
				    })) {
					Report("damaged buffer record");
					return false;
				}
				break;
			case RecordType::MapRange: {
				Range range;
				if (!record.As(range) || mapped) {
					Report("damaged range record");
					return false;
				}
				ranges.push_back(range);
				break;
			}
			case RecordType::Pages: {
				if (!mapped) {
					if (!mapping.Set(ranges)) {
						return false;
					}
					mapped = true;
				}
				const bool ok = Reader::ForEachPage(
				    record, [](uint64_t address, std::span<const uint8_t> bytes) {
					    Memory::WriteBacking(address, bytes.data(), bytes.size());
				    });
				if (!ok) {
					Report("damaged page record");
					return false;
				}
				initial_pages.push_back(std::move(record));
				break;
			}
			case RecordType::FrameBegin: in_frames = true; break;
			default: Report("unexpected record before the first frame"); return false;
		}
	}
	if (!in_frames) {
		Report("the capture has no frames");
		return false;
	}
	if (!mapped && !mapping.Set(ranges)) {
		return false;
	}

	// The frames are replayed more than once, so keep them in memory.
	std::vector<Reader::Record> frames;
	bool                        complete    = false;
	uint32_t                    frame_count = 0;
	while (reader.Next(record)) {
		if (record.type == RecordType::End) {
			complete = true;
			break;
		}
		frame_count += record.type == RecordType::FrameEnd ? 1 : 0;
		frames.push_back(std::move(record));
	}
	if (!complete || reader.Truncated() || reader.Damaged() || frame_count == 0) {
		Report("the capture is incomplete or damaged (it was cut short or abandoned)");
		return false;
	}
	reader.Close();

	// A memory record belongs to the submission, the wait or the frame end that follows it. The
	// GPU thread applies it when the replay reaches that point.
	std::vector<RecordType> types;
	types.reserve(frames.size());
	for (const auto& item: frames) {
		types.push_back(item.type);
	}
	const auto                owners = MemoryRecordOwners(types);
	std::vector<MemoryUpdate> updates(frames.size()); // indexed by the owning record
	// The mapped ranges after each change inside the frames. A deque: the updates point into it.
	std::deque<std::vector<Range>> mappings;
	// Pages the frames write that were all zero at the capture start, so are not in the state.
	std::vector<uint64_t> zero_pages;
	{
		std::unordered_set<uint64_t> known;
		for (const auto& pages: initial_pages) {
			(void)Reader::ForEachPage(
			    pages, [&](uint64_t address, std::span<const uint8_t>) { known.insert(address); });
		}
		const auto note_page = [&](uint64_t address) {
			if (known.insert(address).second) {
				zero_pages.push_back(address);
			}
		};
		for (size_t i = 0; i < frames.size(); i++) {
			const auto& item = frames[i];
			if (item.type == RecordType::WaitBytes) {
				BytesHeader header;
				if (item.As(header) && header.size != 0) {
					note_page(header.address & ~(PageSize - 1));
					note_page((header.address + header.size - 1) & ~(PageSize - 1));
				}
			}
			if (owners[i] == NoOwner) {
				continue;
			}
			if (item.type == RecordType::Pages) {
				if (!Reader::ForEachPage(item, [&](uint64_t address, std::span<const uint8_t>) {
					    note_page(address);
				    })) {
					Report("damaged page record");
					return false;
				}
				updates[owners[i]].pages.push_back(&item);
			} else if (item.type == RecordType::Mapping) {
				auto& stored = mappings.emplace_back();
				if (!Reader::ForEach<Range>(item,
				                            [&](const Range& range) { stored.push_back(range); })) {
					Report("damaged mapping record");
					return false;
				}
				updates[owners[i]].mapping = &stored;
			} else if (!Reader::ForEach<WriteRange>(item, [&](const WriteRange& write) {
				           updates[owners[i]].writes.push_back(write);
			           })) {
				Report("damaged record of guest writes");
				return false;
			}
		}
	}
	// A copy from the host GPU to guest memory and the scan that stores its bytes are not tied
	// to each other: the bytes can be in the update before or after the one that names the
	// range. So an update takes the ranges of its neighbours, too.
	const auto share_host_writes = [](MemoryUpdate& from, MemoryUpdate& to) {
		for (const auto& write: from.writes) {
			if (write.target == static_cast<uint32_t>(WriteTarget::Host) && write.size != 0) {
				to.host_written.Add(write.address, write.size);
			}
		}
	};
	// The live run stored guest memory at these points, in the order of their records.
	const MemoryUpdate* first_update = nullptr;
	for (size_t i = 0, last = NoOwner; i < frames.size(); i++) {
		if (!IsMemoryPoint(frames[i].type)) {
			continue;
		}
		if (frames[i].type == RecordType::ReadPoint) {
			ReadPoint point;
			if (!frames[i].As(point)) {
				Report("damaged read record");
				return false;
			}
			updates[i].read_point = true;
			updates[i].position   = point.position;
		}
		share_host_writes(updates[i], updates[i]);
		if (last != NoOwner) {
			updates[last].next = &updates[i];
			updates[i].order   = updates[last].order + 1;
			share_host_writes(updates[last], updates[i]);
			share_host_writes(updates[i], updates[last]);
		} else {
			first_update = &updates[i];
		}
		last = i;
	}
	std::vector<ImageChecks>                  image_checks;
	std::vector<PlayerObserver::RecordedWait> waits;
	std::vector<FrameWork>                    recorded_work(frame_count);
	std::vector<PlayerObserver::LivePicture>  recorded_pictures(frame_count);
	for (size_t i = 0, frame = 0; i < frames.size(); i++) {
		const auto& item = frames[i];
		bool        ok   = true;
		if (frame >= frame_count) {
			ok = false; // a record after the last frame end
		} else if (item.type == RecordType::WaitBytes) {
			PlayerObserver::RecordedWait wait;
			ok          = Reader::Bytes(item, wait.address, wait.bytes, wait.had_blocked);
			wait.update = &updates[i];
			waits.push_back(wait);
		} else if (item.type == RecordType::Work) {
			ok = item.As(recorded_work[frame]);
		} else if (item.type == RecordType::Picture) {
			ok = item.As(recorded_pictures[frame].picture);
		} else if (item.type == RecordType::PictureBytes) {
			recorded_pictures[frame].bytes = item.payload;
		} else if (item.type == RecordType::ImageChecks) {
			ImageChecksHeader header;
			ok =
			    item.As(header) && (item.payload.size() - sizeof(header)) % sizeof(ImageCheck) == 0;
			if (ok) {
				auto& checks    = image_checks.emplace_back();
				checks.position = header.position;
				checks.images.resize((item.payload.size() - sizeof(header)) / sizeof(ImageCheck));
				std::memcpy(checks.images.data(), item.payload.data() + sizeof(header),
				            checks.images.size() * sizeof(ImageCheck));
			}
		} else if (item.type == RecordType::FrameEnd) {
			frame++;
		}
		if (!ok) {
			Report("damaged record inside a frame");
			return false;
		}
	}

	// Never destroyed: when this function returns early, the GPU thread can still be inside a
	// call, and the process ends right after.
	auto& observer =
	    *new PlayerObserver(renderer, mapping, std::move(waits), std::move(cached_buffers),
	                        recorded_pictures, std::move(image_checks));
	renderer.SetObserver(&observer);
	Report(fmt::format("{} frame(s), {} pass(es)", frame_count, loops));
	if (frame_count == 1) {
		Report("the capture has one frame, so its time includes the upload of every picture the "
		       "GPU wrote before the capture; record two or more frames to time the later ones");
	}

	static const std::array<uint8_t, PageSize> zero_page {};
	std::vector<ReplayedFrame>                 first_pass;
	for (uint32_t loop = 0; loop < loops; loop++) {
		// Every pass starts from the state of the capture start. What the host GPU wrote in the
		// pass before is dropped: guest memory is put back, and every image and buffer the GPU
		// wrote is loaded from it again on first use.
		gpu.SendCommandSync([&] {
			if (loop != 0) {
				FinishHostWork(renderer);
				// The ranges of the capture start come back first: the memory that is put back
				// below is written through them.
				if (!mapping.Set(ranges)) {
					EXIT("GPU replay: the ranges of the capture start cannot be mapped again\n");
				}
				renderer.GetBufferCache().DiscardGpuWrites();
				const auto kept = renderer.GetTextureCache().ReloadGpuWrittenImages();
				if (kept != 0 && loop == 1) {
					Report(fmt::format("{} image(s) cannot be loaded from guest memory and keep "
					                   "the picture of the pass before",
					                   kept));
				}
				for (const auto& pages: initial_pages) {
					(void)Reader::ForEachPage(
					    pages, [&](uint64_t address, std::span<const uint8_t> bytes) {
						    observer.ApplyChangedBytes(address, bytes);
					    });
				}
				for (const auto address: zero_pages) {
					observer.ApplyChangedBytes(address, zero_page);
				}
			}
			// The image checks are taken in the first pass, the one that starts as the live
			// run did.
			observer.StartUpdates(first_update, loop == 0);
			renderer.GetTextureCache().RestoreSurfaceMetas(surface_metas);
			renderer.GetBufferCache().RestoreGds(gds);
			for (const auto& [queue_index, state]: processor_states) {
				gpu.LoadProcessorState(queue_index, *state);
			}
		});
		observer.BeginPass();
		for (const auto& saved: started_submissions) {
			gpu.RestoreStartedSubmission(saved);
		}
		size_t wait_index = 0;
		for (size_t i = 0; i < frames.size(); i++) {
			const auto& item = frames[i];
			switch (item.type) {
				case RecordType::Pages:
				case RecordType::GuestWrites:
				case RecordType::Mapping:
				case RecordType::ReadPoint:
				case RecordType::Picture:
				case RecordType::PictureBytes:
				case RecordType::ImageChecks:
				case RecordType::Work: break;
				case RecordType::Shaders:
					if (!RestoreShaders(item)) {
						Report("damaged shader record");
						return false;
					}
					break;
				case RecordType::Submission: {
					Submission submission;
					if (!item.As(submission)) {
						Report("damaged submission record");
						return false;
					}
					observer.ExpectSubmission(&updates[i]);
					const auto commands =
					    GuestCommands(submission.commands_address, submission.commands_dwords);
					if (submission.kind == static_cast<uint32_t>(SubmissionKind::Compute)) {
						gpu.SubmitCompute(submission.queue, commands);
					} else {
						gpu.Submit(commands, GuestCommands(submission.constant_commands_address,
						                                   submission.constant_commands_dwords));
					}
					// The next submission may only be queued once this one has started: that
					// keeps the recorded start order across the queues.
					observer.WaitUntilStarted();
					break;
				}
				case RecordType::WaitBytes: observer.ReleaseWait(wait_index++); break;
				case RecordType::FrameEnd:
					observer.QueueFrameEnd(&updates[i]);
					gpu.SuspendPoint();
					break;
				default: Report("unexpected record inside a frame"); return false;
			}
		}
		// The pass is over when its last frame end has been processed. Submissions that are
		// suspended at that point wait for a frame the capture does not have: drop them.
		const auto replayed = observer.WaitForFrames();
		gpu.DropStartedSubmissions();
		gpu.SendCommandSync([&renderer] { FinishHostWork(renderer); });
		const auto ended = std::chrono::steady_clock::now();

		// Time: the frames after the first one, without the player's own work.
		const uint32_t first_timed = frame_count > 1 ? 1 : 0;
		const uint32_t timed       = frame_count - first_timed;
		const double   total_ms =
		    std::chrono::duration<double, std::milli>(ended - observer.TimedFrom()).count() / timed;
		const double player_ms = observer.PlayerMilliseconds() / timed;
		Report(fmt::format("pass {}: {:.1f} ms per frame over {} frame(s){}; the player's own "
		                   "work, {:.1f} ms per frame, is not in it",
		                   loop + 1, total_ms - player_ms, timed,
		                   first_timed != 0 ? ", the first frame not counted" : "", player_ms));

		// Work: what the renderer handed to the host GPU in the timed frames, against the live
		// run.
		WorkCounts now;
		FrameWork  live;
		for (uint32_t i = first_timed; i < frame_count; i++) {
			const auto& work = replayed[i].work;
			now.draws += work.draws;
			now.dispatches += work.dispatches;
			now.render_passes += work.render_passes;
			now.buffer_upload_bytes += work.buffer_upload_bytes;
			now.image_upload_bytes += work.image_upload_bytes;
			live.draws += recorded_work[i].draws;
			live.dispatches += recorded_work[i].dispatches;
			live.render_passes += recorded_work[i].render_passes;
			live.buffer_upload_bytes += recorded_work[i].buffer_upload_bytes;
			live.image_upload_bytes += recorded_work[i].image_upload_bytes;
		}
		Report(fmt::format(
		    "pass {}: work in those frames, replay / live run: draws {} / {}, dispatches {} / {}, "
		    "render passes {} / {}, buffer uploads {} / {}, image uploads {} / {}",
		    loop + 1, now.draws, live.draws, now.dispatches, live.dispatches, now.render_passes,
		    live.render_passes, Megabytes(now.buffer_upload_bytes),
		    Megabytes(live.buffer_upload_bytes), Megabytes(now.image_upload_bytes),
		    Megabytes(live.image_upload_bytes)));

		// Pictures: the display buffer of each frame against the live run's, and against the
		// first pass, which tells whether the replay repeats itself.
		uint32_t compared      = 0;
		uint32_t same_as_live  = 0;
		uint32_t same_as_first = 0;
		for (uint32_t i = 0; i < frame_count; i++) {
			const auto& live_picture = recorded_pictures[i].picture;
			const auto& picture      = replayed[i].picture;
			if (!replayed[i].flipped || picture.available == 0 || live_picture.available == 0) {
				continue;
			}
			compared++;
			if (live_picture.handle != picture.handle || live_picture.index != picture.index) {
				Report(fmt::format("pass {} frame {}: ANOTHER display buffer was flipped", loop + 1,
				                   i + 1));
			} else if (live_picture.hash == picture.hash) {
				same_as_live++;
			} else if (replayed[i].picture_size != 0) {
				const auto& difference = replayed[i].difference;
				Report(fmt::format(
				    "pass {} frame {}: picture DIFFERS from the live run: {:.3f} % of its bytes, "
				    "by at most {} of 255",
				    loop + 1, i + 1,
				    100.0 * static_cast<double>(difference.bytes) /
				        static_cast<double>(replayed[i].picture_size),
				    difference.largest));
			} else {
				Report(fmt::format("pass {} frame {}: picture DIFFERS from the live run", loop + 1,
				                   i + 1));
			}
			if (loop != 0 && first_pass[i].picture.hash == picture.hash) {
				same_as_first++;
			}
		}
		Report(fmt::format(
		    "pass {}: {} of {} picture(s) match the live run{}", loop + 1, same_as_live, compared,
		    compared == frame_count ? "" : "; the other frames have no picture hash"));
		if (loop == 0) {
			first_pass = replayed;
		} else {
			Report(fmt::format("pass {}: {} of {} picture(s) are the same as in pass 1", loop + 1,
			                   same_as_first, compared));
		}
	}
	renderer.SetObserver(nullptr);
	return true;
}

} // namespace Libs::Graphics::Capture
