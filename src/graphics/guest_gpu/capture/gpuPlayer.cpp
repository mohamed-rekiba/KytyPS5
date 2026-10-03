#include "graphics/guest_gpu/capture/gpuPlayer.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/capture/captureFile.h"
#include "graphics/guest_gpu/capture/guestMemoryAccess.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/renderContext.h"
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

bool BytesAre(uint64_t address, std::span<const uint8_t> wanted) {
	std::array<uint8_t, 16> current {};
	return wanted.size() <= current.size() &&
	       Memory::TryReadBacking(address, current.data(), wanted.size()) &&
	       std::memcmp(current.data(), wanted.data(), wanted.size()) == 0;
}

// Player side of the GPU thread hooks: applies the memory changes that belong to a submission
// right before it starts, and tells the player thread what the GPU thread is doing.
// Adds the time it lives to a counter of clock ticks.
class MemoryTimer final {
public:
	explicit MemoryTimer(std::atomic<int64_t>& total)
	    : m_total(total), m_started(std::chrono::steady_clock::now()) {}
	~MemoryTimer() { m_total.fetch_add((std::chrono::steady_clock::now() - m_started).count()); }
	MemoryTimer(const MemoryTimer&)            = delete;
	MemoryTimer& operator=(const MemoryTimer&) = delete;

private:
	std::atomic<int64_t>&                 m_total;
	std::chrono::steady_clock::time_point m_started;
};

class PlayerObserver final: public GuestGpuObserver {
public:
	explicit PlayerObserver(RenderContext& renderer): m_renderer(renderer) {}

	// Player thread: the page records to apply when the next submission starts.
	void ExpectSubmission(std::vector<const Reader::Record*> pages) {
		std::lock_guard lock(m_mutex);
		m_pending_pages = std::move(pages);
		m_started       = false;
	}

	void WaitUntilStarted() {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [this] { return m_started; });
	}

	// A wait on guest memory as the live run saw it pass.
	struct RecordedWait {
		uint64_t                 address = 0;
		std::span<const uint8_t> bytes;
		bool                     had_blocked = false;
		// Guest pages that changed in the live run before this wait passed.
		std::vector<const Reader::Record*> pages;
	};

	// Player thread, before a pass: every wait of the pass, in recorded order. The command
	// stream is the same as in the live run, so the waits on one address come in the same order.
	void ArmWaits(std::vector<RecordedWait> waits) {
		std::lock_guard lock(m_mutex);
		m_waits = std::move(waits);
		m_released.assign(m_waits.size(), false);
		m_wait_order.clear();
		for (size_t i = 0; i < m_waits.size(); i++) {
			m_wait_order[m_waits[i].address].push_back(i);
		}
	}

	// Player thread: the replay has reached the point where the live run saw wait `index` pass.
	void ReleaseWait(size_t index) {
		std::lock_guard lock(m_mutex);
		m_released[index] = true;
	}

	// Player thread: the pages that changed in the live run before the next frame end.
	void QueueFrameEndPages(std::vector<const Reader::Record*> pages) {
		std::lock_guard lock(m_mutex);
		m_frame_end_pages.push_back(std::move(pages));
	}

	// Player thread, between passes: hash the display buffer of each frame, or not.
	void CheckPictures(bool check) {
		std::lock_guard lock(m_mutex);
		m_check_pictures = check;
	}

	void OnSubmissionStart(SubmissionKind /*kind*/, uint32_t /*queue*/,
	                       std::span<const uint32_t> /*commands*/,
	                       std::span<const uint32_t> /*constant_commands*/) override {
		std::vector<const Reader::Record*> pages;
		{
			std::lock_guard lock(m_mutex);
			pages.swap(m_pending_pages);
		}
		for (const auto* record: pages) {
			(void)Reader::ForEachPage(*record,
			                          [this](uint64_t address, std::span<const uint8_t> bytes) {
				                          ApplyChangedBytes(address, bytes);
			                          });
		}
		m_last_progress = std::chrono::steady_clock::now();
		{
			std::lock_guard lock(m_mutex);
			m_started = true;
		}
		m_changed.notify_all();
	}

	// The pictures of the frames that ended since the last call, oldest first.
	[[nodiscard]] std::vector<Picture> TakePictures() {
		std::lock_guard lock(m_mutex);
		return std::exchange(m_pictures, {});
	}

	void OnFlip(int handle, int index) override {
		m_flipped     = true;
		m_flip_handle = handle;
		m_flip_index  = index;
	}

	void OnSuspendPoint(uint32_t /*frame_id*/) override {
		bool                               check = false;
		std::vector<const Reader::Record*> pages;
		{
			std::lock_guard lock(m_mutex);
			check = m_check_pictures;
			if (!m_frame_end_pages.empty()) {
				pages = std::move(m_frame_end_pages.front());
				m_frame_end_pages.pop_front();
			}
		}
		for (const auto* record: pages) {
			(void)Reader::ForEachPage(*record,
			                          [this](uint64_t address, std::span<const uint8_t> bytes) {
				                          ApplyChangedBytes(address, bytes);
			                          });
		}
		if (m_flipped && check) {
			Picture picture;
			picture.handle = m_flip_handle;
			picture.index  = m_flip_index;
			picture.available =
			    HashDisplayBuffer(m_renderer, m_flip_handle, m_flip_index, picture.hash) ? 1 : 0;
			std::lock_guard lock(m_mutex);
			m_pictures.push_back(picture);
		}
		m_flipped       = false;
		m_last_progress = std::chrono::steady_clock::now();
		{
			std::lock_guard lock(m_mutex);
			m_frames_ended++;
		}
		m_changed.notify_all();
	}

	// Player thread: starts a pass, and waits until the GPU thread has processed `count` frame
	// ends of it.
	// GPU thread time the pass spent putting recorded guest memory in place. The live run does
	// not pay it on this thread: there the guest's own threads write the memory.
	[[nodiscard]] double MemoryMilliseconds() const {
		return std::chrono::duration<double, std::milli>(
		           std::chrono::steady_clock::duration(m_memory_time.load()))
		    .count();
	}

	void BeginPass() {
		m_memory_time.store(0);
		std::lock_guard lock(m_mutex);
		m_frames_ended = 0;
		m_frame_end_pages.clear();
	}
	void WaitForFrameEnds(uint32_t count) {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [&] { return m_frames_ended >= count; });
	}

	// The first pass keeps the bytes it overwrites. `RestoreMemory`, on the GPU thread, puts them
	// back, so the next pass starts from the guest memory of the capture start.
	void KeepOverwrittenBytes(bool keep) { m_keep_overwritten = keep; }
	void RestoreMemory() {
		const MemoryTimer timer(m_memory_time);
		for (auto entry = m_overwritten.rbegin(); entry != m_overwritten.rend(); ++entry) {
			(void)m_renderer.InvalidateMemory(entry->first, entry->second.size());
			Memory::WriteBacking(entry->first, entry->second.data(), entry->second.size());
		}
	}

	// A wait that passed at once in the live run had its memory in place when it was reached.
	// Put it in place before the test, so the wait does not suspend the submission here either:
	// each suspension flushes the commands recorded so far.
	void OnBeforeWait(uint64_t address, uint32_t /*size*/) override {
		RecordedWait* wait = nullptr;
		{
			std::lock_guard lock(m_mutex);
			const auto      order = m_wait_order.find(address);
			if (order == m_wait_order.end() || order->second.empty()) {
				return;
			}
			wait = &m_waits[order->second.front()];
			if (wait->had_blocked) {
				return;
			}
		}
		ApplyPages(*wait);
	}

	// A wait failed its test. Once the replay is where the live run saw this wait pass, give it
	// what the live run had: the guest memory of that moment, and the awaited bytes. Let the
	// host GPU finish before writing the bytes: an end-of-pipe write may be on its way, and
	// then nothing needs to be written.
	void OnWaitBlocked(uint64_t address, uint32_t /*size*/) override {
		RecordedWait* wait = nullptr;
		{
			std::lock_guard lock(m_mutex);
			const auto      order = m_wait_order.find(address);
			if (order == m_wait_order.end() || order->second.empty()) {
				StopIfStuck(address);
				return;
			}
			const auto index = order->second.front();
			// A wait that never blocked in the live run had its bytes when it was reached.
			if (m_waits[index].had_blocked && !m_released[index]) {
				StopIfStuck(address);
				return;
			}
			wait = &m_waits[index];
		}
		ApplyPages(*wait);
		if (BytesAre(address, wait->bytes)) {
			return;
		}
		auto& scheduler = m_renderer.GetCommandScheduler();
		if (scheduler.Active()) {
			const auto tick = scheduler.CurrentTick();
			scheduler.Finish();
			scheduler.WaitPriorityOperations(tick);
		}
		if (!BytesAre(address, wait->bytes)) {
			ApplyChangedBytes(address, wait->bytes);
		}
		m_last_progress = std::chrono::steady_clock::now();
	}

	void OnWaitPassed(uint64_t address, uint32_t /*size*/, bool /*had_blocked*/) override {
		RecordedWait* wait = nullptr;
		{
			std::lock_guard lock(m_mutex);
			const auto      order = m_wait_order.find(address);
			if (order != m_wait_order.end() && !order->second.empty()) {
				wait = &m_waits[order->second.front()];
				order->second.pop_front();
			}
		}
		// The commands after the wait read what the guest wrote before it.
		if (wait != nullptr) {
			ApplyPages(*wait);
		}
		m_last_progress = std::chrono::steady_clock::now();
	}

	void OnOtherBlock(const char* /*what*/) override {}

	// GPU thread. Writes the bytes of `wanted` that differ from guest memory, as a CPU write.
	void ApplyChangedBytes(uint64_t address, std::span<const uint8_t> wanted) {
		const MemoryTimer timer(m_memory_time);
		// Runs of differing bytes closer than this are written as one run.
		constexpr size_t MergeGap = 64;
		m_current.resize(wanted.size());
		if (!Memory::TryReadBacking(address, m_current.data(), wanted.size())) {
			EXIT("GPU replay: page 0x%016" PRIx64 " is not mapped\n", address);
		}
		size_t i = 0;
		while (i < wanted.size()) {
			if (wanted[i] == m_current[i]) {
				i++;
				continue;
			}
			size_t end  = i + 1;
			size_t same = 0;
			for (size_t j = end; j < wanted.size() && same < MergeGap; j++) {
				if (wanted[j] != m_current[j]) {
					end  = j + 1;
					same = 0;
				} else {
					same++;
				}
			}
			// First let the caches flush what they own of this run and mark it CPU written,
			// then put the bytes in place.
			if (m_keep_overwritten) {
				m_overwritten.emplace_back(
				    address + i, std::vector<uint8_t>(m_current.begin() + static_cast<long>(i),
				                                      m_current.begin() + static_cast<long>(end)));
			}
			(void)m_renderer.InvalidateMemory(address + i, end - i);
			Memory::WriteBacking(address + i, wanted.data() + i, end - i);
			i = end;
		}
	}

private:
	// GPU thread. Each wait gets its pages once.
	void ApplyPages(RecordedWait& wait) {
		for (const auto* record: wait.pages) {
			(void)Reader::ForEachPage(*record,
			                          [this](uint64_t address, std::span<const uint8_t> bytes) {
				                          ApplyChangedBytes(address, bytes);
			                          });
		}
		wait.pages.clear();
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

	std::chrono::steady_clock::time_point m_last_retry = std::chrono::steady_clock::now();
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> m_overwritten;
	bool                                                   m_keep_overwritten = false;
	uint32_t                                               m_frames_ended     = 0;
	std::atomic<int64_t>                                   m_memory_time {0};
	// GPU thread only: the last time a submission started or a wait was served.
	std::chrono::steady_clock::time_point m_last_progress = std::chrono::steady_clock::now();
	RenderContext&                        m_renderer;
	std::mutex                            m_mutex;
	std::condition_variable               m_changed;
	std::vector<const Reader::Record*>    m_pending_pages;
	bool                                  m_started = false;
	std::vector<RecordedWait>             m_waits;
	std::vector<bool>                     m_released;
	std::unordered_map<uint64_t, std::deque<size_t>> m_wait_order;
	bool                                             m_check_pictures = true;
	std::vector<uint8_t>                             m_current;
	std::deque<std::vector<const Reader::Record*>>   m_frame_end_pages;
	std::vector<Picture>                             m_pictures;
	bool                                             m_flipped     = false;
	int                                              m_flip_handle = 0;
	int                                              m_flip_index  = 0;
};

bool MapRanges(const std::vector<Range>& ranges) {
	// Guest mappings are made of 16 KiB pages; neighbours that touch after rounding become one.
	RangeSet set;
	for (const auto& range: ranges) {
		const uint64_t begin = range.address & ~(GuestPage - 1);
		const uint64_t end   = (range.address + range.size + GuestPage - 1) & ~(GuestPage - 1);
		set.Add(begin, end - begin);
	}
	bool ok = true;
	set.ForEach([&ok](uint64_t begin, uint64_t end) {
		if (!ok) {
			return;
		}
		constexpr int ProtCpuGpuReadWrite = 0x33;
		constexpr int MapFixed            = 0x10;
		const auto    size                = end - begin;
		int64_t       physical            = 0;
		void*         address             = reinterpret_cast<void*>(begin);
		if (Memory::KernelAllocateMainDirectMemory(size, GuestPage, 0, &physical) != 0 ||
		    Memory::KernelMapDirectMemory(&address, size, ProtCpuGpuReadWrite, MapFixed, physical,
		                                  GuestPage) != 0 ||
		    reinterpret_cast<uint64_t>(address) != begin) {
			Report(fmt::format("cannot map 0x{:x} bytes at 0x{:016x}", size, begin));
			ok = false;
		}
	});
	return ok;
}

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

} // namespace

bool Play(RenderContext& renderer, const std::filesystem::path& path, uint32_t loops) {
	Reader reader;
	if (!reader.Open(path)) {
		Report(fmt::format("{} is not a capture this build can read", path.string()));
		return false;
	}
	auto& gpu = renderer.GetGpu();

	// State, up to the first frame.
	std::vector<GuestGpu::StartedSubmission>                                 started_submissions;
	std::vector<std::pair<uint32_t, std::unique_ptr<CommandProcessorState>>> processor_states;
	std::vector<Range> ranges;
	bool               mapped = false;
	Reader::Record     record;
	bool               in_frames = false;
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
					Report("the command processor state was written by another build");
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
					Report("a suspended submission was written by another build");
					return false;
				}
				started_submissions.push_back(started);
				break;
			}
			case RecordType::Gds:
				gpu.SendCommandSync(
				    [&renderer, &record] { renderer.GetBufferCache().RestoreGds(record.payload); });
				break;
			case RecordType::Shaders:
				if (!RestoreShaders(record)) {
					Report("damaged shader record");
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
					if (!MapRanges(ranges)) {
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
	if (!mapped && !MapRanges(ranges)) {
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
	if (!complete || reader.Truncated() || frame_count == 0) {
		Report("the capture is incomplete (it was cut short or abandoned)");
		return false;
	}
	reader.Close();

	// Pages belong to the submission or the wait that follows them. The GPU thread applies a
	// wait's pages when the replay reaches that wait, so they are attached to it here.
	std::vector<PlayerObserver::RecordedWait> waits;
	std::unordered_set<const Reader::Record*> wait_pages;
	{
		std::vector<const Reader::Record*> pages;
		for (const auto& item: frames) {
			if (item.type == RecordType::Pages) {
				pages.push_back(&item);
			} else if (item.type == RecordType::Submission || item.type == RecordType::FrameEnd) {
				pages.clear();
			} else if (item.type == RecordType::WaitBytes) {
				PlayerObserver::RecordedWait wait;
				if (!Reader::Bytes(item, wait.address, wait.bytes, wait.had_blocked)) {
					Report("damaged wait record");
					return false;
				}
				wait_pages.insert(pages.begin(), pages.end());
				wait.pages = std::move(pages);
				pages.clear();
				waits.push_back(std::move(wait));
			}
		}
	}

	// Never destroyed: when this function returns early, the GPU thread can still be inside a
	// call, and the process ends right after.
	auto& observer = *new PlayerObserver(renderer);
	gpu.SetObserver(&observer);
	Report(fmt::format("{} frame(s), {} loop(s)", frame_count, loops));

	for (uint32_t loop = 0; loop < loops; loop++) {
		// The first pass starts from the recorded state: it is the one to compare with the live
		// run. Later passes measure time, so they skip the picture check and its GPU read-back.
		observer.CheckPictures(loop == 0);
		const auto                         started = std::chrono::steady_clock::now();
		std::vector<const Reader::Record*> pending_pages;
		std::vector<Picture>               recorded_pictures;
		size_t                             wait_index = 0;
		// Every pass starts from the state of the capture start: processor registers, the guest
		// memory the last pass changed, and the submissions that were suspended at that moment.
		// Images that exist only on the host keep what the last pass left in them.
		observer.BeginPass();
		observer.KeepOverwrittenBytes(loop == 0);
		gpu.SendCommandSync([&] {
			if (loop != 0) {
				observer.RestoreMemory();
			}
			for (const auto& [queue_index, state]: processor_states) {
				gpu.LoadProcessorState(queue_index, *state);
			}
		});
		observer.ArmWaits(waits);
		for (const auto& saved: started_submissions) {
			gpu.RestoreStartedSubmission(saved);
		}
		for (const auto& item: frames) {
			switch (item.type) {
				case RecordType::Pages:
					if (!wait_pages.contains(&item)) {
						pending_pages.push_back(&item);
					}
					break;
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
					observer.ExpectSubmission(std::move(pending_pages));
					pending_pages.clear();
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
				case RecordType::Picture: {
					Picture picture;
					if (!item.As(picture)) {
						Report("damaged picture record");
						return false;
					}
					recorded_pictures.push_back(picture);
					break;
				}
				case RecordType::FrameEnd:
					// Pages stored after the last wait belong to the frame end.
					observer.QueueFrameEndPages(std::move(pending_pages));
					pending_pages.clear();
					gpu.SuspendPoint();
					break;
				default: Report("unexpected record inside a frame"); return false;
			}
		}
		// The pass is over when its last frame end has been processed. Submissions that are
		// suspended at that point wait for a frame the capture does not have: drop them.
		observer.WaitForFrameEnds(frame_count);
		gpu.DropStartedSubmissions();
		gpu.SendCommandSync([&renderer] {
			auto& scheduler = renderer.GetCommandScheduler();
			if (scheduler.Active()) {
				const auto tick = scheduler.CurrentTick();
				scheduler.Finish();
				scheduler.WaitPriorityOperations(tick);
			}
		});
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
		                    std::chrono::steady_clock::now() - started)
		                    .count();
		const double total_ms  = static_cast<double>(us) / 1000.0 / frame_count;
		const double memory_ms = observer.MemoryMilliseconds() / frame_count;
		Report(fmt::format("loop {}: {:.1f} ms per frame, of which {:.1f} ms put recorded guest "
		                   "memory in place; {:.1f} ms without that",
		                   loop + 1, total_ms, memory_ms, total_ms - memory_ms));

		if (loop != 0) {
			continue;
		}
		// Picture check: the display buffer of each frame against the live run's.
		const auto replayed = observer.TakePictures();
		if (replayed.size() != recorded_pictures.size()) {
			Report(fmt::format("{} flip(s) replayed, {} recorded", replayed.size(),
			                   recorded_pictures.size()));
		}
		for (size_t i = 0; i < std::min(replayed.size(), recorded_pictures.size()); i++) {
			const auto& live    = recorded_pictures[i];
			const auto& now     = replayed[i];
			const char* verdict = "no picture hash available";
			if (live.handle != now.handle || live.index != now.index) {
				verdict = "ANOTHER display buffer was flipped";
			} else if (live.available != 0 && now.available != 0) {
				verdict = live.hash == now.hash ? "picture matches the live run"
				                                : "picture DIFFERS from the live run";
			}
			Report(fmt::format("flip {}: {}", i + 1, verdict));
		}
	}
	gpu.SetObserver(nullptr);
	return true;
}

} // namespace Libs::Graphics::Capture
