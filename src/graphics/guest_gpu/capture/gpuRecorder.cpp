#include "graphics/guest_gpu/capture/gpuRecorder.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/capture/guestMemoryAccess.h"
#include "graphics/guest_gpu/capture/stateLayout.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/workCounters.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <thread>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics::Capture {

namespace {

constexpr uint64_t ChunkSize      = 1024 * 1024;
constexpr uint64_t GuestPage      = 0x4000;
constexpr size_t   PagesPerRecord = 256;
constexpr uint32_t MaxAttempts    = 8;
// Guest writes the recorder can hold between two scans of guest memory. A heavy frame has a few
// thousand in total.
constexpr size_t MaxQueuedWrites = 64 * 1024;

bool IsZeroPage(const uint8_t* page) {
	static const std::array<uint8_t, PageSize> zero {};
	return std::memcmp(page, zero.data(), PageSize) == 0;
}

} // namespace

Recorder::Recorder(RenderContext& renderer, std::filesystem::path path, uint32_t first_frame,
                   uint32_t frame_count, std::filesystem::path trigger, uint32_t check_interval)
    : m_renderer(renderer), m_path(std::move(path)), m_trigger(std::move(trigger)),
      m_first_frame(first_frame), m_frame_count(frame_count), m_check_interval(check_interval) {
	m_guest_writes.reserve(MaxQueuedWrites);
	m_taken_writes.reserve(MaxQueuedWrites);
}

void Recorder::OnSuspendPoint(uint32_t frame_id) {
	if (m_state == State::Recording) {
		// What the guest wrote since the last wait belongs to this frame; a display buffer the
		// guest fills from the CPU is part of it.
		if (!WriteChangedPages(false)) {
			return;
		}
		if (m_flipped) {
			Picture picture;
			picture.handle = m_flip_handle;
			picture.index  = m_flip_index;
			std::vector<uint8_t> pixels;
			picture.available =
			    HashDisplayBuffer(m_renderer, m_flip_handle, m_flip_index, picture.hash, &pixels)
			        ? 1
			        : 0;
			m_writer.WriteStruct(RecordType::Picture, picture);
			if (picture.available != 0) {
				m_writer.WriteCompressed(RecordType::PictureBytes, pixels);
			}
			m_flipped = false;
		}
		if (m_check_interval != 0) {
			WriteImageChecks(FrameEndCheck);
		}
		const auto work = g_work.Take();
		m_writer.WriteStruct(RecordType::Work,
		                     FrameWork {work.draws, work.dispatches, work.render_passes,
		                                work.buffer_uploads, work.buffer_upload_bytes,
		                                work.image_uploads, work.image_upload_bytes});
		m_writer.Write(RecordType::FrameEnd);
		if (++m_frames_done == m_frame_count) {
			Finish();
		}
		return;
	}
	std::error_code error;
	const bool      due =
	    m_trigger.empty() ? frame_id >= m_first_frame : std::filesystem::exists(m_trigger, error);
	if (m_state == State::Waiting && due) {
		Begin(frame_id);
	}
}

void Recorder::Begin(uint32_t frame_id) {
	if (m_attempts > MaxAttempts) {
		Log::WriteToConsoleAndLog("GPU capture: gave up after too many abandoned attempts\n");
		m_state = State::Done;
		return;
	}
	const auto started   = std::chrono::steady_clock::now();
	m_mapping_generation = m_renderer.MappingGeneration();
	m_started_submissions.clear();
	if (!m_renderer.GetGpu().SaveStartedSubmissions(m_started_submissions)) {
		NoteSkip("a suspended submission that cannot be stored"); // try at the next one
		return;
	}
	m_segments.clear();
	m_page_hashes.clear();
	if (!BuildSegments()) {
		NoteSkip("a guest memory operation is running"); // try at the next suspend point
		return;
	}

	// Nothing may be in flight on the host while the state is read.
	auto& scheduler = m_renderer.GetCommandScheduler();
	if (scheduler.Active()) {
		const auto tick = scheduler.CurrentTick();
		scheduler.Finish();
		scheduler.WaitPriorityOperations(tick);
	}

	// What only the host GPU holds goes to guest memory, where the page scan finds it. Buffer
	// bytes first: where a picture was drawn over them later, the picture is the newer content.
	auto& buffer_cache = m_renderer.GetBufferCache();
	for (const auto& segment: m_segments) {
		for (uint64_t page = segment.address; page < segment.address + segment.size;
		     page += PageSize) {
			if (buffer_cache.HasGpuDirtyBytes(page, PageSize)) {
				buffer_cache.ReadMemory(page, PageSize);
			}
		}
	}
	const auto images = m_renderer.GetTextureCache().WriteBackGpuImages();
	// A write-back can lose something: the stencil part of a depth image is not written, and
	// some formats are stored with fewer bits. So the live run goes on from guest memory too:
	// every picture the GPU wrote is loaded from it again, as the player will do. Both then
	// start the captured frames from the same pictures.
	const auto kept = m_renderer.GetTextureCache().ReloadGpuWrittenImages();

	if (!m_writer.Open(m_path, CurrentStateLayout())) {
		Log::WriteToConsoleAndLog(fmt::format("GPU capture: cannot create {}\n", m_path.string()));
		m_state = State::Done;
		return;
	}
	m_frames_done    = 0;
	m_read_points    = 0;
	m_checks_done    = 0;
	m_start_position = g_work.Position();
	m_flipped     = false;
	WriteState();
	m_writer.Write(RecordType::FrameBegin);
	{
		std::lock_guard lock(m_guest_writes_mutex);
		m_guest_writes.clear();
		m_guest_writes_lost = false;
		m_scan_position.store(g_work.Position(), std::memory_order_relaxed);
	}
	(void)g_work.Take();
	m_state = State::Recording;
	m_recording.store(true);

	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::steady_clock::now() - started)
	                    .count();
	Log::WriteToConsoleAndLog(
	    fmt::format("GPU capture: started at frame {}, state stored in {} ms\n", frame_id, ms));
	Log::WriteToConsoleAndLog(fmt::format(
	    "GPU capture: {} picture(s) held by the host GPU written to guest memory ({} MiB); {} "
	    "({} KiB) could not be written and start the capture from the guest memory they had; {} "
	    "cannot be loaded from guest memory and are not in the capture\n",
	    images.written, images.written_bytes >> 20u, images.not_written,
	    images.not_written_bytes >> 10u, kept));
}

void Recorder::WriteState() {
	std::vector<VideoOutCall> video_out_calls;
	VideoOut::VideoOutSaveConfiguration(video_out_calls);
	for (const auto& call: video_out_calls) {
		m_writer.WriteStruct(RecordType::VideoOut, call);
	}

	std::vector<std::pair<uint32_t, CommandProcessorState>> processors;
	m_renderer.GetGpu().SaveProcessorStates(processors);
	std::vector<uint8_t> payload(sizeof(ProcessorHeader) + sizeof(CommandProcessorState));
	for (const auto& [queue_index, state]: processors) {
		const ProcessorHeader header {queue_index, sizeof(CommandProcessorState)};
		std::memcpy(payload.data(), &header, sizeof(header));
		std::memcpy(payload.data() + sizeof(header), &state, sizeof(state));
		m_writer.Write(RecordType::ProcessorState, payload);
	}

	for (const auto& started: m_started_submissions) {
		m_writer.WriteStruct(RecordType::StartedSubmission, started);
	}
	m_writer.Write(RecordType::Gds, m_renderer.GetBufferCache().SaveGds());
	std::vector<SurfaceMeta> metas;
	for (const auto& meta: m_renderer.GetTextureCache().SaveSurfaceMetas()) {
		metas.push_back({meta.address, meta.type, meta.clear_mask});
	}
	m_writer.Write(RecordType::SurfaceMetas, {reinterpret_cast<const uint8_t*>(metas.data()),
	                                          metas.size() * sizeof(SurfaceMeta)});
	WriteShaders();
	std::vector<Range> buffers;
	for (const auto& buffer: m_renderer.GetBufferCache().SaveBufferRanges()) {
		buffers.push_back({buffer.address, buffer.size});
	}
	m_writer.Write(RecordType::Buffers, {reinterpret_cast<const uint8_t*>(buffers.data()),
	                                     buffers.size() * sizeof(Range)});

	std::vector<GuestRange> ranges;
	m_renderer.GetMappedRanges(ranges);
	for (const auto& range: ranges) {
		m_writer.WriteStruct(RecordType::MapRange, Range {range.address, range.size});
	}
	(void)WriteChangedPages(true);
}

bool Recorder::BuildSegments() {
	namespace Memory = LibKernel::Memory;
	std::vector<GuestRange> ranges;
	m_renderer.GetMappedRanges(ranges);
	// Built aside: when a guest memory operation is in the way, the known segments stay.
	std::vector<Segment> segments;
	size_t               pages = 0;
	for (const auto& range: ranges) {
		if ((range.address | range.size) % GuestPage != 0) {
			EXIT("GPU capture: mapped range is not aligned to guest pages\n");
		}
		for (uint64_t address = range.address; address < range.address + range.size;
		     address += GuestPage) {
			const uint8_t* backing = nullptr;
			if (!Memory::TryGetBackingPointer(address, GuestPage, &backing)) {
				// Program memory is private and has no backing alias. It is read through the
				// guest address when the guest itself may read it; anything else is a hole.
				// The memory operation that holds the lock is usually short. One that waits for
				// the GPU thread is not, so the wait here is bounded.
				bool readable = false;
				bool answered = false;
				for (int attempt = 0; attempt < 20 && !answered; attempt++) {
					answered = Memory::TryQueryCpuReadable(address, readable);
					if (!answered) {
						std::this_thread::sleep_for(std::chrono::microseconds(100));
					}
				}
				if (!answered) {
					return false;
				}
				if (!readable) {
					continue;
				}
				backing = nullptr;
			}
			bool extends = false;
			if (!segments.empty()) {
				const auto& last = segments.back();
				extends = last.address + last.size == address &&
				          (backing == nullptr
				               ? last.backing == nullptr
				               : last.backing != nullptr && last.backing + last.size == backing);
			}
			if (extends) {
				segments.back().size += GuestPage;
			} else {
				segments.push_back({address, GuestPage, backing, pages});
			}
			pages += GuestPage / PageSize;
		}
	}
	// A page that was known before keeps the hash of its stored bytes. A new page gets the hash
	// of no page at all, so the next scan stores it.
	std::vector<uint64_t> hashes(pages, 0);
	auto                  old = m_segments.begin();
	for (const auto& segment: segments) {
		while (old != m_segments.end() && old->address + old->size <= segment.address) {
			++old;
		}
		for (auto item = old;
		     item != m_segments.end() && item->address < segment.address + segment.size; ++item) {
			const uint64_t from = std::max(item->address, segment.address);
			const uint64_t to =
			    std::min(item->address + item->size, segment.address + segment.size);
			std::copy_n(m_page_hashes.begin() +
			                static_cast<long>(item->first_page + (from - item->address) / PageSize),
			            (to - from) / PageSize,
			            hashes.begin() + static_cast<long>(segment.first_page +
			                                               (from - segment.address) / PageSize));
		}
	}
	m_segments    = std::move(segments);
	m_page_hashes = std::move(hashes);
	return true;
}

void Recorder::FollowMapping() {
	const auto generation = m_renderer.MappingGeneration();
	if (generation == m_mapping_generation) {
		return;
	}
	// The guest mapped or unmapped memory. Store the ranges that are mapped now; the scan that
	// follows stores every page of a new range. The memory operation that changed the ranges
	// can still be running: then the next scan tries again, and this one goes over the known
	// segments.
	if (!BuildSegments()) {
		return;
	}
	m_mapping_generation = generation;
	std::vector<GuestRange> ranges;
	m_renderer.GetMappedRanges(ranges);
	std::vector<Range> stored;
	stored.reserve(ranges.size());
	for (const auto& range: ranges) {
		stored.push_back({range.address, range.size});
	}
	m_writer.Write(RecordType::Mapping, {reinterpret_cast<const uint8_t*>(stored.data()),
	                                     stored.size() * sizeof(Range)});
}

void Recorder::WriteShaders() {
	m_shader_generation = ShaderMapGeneration();
	std::vector<ShaderMapRecord> shaders;
	ShaderMapSave(shaders);
	std::vector<ShaderEntry> entries;
	entries.reserve(shaders.size());
	for (const auto& shader: shaders) {
		ShaderEntry entry;
		entry.address             = shader.address;
		entry.hash                = shader.hash;
		entry.user_data           = reinterpret_cast<uint64_t>(shader.data.user_data);
		entry.input_semantics     = reinterpret_cast<uint64_t>(shader.data.input_semantics);
		entry.type                = static_cast<uint32_t>(shader.data.type);
		entry.num_input_semantics = shader.data.num_input_semantics;
		entry.code_size_bytes     = shader.data.code_size_bytes;
		entry.scratch_size_dwords = shader.data.scratch_size_dwords;
		entries.push_back(entry);
	}
	m_writer.Write(RecordType::Shaders, {reinterpret_cast<const uint8_t*>(entries.data()),
	                                     entries.size() * sizeof(ShaderEntry)});
}

void Recorder::FlushPages(std::vector<uint64_t>& addresses, std::vector<uint8_t>& data) {
	if (addresses.empty()) {
		return;
	}
	// Compressed on the calling thread, so the scanning threads do it side by side.
	const auto      packed = Writer::PackPages(addresses, data);
	std::lock_guard lock(m_writer_mutex);
	m_writer.WritePacked(packed);
	addresses.clear();
	data.clear();
}

void Recorder::ScanPages(const Segment& segment, uint64_t offset, uint64_t bytes, bool initial,
                         std::vector<uint64_t>& addresses, std::vector<uint8_t>& data) {
	const uint8_t* source = segment.backing != nullptr
	                            ? segment.backing + offset
	                            : reinterpret_cast<const uint8_t*>(segment.address + offset);
	for (uint64_t page = 0; page < bytes; page += PageSize) {
		auto& known = m_page_hashes[segment.first_page + (offset + page) / PageSize];
		if (!initial && XXH3_64bits(source + page, PageSize) == known) {
			continue;
		}
		// The guest may write the page while it is read. Store a copy and the hash of that
		// copy, so a later write is seen as a change against exactly what was stored.
		std::array<uint8_t, PageSize> copy;
		std::memcpy(copy.data(), source + page, PageSize);
		known = XXH3_64bits(copy.data(), PageSize);
		if (initial && IsZeroPage(copy.data())) {
			continue;
		}
		addresses.push_back(segment.address + offset + page);
		data.insert(data.end(), copy.begin(), copy.end());
		if (addresses.size() == PagesPerRecord) {
			FlushPages(addresses, data);
		}
	}
}

void Recorder::OnGuestWrite(WriteTarget target, uint64_t address, uint64_t size) noexcept {
	if (!m_recording.load(std::memory_order_relaxed)) {
		return;
	}
	std::lock_guard lock(m_guest_writes_mutex);
	// The queue never grows here: this runs inside a page fault of a guest thread.
	if (m_guest_writes.size() == m_guest_writes.capacity()) {
		m_guest_writes_lost = true;
		return;
	}
	const auto position = g_work.Position() - m_scan_position.load(std::memory_order_relaxed);
	m_guest_writes.push_back(
	    {address, size, static_cast<uint32_t>(target), static_cast<uint32_t>(position)});
}

bool Recorder::WriteGuestWrites() {
	bool lost = false;
	m_taken_writes.clear();
	{
		std::lock_guard lock(m_guest_writes_mutex);
		m_taken_writes.swap(m_guest_writes);
		lost = std::exchange(m_guest_writes_lost, false);
		m_scan_position.store(g_work.Position(), std::memory_order_relaxed);
	}
	if (!m_taken_writes.empty()) {
		m_writer.Write(RecordType::GuestWrites,
		               {reinterpret_cast<const uint8_t*>(m_taken_writes.data()),
		                m_taken_writes.size() * sizeof(WriteRange)});
	}
	return !lost;
}

bool Recorder::WriteChangedPages(bool initial) {
	const auto started = std::chrono::steady_clock::now();
	if (!initial) {
		FollowMapping();
	}

	// Backed memory is read through the alias, which never faults, so other threads can share
	// the work. The game is stopped on the GPU thread for the length of this scan.
	struct Task {
		const Segment* segment;
		uint64_t       offset;
		uint64_t       bytes;
	};
	std::vector<Task> tasks;
	for (const auto& segment: m_segments) {
		if (segment.backing == nullptr) {
			continue;
		}
		for (uint64_t offset = 0; offset < segment.size; offset += ChunkSize) {
			tasks.push_back({&segment, offset, std::min(ChunkSize, segment.size - offset)});
		}
	}
	std::atomic<size_t> next {0};
	const auto          work = [&] {
		std::vector<uint64_t> addresses;
		std::vector<uint8_t>  data;
		for (size_t i = next.fetch_add(1); i < tasks.size(); i = next.fetch_add(1)) {
			ScanPages(*tasks[i].segment, tasks[i].offset, tasks[i].bytes, initial, addresses, data);
		}
		FlushPages(addresses, data);
	};
	const unsigned           workers = std::clamp(std::thread::hardware_concurrency(), 2u, 8u) - 1;
	std::vector<std::thread> threads;
	threads.reserve(workers);
	for (unsigned i = 0; i < workers; i++) {
		threads.emplace_back(work);
	}
	work();
	for (auto& thread: threads) {
		thread.join();
	}

	// Private memory is read through guest addresses. The renderer may have protected a page;
	// the fault must then arrive on this thread, the GPU thread.
	std::vector<uint64_t> addresses;
	std::vector<uint8_t>  data;
	for (const auto& segment: m_segments) {
		// A private segment the guest has unmapped since the segments were built has no memory
		// to read.
		if (segment.backing == nullptr && m_renderer.IsMapped(segment.address, segment.size)) {
			ScanPages(segment, 0, segment.size, initial, addresses, data);
		}
	}
	FlushPages(addresses, data);
	// The writes the caches were told about since the last scan, this scan included: the guest
	// keeps running while the GPU thread scans, so most of them come during a scan. A write
	// after this point is stored by the next scan, stamped as coming right after this one.
	if (!initial && !WriteGuestWrites()) {
		Abandon("more guest writes between two scans than the recorder can hold");
		return false;
	}
	if (!initial) {
		m_scans++;
		m_scan_time += std::chrono::steady_clock::now() - started;
	}
	return true;
}

void Recorder::OnSubmissionStart(SubmissionKind kind, uint32_t queue,
                                 std::span<const uint32_t> commands,
                                 std::span<const uint32_t> constant_commands) {
	if (m_state != State::Recording) {
		return;
	}
	if (kind == SubmissionKind::CpuFlip) {
		Abandon("the frame has a flip submitted from the CPU");
		return;
	}
	if (ShaderMapGeneration() != m_shader_generation) {
		WriteShaders();
	}
	if (!WriteChangedPages(false)) {
		return;
	}

	Submission submission;
	submission.kind =
	    static_cast<uint32_t>(kind == SubmissionKind::Compute ? Capture::SubmissionKind::Compute
	                                                          : Capture::SubmissionKind::Graphics);
	submission.queue                     = queue;
	submission.commands_address          = reinterpret_cast<uint64_t>(commands.data());
	submission.commands_dwords           = commands.size();
	submission.constant_commands_address = reinterpret_cast<uint64_t>(constant_commands.data());
	submission.constant_commands_dwords  = constant_commands.size();
	m_writer.WriteStruct(RecordType::Submission, submission);
}

void Recorder::OnWaitBlocked(uint64_t /*address*/, uint32_t /*size*/) {}

void Recorder::OnWaitPassed(uint64_t address, uint32_t size, bool had_blocked) {
	if (m_state != State::Recording) {
		return;
	}
	// A wait is where the guest synchronises with its own writes: it fills commands and data,
	// then writes the value the GPU waits for. So the pages that changed since the last stored
	// state belong to this point of the stream.
	if (!WriteChangedPages(false)) {
		return;
	}
	// Every wait is stored, so the player can tell the waits on one address apart by their
	// order. The command processor has just read these bytes through this pointer.
	m_writer.WriteBytes(address, {reinterpret_cast<const uint8_t*>(address), size}, had_blocked);
}

void Recorder::OnGuestRead(uint64_t address, uint64_t size) {
	if (m_state != State::Recording || size == 0) {
		return;
	}
	// The guest runs while the GPU thread works, so memory the renderer reads now can differ
	// from what the last scan stored. Store the pages it reads as they are at this moment, so
	// the player can give the same draw the same bytes.
	std::vector<uint64_t> addresses;
	std::vector<uint8_t>  data;
	const uint64_t        begin   = address & ~(PageSize - 1);
	const uint64_t        end     = (address + size + PageSize - 1) & ~(PageSize - 1);
	auto                  segment = std::upper_bound(
	    m_segments.begin(), m_segments.end(), begin,
	    [](uint64_t value, const Segment& item) { return value < item.address + item.size; });
	for (; segment != m_segments.end() && segment->address < end; ++segment) {
		const uint64_t from = std::max(begin, segment->address);
		const uint64_t to   = std::min(end, segment->address + segment->size);
		ScanPages(*segment, from - segment->address, to - from, false, addresses, data);
	}
	if (addresses.empty()) {
		return;
	}
	const auto position = g_work.Position() - m_scan_position.load(std::memory_order_relaxed);
	FlushPages(addresses, data);
	if (!WriteGuestWrites()) {
		Abandon("more guest writes between two scans than the recorder can hold");
		return;
	}
	m_writer.WriteStruct(RecordType::ReadPoint, ReadPoint {static_cast<uint32_t>(position), 0});
	m_read_points++;
}

void Recorder::OnBetweenCommands() {
	if (m_state != State::Recording || m_check_interval == 0) {
		return;
	}
	const uint64_t position = g_work.Position() - m_start_position;
	if (position / m_check_interval == m_checks_done) {
		return;
	}
	m_checks_done = position / m_check_interval;
	WriteImageChecks(position);
}

void Recorder::WriteImageChecks(uint64_t position) {
	const auto              hashes = m_renderer.GetTextureCache().HashGpuImages();
	std::vector<uint8_t>    payload(sizeof(ImageChecksHeader) + hashes.size() * sizeof(ImageCheck));
	const ImageChecksHeader header {position};
	std::memcpy(payload.data(), &header, sizeof(header));
	for (size_t i = 0; i < hashes.size(); i++) {
		const ImageCheck check {hashes[i].range.address,
		                        hashes[i].range.size,
		                        hashes[i].hash,
		                        hashes[i].width,
		                        hashes[i].height,
		                        hashes[i].depth,
		                        static_cast<uint32_t>(hashes[i].format)};
		std::memcpy(payload.data() + sizeof(header) + i * sizeof(check), &check, sizeof(check));
	}
	m_writer.Write(RecordType::ImageChecks, payload);
}

void Recorder::OnFlip(int handle, int index) {
	m_flipped     = true;
	m_flip_handle = handle;
	m_flip_index  = index;
}

void Recorder::OnOtherBlock(const char* what) {
	if (m_state == State::Recording) {
		Abandon(what);
	}
}

void Recorder::NoteSkip(const char* reason) {
	// Says why no capture starts, without a line per frame.
	if (m_skips++ % 100 == 0) {
		Log::WriteToConsoleAndLog(
		    fmt::format("GPU capture: waiting for a frame boundary without {}\n", reason));
	}
}

void Recorder::Abandon(const char* reason) {
	// The file keeps no `End` record, so a reader rejects it. The next attempt overwrites it.
	m_recording.store(false);
	(void)m_writer.Close();
	m_state = State::Waiting;
	m_attempts++;
	Log::WriteToConsoleAndLog(fmt::format("GPU capture: abandoned ({}); trying again\n", reason));
}

void Recorder::Finish() {
	m_recording.store(false);
	m_writer.Write(RecordType::End);
	const bool ok = m_writer.Close();
	m_state       = State::Done;
	// Each scan stops the game; this tells how long the pauses were.
	Log::WriteToConsoleAndLog(fmt::format(
	    "GPU capture: {} memory scans, {:.0f} ms each on average; {} time(s) the renderer read "
	    "memory that had changed since the last scan\n",
	    m_scans,
	    m_scans == 0 ? 0.0
	                 : std::chrono::duration<double, std::milli>(m_scan_time).count() / m_scans,
	    m_read_points));
	m_segments.clear();
	m_page_hashes.clear();
	Log::WriteToConsoleAndLog(
	    ok ? fmt::format("GPU capture: {} frame(s) written to {}\n", m_frames_done, m_path.string())
	       : fmt::format("GPU capture: writing {} failed\n", m_path.string()));
}

} // namespace Libs::Graphics::Capture
