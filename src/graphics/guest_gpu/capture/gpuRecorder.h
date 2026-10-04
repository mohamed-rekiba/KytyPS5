#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_

#include "graphics/guest_gpu/capture/captureFile.h"
#include "graphics/guest_gpu/gpuObserver.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
}

namespace Libs::Graphics::Capture {

// Records frames of the guest GPU stream into a capture file. It starts at a suspend point, where
// the graphics command processor is reset, and stores the state a fresh process needs: guest
// memory, command processor registers, GDS, shader registrations, the clear state of surface
// metadata, the ranges that have a cached buffer and the video-out configuration. Pictures that
// only the host GPU holds are written to guest memory first, so the stored memory has them.
//
// Then it stores, in GPU-thread order, each submission and each wait on guest memory that
// passes, with what happened to guest memory before it: the pages that changed, the ranges the
// renderer's caches were told the guest wrote, the ranges the renderer wrote itself, and changes
// of the mapped ranges. The guest runs while the GPU thread works, so the renderer can read
// memory that changed after the last scan: such pages are stored at the moment it reads them.
// Each frame also gets the picture it flipped and the work the renderer handed to the host GPU.
//
// A write-back does not keep everything (the stencil part of a depth image, for one), so the
// live run itself continues from guest memory: at the capture start every picture the GPU wrote
// is loaded from it again. The game shows that as one frame that takes longer.
//
// Not stored: host pictures that have no upload from guest memory (multisampled, compressed),
// and flips submitted from the CPU. A capture that meets a CPU flip is abandoned and started
// again at the next suspend point.
class Recorder final: public GuestGpuObserver {
public:
	// Recording starts at the first suspend point numbered `first_frame` or later; with a
	// `trigger` path, at the first suspend point after that file exists.
	// `check_interval`, when not 0: store a hash of every image the host GPU wrote each time
	// that many draws and dispatches have passed, so a replay can name the first image that
	// differs. Each check stops the game for a second or more.
	Recorder(RenderContext& renderer, std::filesystem::path path, uint32_t first_frame,
	         uint32_t frame_count, std::filesystem::path trigger, uint32_t check_interval);
	~Recorder() override = default;

	void OnSubmissionStart(SubmissionKind kind, uint32_t queue, std::span<const uint32_t> commands,
	                       std::span<const uint32_t> constant_commands) override;
	void OnSuspendPoint(uint32_t frame_id) override;
	void OnBeforeWait(uint64_t /*address*/, uint32_t /*size*/) override {}
	void OnWaitBlocked(uint64_t address, uint32_t size) override;
	void OnWaitPassed(uint64_t address, uint32_t size, bool had_blocked) override;
	void OnFlip(int handle, int index) override;
	void OnOtherBlock(const char* what) override;
	void OnHostWork() override {}
	void OnBetweenCommands() override;
	void OnGuestRead(uint64_t address, uint64_t size) override;
	void OnGuestWrite(WriteTarget target, uint64_t address, uint64_t size) noexcept override;

private:
	enum class State { Waiting, Recording, Done };

	// A run of guest pages that are read the same way.
	struct Segment {
		uint64_t address = 0;
		uint64_t size    = 0;
		// Backing alias of the run; null for private memory, which is read through the guest
		// address.
		const uint8_t* backing    = nullptr;
		size_t         first_page = 0; // index into m_page_hashes
	};

	void Begin(uint32_t frame_id);
	void NoteSkip(const char* reason);
	void Abandon(const char* reason);
	void Finish();
	void WriteState();
	void WriteShaders();
	// Stores a hash of every image the host GPU wrote, taken now.
	void WriteImageChecks(uint64_t position);
	// Splits the mapped ranges into segments. False when a guest memory operation is running;
	// the GPU thread must not wait for one.
	[[nodiscard]] bool BuildSegments();
	// Stores a change of the mapped ranges, if there was one since the last scan and the new
	// ranges can be read by now.
	void FollowMapping();
	// Stores the pages whose bytes differ from the last stored ones. `initial`: every page that
	// is not zero is stored, and bytes the GPU holds are read back first.
	// False when the capture was abandoned: nothing more may be written.
	[[nodiscard]] bool WriteChangedPages(bool initial);
	void ScanPages(const Segment& segment, uint64_t offset, uint64_t bytes, bool initial,
	               std::vector<uint64_t>& addresses, std::vector<uint8_t>& data);
	void FlushPages(std::vector<uint64_t>& addresses, std::vector<uint8_t>& data);
	// Stores the guest writes queued since the last call. False when some were lost.
	[[nodiscard]] bool WriteGuestWrites();

	RenderContext&                      m_renderer;
	std::filesystem::path               m_path;
	std::filesystem::path                    m_trigger;
	uint32_t                            m_first_frame        = 0;
	uint32_t                            m_frame_count        = 1;
	uint32_t                                 m_check_interval     = 0;
	uint64_t                                 m_start_position     = 0; // renderer position at Begin
	uint64_t                                 m_checks_done        = 0;
	State                               m_state              = State::Waiting;
	uint32_t                            m_attempts           = 0;
	uint32_t                            m_frames_done        = 0;
	bool                                m_flipped            = false;
	int                                 m_flip_handle        = 0;
	int                                 m_flip_index         = 0;
	uint64_t                            m_mapping_generation = 0;
	uint64_t                            m_shader_generation  = 0;
	Writer                              m_writer;
	std::vector<GuestGpu::StartedSubmission> m_started_submissions;
	std::vector<Segment>                m_segments;
	std::vector<uint64_t>               m_page_hashes;
	uint32_t                                 m_skips = 0;
	uint32_t                            m_scans = 0;
	uint32_t                                 m_read_points = 0;
	std::chrono::steady_clock::duration m_scan_time {};
	std::mutex                          m_writer_mutex; // page records come from several threads
	// Guest threads queue their writes here; the GPU thread stores them at the next scan.
	std::atomic<bool>       m_recording {false};
	std::mutex              m_guest_writes_mutex;
	std::vector<WriteRange> m_guest_writes;
	std::vector<WriteRange> m_taken_writes;
	bool                    m_guest_writes_lost = false;
	// The renderer's position at the last scan; a queued write is stamped relative to it.
	std::atomic<uint64_t> m_scan_position {0};
};

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_
