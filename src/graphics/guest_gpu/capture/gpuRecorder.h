#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_

#include "graphics/guest_gpu/capture/captureFile.h"
#include "graphics/guest_gpu/gpuObserver.h"
#include "graphics/guest_gpu/graphicsRun.h"

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
// the graphics command processor is reset, and stores the state a
// fresh process needs: guest memory, command processor registers, GDS, shader registrations and
// the video-out configuration. Then it stores, in GPU-thread order, each submission and each wait
// on guest memory that passes, with the guest pages that changed before it.
//
// Not stored: images that exist only on the host, flips submitted from the CPU, and mapping
// changes. A capture that meets one of the last two is abandoned and started again at the next
// suspend point.
class Recorder final: public GuestGpuObserver {
public:
	// Recording starts at the first suspend point numbered `first_frame` or later; with a
	// `trigger` path, at the first suspend point after that file exists.
	Recorder(RenderContext& renderer, std::filesystem::path path, uint32_t first_frame,
	         uint32_t frame_count, std::filesystem::path trigger);
	~Recorder() override = default;

	void OnSubmissionStart(SubmissionKind kind, uint32_t queue, std::span<const uint32_t> commands,
	                       std::span<const uint32_t> constant_commands) override;
	void OnSuspendPoint(uint32_t frame_id) override;
	void OnBeforeWait(uint64_t /*address*/, uint32_t /*size*/) override {}
	void OnWaitBlocked(uint64_t address, uint32_t size) override;
	void OnWaitPassed(uint64_t address, uint32_t size, bool had_blocked) override;
	void OnFlip(int handle, int index) override;
	void OnOtherBlock(const char* what) override;

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
	// Splits the mapped ranges into segments. False when a guest memory operation is running;
	// the GPU thread must not wait for one.
	[[nodiscard]] bool BuildSegments();
	// Stores the pages whose bytes differ from the last stored ones. `initial`: every page that
	// is not zero is stored, and bytes the GPU holds are read back first.
	void WriteChangedPages(bool initial);
	void ScanPages(const Segment& segment, uint64_t offset, uint64_t bytes, bool initial,
	               std::vector<uint64_t>& addresses, std::vector<uint8_t>& data);
	void FlushPages(std::vector<uint64_t>& addresses, std::vector<uint8_t>& data);

	RenderContext&                      m_renderer;
	std::filesystem::path               m_path;
	std::filesystem::path                    m_trigger;
	uint32_t                            m_first_frame        = 0;
	uint32_t                            m_frame_count        = 1;
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
	std::chrono::steady_clock::duration m_scan_time {};
	std::mutex                          m_writer_mutex; // page records come from several threads
};

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPURECORDER_H_
