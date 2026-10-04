#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WORKCOUNTERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WORKCOUNTERS_H_

// Counts of the work the renderer hands to the host GPU. The capture recorder stores them per
// frame and the capture player compares its own, to show that a replayed frame costs the host
// what the live frame did.

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

struct WorkCounts {
	uint64_t draws               = 0;
	uint64_t dispatches          = 0;
	uint64_t render_passes       = 0;
	uint64_t buffer_uploads      = 0;
	uint64_t buffer_upload_bytes = 0;
	uint64_t image_uploads       = 0;
	uint64_t image_upload_bytes  = 0;
};

class WorkCounters final {
public:
	void Draw() noexcept {
		Add(m_draws, 1);
		Add(m_position, 1);
	}
	void Dispatch() noexcept {
		Add(m_dispatches, 1);
		Add(m_position, 1);
	}
	void RenderPass() noexcept { Add(m_render_passes, 1); }
	void BufferUpload(uint64_t bytes) noexcept {
		Add(m_buffer_uploads, 1);
		Add(m_buffer_upload_bytes, bytes);
	}
	void ImageUpload(uint64_t bytes) noexcept {
		Add(m_image_uploads, 1);
		Add(m_image_upload_bytes, bytes);
	}

	// Draws and dispatches since the process started: a clock of the renderer's progress.
	[[nodiscard]] uint64_t Position() const noexcept {
		return m_position.load(std::memory_order_relaxed);
	}

	// The counts since the last call.
	[[nodiscard]] WorkCounts Take() noexcept {
		return {m_draws.exchange(0),
		        m_dispatches.exchange(0),
		        m_render_passes.exchange(0),
		        m_buffer_uploads.exchange(0),
		        m_buffer_upload_bytes.exchange(0),
		        m_image_uploads.exchange(0),
		        m_image_upload_bytes.exchange(0)};
	}

private:
	static void Add(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
		counter.fetch_add(value, std::memory_order_relaxed);
	}

	std::atomic<uint64_t> m_position {0};
	std::atomic<uint64_t> m_draws {0};
	std::atomic<uint64_t> m_dispatches {0};
	std::atomic<uint64_t> m_render_passes {0};
	std::atomic<uint64_t> m_buffer_uploads {0};
	std::atomic<uint64_t> m_buffer_upload_bytes {0};
	std::atomic<uint64_t> m_image_uploads {0};
	std::atomic<uint64_t> m_image_upload_bytes {0};
};

inline WorkCounters g_work;

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WORKCOUNTERS_H_
