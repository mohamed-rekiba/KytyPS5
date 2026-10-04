#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPUPLAYER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPUPLAYER_H_

#include <cstdint>
#include <filesystem>

namespace Libs::Graphics {
class RenderContext;
}

namespace Libs::Graphics::Capture {

// Replays a capture file into a renderer that has run nothing yet: restores the stored state,
// then feeds the stored submissions in their order. No guest code runs. Call it from a thread
// that is not the GPU thread and not the window thread.
//
// `loops` passes feed the captured frames. Every pass starts from the state of the capture start:
// guest memory is put back, and what the host GPU wrote in the pass before is loaded from guest
// memory again. That load happens in the first frame of a pass, so the time of a pass is taken
// over the frames after the first one. Each pass reports its time per frame, the work the
// renderer handed to the host GPU against the live run's, and whether each picture matches the
// live run's and the first pass's.
//
// Returns false, after printing the reason, when the file cannot be replayed.
[[nodiscard]] bool Play(RenderContext& renderer, const std::filesystem::path& path, uint32_t loops);

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPUPLAYER_H_
