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
// `loops` above 1 feeds the captured frames again without restoring the state in between. The
// first pass is the faithful one; later passes show the steady-state cost of the same frames.
//
// Returns false, after printing the reason, when the file cannot be replayed.
[[nodiscard]] bool Play(RenderContext& renderer, const std::filesystem::path& path, uint32_t loops);

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GPUPLAYER_H_
