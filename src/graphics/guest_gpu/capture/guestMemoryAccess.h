#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GUESTMEMORYACCESS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GUESTMEMORYACCESS_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics::Capture {

// Copies guest memory for the capture without taking a page fault where the range has a backing
// alias. A part that has no readable memory behind it (a hole in a sparse range) reads as zero.
// False when the read had to be given up because a guest memory operation is running.
[[nodiscard]] bool ReadGuestMemory(uint64_t address, uint8_t* out, uint64_t size);

} // namespace Libs::Graphics::Capture

namespace Libs::Graphics {
class RenderContext;
}

namespace Libs::Graphics::Capture {

// Hash of the picture in display buffer `index` of video-out port `handle`. GPU thread; waits for
// the GPU work recorded so far. False when the buffer is not registered or the GPU holds a
// picture that cannot be written back to guest memory.
// `pixels`, when given, receives the bytes the hash covers.
[[nodiscard]] bool HashDisplayBuffer(RenderContext& renderer, int handle, int index, uint64_t& hash,
                                     std::vector<uint8_t>* pixels = nullptr);

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_GUESTMEMORYACCESS_H_
