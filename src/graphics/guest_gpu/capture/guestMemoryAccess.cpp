#include "graphics/guest_gpu/capture/guestMemoryAccess.h"

#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics::Capture {

bool ReadGuestMemory(uint64_t address, uint8_t* out, uint64_t size) {
	namespace Memory = LibKernel::Memory;
	if (Memory::TryReadBacking(address, out, size)) {
		return true;
	}
	// Mixed range: go by guest page. Program memory is private and has no backing alias; it is
	// read through the guest address when the guest itself may read it.
	constexpr uint64_t GuestPage = 0x4000;
	for (uint64_t offset = 0; offset < size;) {
		const uint64_t current = address + offset;
		const uint64_t bytes   = std::min(GuestPage - current % GuestPage, size - offset);
		if (!Memory::TryReadBacking(current, out + offset, bytes)) {
			bool readable = false;
			if (!Memory::TryQueryCpuReadable(current, readable)) {
				return false;
			}
			if (readable) {
				std::memcpy(out + offset, reinterpret_cast<const void*>(current), bytes);
			} else {
				std::memset(out + offset, 0, bytes);
			}
		}
		offset += bytes;
	}
	return true;
}

bool HashDisplayBuffer(RenderContext& renderer, int handle, int index, uint64_t& hash,
                       std::vector<uint8_t>* pixels_out) {
	uint64_t address = 0;
	uint64_t size    = 0;
	if (!VideoOut::VideoOutGetBufferRange(handle, index, address, size) ||
	    !renderer.GetTextureCache().WriteBackImage(address, size)) {
		return false;
	}
	auto& scheduler = renderer.GetCommandScheduler();
	if (scheduler.Active()) {
		const auto tick = scheduler.CurrentTick();
		scheduler.Finish();
		scheduler.WaitPriorityOperations(tick);
	}
	std::vector<uint8_t> pixels(size);
	if (!ReadGuestMemory(address, pixels.data(), size)) {
		return false;
	}
	hash = XXH3_64bits(pixels.data(), pixels.size());
	if (pixels_out != nullptr) {
		*pixels_out = std::move(pixels);
	}
	return true;
}

} // namespace Libs::Graphics::Capture
