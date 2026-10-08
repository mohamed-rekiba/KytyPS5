#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CACHECOLLECTION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CACHECOLLECTION_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// What a garbage-collection pass of the buffer and texture caches frees.
//
// The caches collect when the device memory in use reaches a trigger derived from the memory
// budget. On a discrete GPU the budget is its own memory, so the trigger is reached. On a GPU that
// shares the system memory the budget is most of that memory (32 GB on an M4 Max with 36 GB), and
// a game never reaches the trigger: the caches keep every buffer and image they ever made.
//
// That costs more than memory. MoltenVK attaches every live resource to every command buffer, and
// the buffer cache watches the pages of every buffer it holds. In Crash Bandicoot 4 the buffer
// cache grew to 7,000 buffers in five minutes, the GPU time of a frame grew from about 20 ms to
// 75 ms, and the game fell from 27 to 7 fps. With the idle passes below, GPU memory stayed at
// about 5 GB and the game at 20 to 25 fps for six minutes of play.
//
// So on shared memory a pass also frees, below the trigger, items that were unused for the idle
// age. Items the GPU wrote and the CPU has not read back are left to the memory passes.
enum class CollectionKind : uint8_t {
	// Nothing to free.
	None,
	// Free clean items that were unused for the idle age.
	Idle,
	// Memory is short: free with the cache's own ages and limits.
	Memory,
};

// In collection passes, which run once for each completed guest submission: about 28 per frame
// and 480 per second in Crash Bandicoot 4, so about two seconds.
inline constexpr uint64_t IDLE_COLLECT_AGE = 1000;

[[nodiscard]] constexpr CollectionKind PlanCollection(uint64_t used_memory, uint64_t trigger_memory,
                                                      bool shared_memory) noexcept {
	if (used_memory >= trigger_memory) {
		return CollectionKind::Memory;
	}
	return shared_memory ? CollectionKind::Idle : CollectionKind::None;
}

// The minimum age, in passes, of an item a pass may free. `memory_age` is the cache's own age for
// a memory pass; `passes` is the number of passes run so far.
[[nodiscard]] constexpr uint64_t CollectionAge(CollectionKind kind, uint64_t memory_age,
                                               uint64_t passes) noexcept {
	return std::min(kind == CollectionKind::Idle ? IDLE_COLLECT_AGE : memory_age, passes);
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_CACHECOLLECTION_H_ */
