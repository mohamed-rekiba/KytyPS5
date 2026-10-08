// Tests for the rule that decides what a garbage-collection pass of the buffer and texture caches
// frees.

#include "graphics/host_gpu/cacheCollection.h"

#include <iostream>

namespace {

using Libs::Graphics::CollectionAge;
using Libs::Graphics::CollectionKind;
using Libs::Graphics::IDLE_COLLECT_AGE;
using Libs::Graphics::PlanCollection;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

constexpr uint64_t GiB = 1024ull * 1024 * 1024;

void TestADiscreteGpuCollectsOnlyUnderMemoryPressure() {
	Check(PlanCollection(1 * GiB, 4 * GiB, false) == CollectionKind::None,
	      "below the trigger a discrete GPU keeps everything");
	Check(PlanCollection(4 * GiB, 4 * GiB, false) == CollectionKind::Memory,
	      "at the trigger a discrete GPU collects for memory");
}

void TestUnifiedMemoryAlsoCollectsIdleItems() {
	Check(PlanCollection(1 * GiB, 32 * GiB, true) == CollectionKind::Idle,
	      "below a trigger it never reaches, unified memory still frees unused items");
	Check(PlanCollection(32 * GiB, 32 * GiB, true) == CollectionKind::Memory,
	      "at the trigger unified memory collects for memory as before");
}

void TestIdleItemsMustBeUnusedForLong() {
	Check(CollectionAge(CollectionKind::Idle, 160, 10'000) == IDLE_COLLECT_AGE,
	      "an idle pass frees only what was unused for the idle age");
	Check(CollectionAge(CollectionKind::Memory, 160, 10'000) == 160,
	      "a memory pass keeps the cache's own age");
	Check(CollectionAge(CollectionKind::Idle, 160, 100) == 100,
	      "early in a run the age is capped by the passes done so far");
	Check(IDLE_COLLECT_AGE >= 500, "the idle age spans many frames: about 28 passes run per frame");
}

} // namespace

int main() {
	TestADiscreteGpuCollectsOnlyUnderMemoryPressure();
	TestUnifiedMemoryAlsoCollectsIdleItems();
	TestIdleItemsMustBeUnusedForLong();
	if (g_failures != 0) {
		std::cerr << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "cache collection tests passed\n";
	return 0;
}
