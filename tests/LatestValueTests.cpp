#include "common/latestValue.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LatestValueTests: failed: %s\n", message);
		std::abort();
	}
}

void TestOneConsumerForManyValues() {
	Common::LatestValue<std::string> slot;
	Check(!slot.Take().has_value(), "an empty slot gave a value");
	Check(slot.Put("frame 1"), "the first value did not ask for a consumer");
	Check(!slot.Put("frame 2") && !slot.Put("frame 3"),
	      "a value put while a consumer is pending asked for a second consumer");
	const auto taken = slot.Take();
	Check(taken.has_value() && *taken == "frame 3", "the consumer did not get the newest value");
	Check(!slot.Take().has_value(), "a taken value was given twice");
}

void TestNextValueAfterTheConsumerRan() {
	Common::LatestValue<std::string> slot;
	(void)slot.Put("frame 1");
	(void)slot.Take();
	Check(slot.Put("frame 2"), "a value put after the consumer ran did not ask for a new one");
	Check(*slot.Take() == "frame 2", "the second consumer did not get its value");
}

} // namespace

int main() {
	TestOneConsumerForManyValues();
	TestNextValueAfterTheConsumerRan();
	std::puts("LatestValueTests: all cases passed");
	return 0;
}
