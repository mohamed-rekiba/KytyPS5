// Tests for the rule that keeps pages the guest writes again and again open.

#include "graphics/host_gpu/writeHeat.h"

#include <iostream>

namespace {

using Heat = Libs::Graphics::WriteHeat<64>;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

void TestAPageIsHotAfterItsSecondWrite() {
	Heat heat;
	heat.NoteWrite(3, 5);
	Check(!heat.IsHot(3) && !heat.IsHot(4), "one write must not make a page hot");
	heat.NoteWrite(4, 6);
	Check(!heat.IsHot(3) && heat.IsHot(4) && !heat.IsHot(5), "only the page written twice is hot");
	heat.NoteWrite(4, 5);
	Check(heat.IsHot(4), "a hot page stays hot");
}

void TestHotPagesStayDirtyWhenTheRangeIsCleared() {
	Heat heat;
	heat.NoteWrite(8, 10);
	heat.NoteWrite(8, 9);
	Heat::Bits dirty;
	dirty.SetRange(8, 12);
	dirty.UnsetRange(8, 12); // the renderer uploaded the range
	heat.KeepHotDirty(dirty, 8, 12);
	Check(dirty.Get(8), "a hot page must stay dirty, so that it is uploaded again");
	Check(!dirty.Get(9) && !dirty.Get(10) && !dirty.Get(11), "a page that is not hot is clean");
}

void TestCoolingEndsHeat() {
	Heat heat;
	heat.NoteWrite(1, 3);
	heat.NoteWrite(1, 3);
	Check(heat.Cool(2, 4), "cooling a range with a hot page must say so");
	Check(heat.IsHot(1) && !heat.IsHot(2), "only the cooled page lost its heat");
	Check(!heat.Cool(2, 4), "a range with no hot page reports none");
	heat.NoteWrite(2, 3);
	Check(!heat.IsHot(2), "a cooled page needs two writes again");
	heat.CoolAll();
	Check(!heat.IsHot(1), "cooling everything ends all heat");
	heat.NoteWrite(1, 2);
	Check(!heat.IsHot(1), "after cooling everything a page needs two writes again");
}

} // namespace

int main() {
	TestAPageIsHotAfterItsSecondWrite();
	TestHotPagesStayDirtyWhenTheRangeIsCleared();
	TestCoolingEndsHeat();
	if (g_failures != 0) {
		std::cerr << "WriteHeatTests: " << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "WriteHeatTests: all cases passed\n";
	return 0;
}
