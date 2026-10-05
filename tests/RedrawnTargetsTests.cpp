#include "graphics/host_gpu/renderer/redrawnTargets.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::RedrawnTargets;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "RedrawnTargetsTests: failed: %s\n", message);
		std::abort();
	}
}

void TestATargetDrawnEveryFrame() {
	RedrawnTargets targets;
	Check(!targets.Draw(0x1000), "a target that was never drawn counts as drawn every frame");
	Check(!targets.Draw(0x1000), "a second draw in the first frame changed the answer");
	targets.NextFrame();
	Check(targets.Draw(0x1000), "a target drawn in the frame before is not drawn every frame");
	Check(targets.Draw(0x1000), "a second draw in the same frame changed the answer");
	Check(!targets.Draw(0x2000), "a new target took the answer of another target");
}

void TestATargetDrawnNowAndThen() {
	RedrawnTargets targets;
	(void)targets.Draw(0x1000);
	targets.NextFrame();
	targets.NextFrame();
	Check(!targets.Draw(0x1000), "a target that missed a frame counts as drawn every frame");
	targets.NextFrame();
	Check(targets.Draw(0x1000), "a target that is drawn every frame again was not recognised");
}

void TestOldTargetsAreForgotten() {
	RedrawnTargets targets;
	(void)targets.Draw(0x1000);
	for (int frame = 0; frame < 2000; frame++) {
		targets.NextFrame();
		Check(targets.Draw(0x2000) == (frame > 0), "a target drawn every frame was forgotten");
	}
	Check(!targets.Draw(0x1000), "a target from long ago counts as drawn every frame");
}

} // namespace

int main() {
	TestATargetDrawnEveryFrame();
	TestATargetDrawnNowAndThen();
	TestOldTargetsAreForgotten();
	std::puts("RedrawnTargetsTests: all cases passed");
	return 0;
}
