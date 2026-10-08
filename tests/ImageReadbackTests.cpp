// Which GPU-written images are copied back to guest memory for the CPU.
#include "graphics/host_gpu/imageReadback.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::ReadsBackToGuest;

void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "ImageReadbackTests: %s\n", text);
		std::abort();
	}
}

void TestTinyImagesAreReadBack() {
	// The 1x1 target Unreal Engine renders a sky light's average brightness into is tiled.
	Check(ReadsBackToGuest(1, 1, true, false, false), "a tiled 1x1 target was not read back");
	Check(ReadsBackToGuest(4, 4, true, false, false), "a tiled 4x4 target was not read back");
	Check(ReadsBackToGuest(1, 1, false, false, false), "a linear 1x1 target was not read back");
	Check(!ReadsBackToGuest(5, 1, true, false, false), "a 5x1 target counted as tiny");
	Check(!ReadsBackToGuest(1, 5, true, false, false), "a 1x5 target counted as tiny");
	Check(!ReadsBackToGuest(4, 4, true, true, false),
	      "a block-compressed image counted as a readback target");
}

void TestLargerImagesFollowTheSetting() {
	Check(!ReadsBackToGuest(64, 64, false, false, false),
	      "a linear 64x64 image was read back with the setting off");
	Check(ReadsBackToGuest(64, 64, false, false, true),
	      "a linear 64x64 image was not read back with the setting on");
	Check(!ReadsBackToGuest(64, 64, true, false, true),
	      "a tiled 64x64 image was read back by the linear setting");
}

} // namespace

int main() {
	TestTinyImagesAreReadBack();
	TestLargerImagesFollowTheSetting();
	std::puts("ImageReadbackTests: passed");
	return 0;
}
