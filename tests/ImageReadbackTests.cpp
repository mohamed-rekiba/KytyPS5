// Which GPU-written images are copied back to guest memory for the CPU.
#include "graphics/host_gpu/imageReadback.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::PublishedPages;
using Libs::Graphics::ReadsBackToGuest;

void Check(bool condition, const char *text) {
  if (!condition) {
    std::fprintf(stderr, "ImageReadbackTests: %s\n", text);
    std::abort();
  }
}

void TestTinyImagesAreReadBack() {
  // A 1x1 target that holds a sky light's average brightness can be tiled.
  Check(ReadsBackToGuest({.width = 1, .height = 1, .tiled = true}, false),
        "a tiled 1x1 target was not read back");
  Check(ReadsBackToGuest({.width = 4, .height = 4, .tiled = true}, false),
        "a tiled 4x4 target was not read back");
  Check(ReadsBackToGuest({.width = 1, .height = 1}, false),
        "a linear 1x1 target was not read back");
  Check(!ReadsBackToGuest({.width = 5, .height = 1, .tiled = true}, false),
        "a 5x1 target counted as tiny");
  Check(!ReadsBackToGuest({.width = 1, .height = 5, .tiled = true}, false),
        "a 1x5 target counted as tiny");
  Check(!ReadsBackToGuest(
            {.width = 4, .height = 4, .tiled = true, .block_compressed = true},
            false),
        "a block-compressed image counted as a readback target");
}

// A tiny array or volume image is not a few bytes: each of its layers or slices
// is copied.
void TestTinyArraysAndVolumesAreNotReadBack() {
  Check(!ReadsBackToGuest(
            {.width = 4, .height = 4, .layers = 64, .tiled = true}, false),
        "a 4x4 array of 64 layers counted as tiny");
  Check(!ReadsBackToGuest({.width = 4, .height = 4, .depth = 64, .tiled = true},
                          false),
        "a 4x4x64 volume counted as tiny");
}

void TestLargerImagesFollowTheSetting() {
  Check(!ReadsBackToGuest({.width = 64, .height = 64}, false),
        "a linear 64x64 image was read back with the setting off");
  Check(ReadsBackToGuest({.width = 64, .height = 64}, true),
        "a linear 64x64 image was not read back with the setting on");
  Check(!ReadsBackToGuest({.width = 64, .height = 64, .tiled = true}, true),
        "a tiled 64x64 image was read back by the linear setting");
}

} // namespace

// A game polls a result the GPU writes in the last bytes of an image that ends
// inside a page: that page is written back while the image still tracks it.
void TestTrackedEndPagesArePublished() {
  constexpr uint64_t page = 0x1000;
  constexpr uint64_t begin = 0x10d00;
  constexpr uint64_t end = 0x61d00;
  const auto tracked = PublishedPages(begin, end, begin, end, page);
  Check(tracked.begin == begin && tracked.end == end,
        "a tracked image did not publish its end pages");
  const auto head_owned = PublishedPages(begin, end, 0x11000, end, page);
  Check(head_owned.begin == 0x11000 && head_owned.end == end,
        "a head page the CPU owns was published");
  const auto tail_owned = PublishedPages(begin, end, begin, 0x61000, page);
  Check(tail_owned.begin == begin && tail_owned.end == 0x61000,
        "a tail page the CPU owns was published");
  const auto untracked = PublishedPages(begin, end, 0, 0, page);
  Check(untracked.begin == 0x11000 && untracked.end == 0x61000,
        "an untracked image published a shared end page");
  const auto aligned = PublishedPages(0x10000, 0x62000, 0, 0, page);
  Check(aligned.begin == 0x10000 && aligned.end == 0x62000,
        "an image of whole pages lost a page");
}

int main() {
  TestTinyImagesAreReadBack();
  TestTinyArraysAndVolumesAreNotReadBack();
  TestLargerImagesFollowTheSetting();
  TestTrackedEndPagesArePublished();
  std::puts("ImageReadbackTests: passed");
  return 0;
}
