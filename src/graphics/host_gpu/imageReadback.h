#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_IMAGEREADBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_IMAGEREADBACK_H_

#include <cstdint>

namespace Libs::Graphics {

// Which images the GPU writes are copied back to guest memory after the submission that wrote
// them, so that the game's CPU can read them.
//
// A game reads back results the GPU computes: an engine can render the average brightness of a sky
// light into a 1x1 render target, tiled or linear, and read that pixel with the CPU. When guest
// memory keeps its old bytes, the CPU reads a brightness of 0, and lighting divided by it comes out
// white.
//
// So every image of at most READBACK_TINY_EXTENT pixels on each side, with one layer and one
// slice, is copied back, tiled or linear: such images are read back far more often than sampled
// again, and each copy is a few bytes. Block-compressed images are left out; the GPU does not
// render them. Larger linear images are copied back only when the setting asks for it
// (--readback-linear-images).
inline constexpr uint32_t READBACK_TINY_EXTENT = 4;

struct ReadbackImage {
	uint32_t width            = 1;
	uint32_t height           = 1;
	uint32_t depth            = 1;
	uint32_t layers           = 1;
	bool     tiled            = false;
	bool     block_compressed = false;
};

[[nodiscard]] constexpr bool ReadsBackToGuest(const ReadbackImage& image,
                                              bool readback_linear_images) noexcept {
	const bool tiny = image.width <= READBACK_TINY_EXTENT && image.height <= READBACK_TINY_EXTENT &&
	                  image.depth == 1 && image.layers == 1 && !image.block_compressed;
	return tiny || (readback_linear_images && !image.tiled);
}

// Whether a compute shader's write to a storage image is copied back when the CPU wrote the image
// since the GPU last did: a game can write a sentinel into a result and poll until the GPU
// replaces it. A linear image always is: the CPU reads linear results, and a tiled storage image
// is written back so often in some games that it costs frames, so it waits for
// --readback-storage-images.
[[nodiscard]] constexpr bool ReadsBackStorageWrite(bool tiled,
                                                   bool readback_storage_images) noexcept {
	return readback_storage_images || !tiled;
}

// The bytes of a downloaded image [begin, end) that are written to guest memory. Pages the image
// fills are always written. A page at either end that the image shares with other data is written
// only while the image still tracks it, [tracked_begin, tracked_end): the image stops tracking an
// end page when the CPU writes the other data in it, and from then on a CPU write to the image's
// bytes there goes unseen. `page` is a power of two.
struct PublishSpan {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

[[nodiscard]] constexpr PublishSpan PublishedPages(uint64_t begin, uint64_t end,
                                                   uint64_t tracked_begin, uint64_t tracked_end,
                                                   uint64_t page) noexcept {
	PublishSpan span {(begin + page - 1) & ~(page - 1), end & ~(page - 1)};
	if (tracked_begin <= begin && begin < tracked_end) {
		span.begin = begin;
	}
	if (tracked_begin < end && end <= tracked_end) {
		span.end = end;
	}
	return span;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_IMAGEREADBACK_H_ */
