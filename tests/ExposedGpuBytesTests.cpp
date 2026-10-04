// Tests for the few GPU-owned bytes that stay reachable for the guest: an upload must leave them
// out, until the guest writes them itself.

#include "graphics/host_gpu/exposedGpuBytes.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::ExposedGpuBytes;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		g_failures++;
	}
}

using Parts = std::vector<std::pair<uint64_t, uint64_t>>;

// A page of guest memory at address 0x1000.
struct Guest {
	static constexpr uint64_t Base  = 0x1000;
	std::vector<uint8_t>      bytes = std::vector<uint8_t>(0x1000, 0);

	void Write(uint64_t address, std::vector<uint8_t> value) {
		std::copy(value.begin(), value.end(), bytes.begin() + static_cast<long>(address - Base));
	}

	Parts UploadParts(ExposedGpuBytes& exposed, uint64_t address, uint64_t size) {
		Parts parts;
		exposed.ForEachUploadPart(
		    address, size,
		    [this](uint64_t from, std::span<uint8_t> out) {
			    std::copy_n(bytes.begin() + static_cast<long>(from - Base), out.size(),
			                out.begin());
		    },
		    [&parts](uint64_t from, uint64_t count) { parts.emplace_back(from, count); });
		return parts;
	}
};

void TestNothingExposed() {
	ExposedGpuBytes exposed;
	Guest           guest;
	Check(exposed.Empty() && !exposed.Intersects(0x1000, 0x1000), "a new set is empty");
	Check(guest.UploadParts(exposed, 0x1000, 0x1000) == Parts {{0x1000, 0x1000}},
	      "with nothing exposed the whole range is uploaded");
}

void TestUploadLeavesExposedBytesOut() {
	ExposedGpuBytes exposed;
	Guest           guest;
	exposed.Expose(0x1100, 4);
	Check(exposed.Intersects(0x1000, 0x1000) && !exposed.Intersects(0x1104, 8),
	      "the exposed bytes are found by address");
	Check(guest.UploadParts(exposed, 0x1000, 0x1000) == Parts {{0x1000, 0x100}, {0x1104, 0xefc}},
	      "an upload of the page goes around the exposed bytes");
	Check(guest.UploadParts(exposed, 0x1100, 4).empty(),
	      "an upload of only the exposed bytes uploads nothing");
	Check(guest.UploadParts(exposed, 0x10fe, 4) == Parts {{0x10fe, 2}},
	      "an upload that ends inside the exposed bytes stops in front of them");
	Check(guest.UploadParts(exposed, 0x1102, 6) == Parts {{0x1104, 4}},
	      "an upload that starts inside the exposed bytes starts behind them");

	exposed.Expose(0x1200, 8);
	Check(guest.UploadParts(exposed, 0x1000, 0x1000) ==
	          Parts {{0x1000, 0x100}, {0x1104, 0xfc}, {0x1208, 0xdf8}},
	      "two exposed ranges make three parts");
}

void TestGuestWriteTakesTheBytesBack() {
	ExposedGpuBytes exposed;
	Guest           guest;
	exposed.Expose(0x1100, 4);
	guest.Write(0x1100, {9, 9, 9, 9});
	Check(guest.UploadParts(exposed, 0x1000, 0x1000).size() == 2,
	      "before the GPU value was copied to guest memory, nothing shows a guest write");

	// The host copies the GPU's value to guest memory.
	guest.Write(0x1100, {1, 2, 3, 4});
	exposed.Landed(0x1100, std::vector<uint8_t> {1, 2, 3, 4});
	Check(guest.UploadParts(exposed, 0x1000, 0x1000).size() == 2,
	      "guest memory holds the copied value: the GPU still owns the bytes");

	// The guest writes the bytes itself.
	guest.Write(0x1100, {1, 2, 3, 5});
	Check(guest.UploadParts(exposed, 0x1000, 0x1000) == Parts {{0x1000, 0x1000}},
	      "guest memory holds another value: the upload takes the guest's bytes");
	Check(!exposed.Intersects(0x1100, 4), "and the bytes are not exposed any more");
}

void TestNewGpuWriteDropsTheCopiedValue() {
	ExposedGpuBytes exposed;
	Guest           guest;
	exposed.Expose(0x1100, 4);
	guest.Write(0x1100, {1, 2, 3, 4});
	exposed.Landed(0x1100, std::vector<uint8_t> {1, 2, 3, 4});

	// The GPU writes again and the bytes are exposed again: the copied value is the old one, and
	// guest memory still holds it. That must not look like a guest write.
	exposed.Expose(0x1100, 4);
	guest.Write(0x1100, {7, 7, 7, 7}); // would differ from the old value
	Check(guest.UploadParts(exposed, 0x1000, 0x1000).size() == 2,
	      "after a new GPU write the old copied value proves nothing");

	exposed.Forget(0x1000, 0x1000);
	Check(exposed.Empty(), "forgetting the page removes the range");
	exposed.Landed(0x1100, std::vector<uint8_t> {7, 7, 7, 7}); // a late copy
	exposed.Clear();
	Check(guest.UploadParts(exposed, 0x1000, 0x1000) == Parts {{0x1000, 0x1000}},
	      "a cleared set leaves nothing out");
}

} // namespace

int main() {
	TestNothingExposed();
	TestUploadLeavesExposedBytesOut();
	TestGuestWriteTakesTheBytesBack();
	TestNewGpuWriteDropsTheCopiedValue();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("exposed gpu bytes tests passed\n");
	return 0;
}
