// Tests for the few GPU-owned bytes that stay reachable for the guest: an upload must leave them
// out until the guest writes them itself, and a GPU value that arrives late must not undo what
// the guest or a newer GPU write did.

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

// A page of guest memory at address 0x1000, and the exposed bytes in it.
struct Guest {
	static constexpr uint64_t Base  = 0x1000;
	std::vector<uint8_t>      bytes = std::vector<uint8_t>(0x1000, 0);
	ExposedGpuBytes           exposed;

	[[nodiscard]] std::span<const uint8_t> At(uint64_t address, uint64_t size) const {
		return {bytes.data() + (address - Base), size};
	}

	void Write(uint64_t address, std::vector<uint8_t> value) {
		std::copy(value.begin(), value.end(), bytes.begin() + static_cast<long>(address - Base));
	}

	uint64_t Expose(uint64_t address, uint64_t size, uint64_t tick = 1) {
		return exposed.Expose(address, At(address, size), tick);
	}

	// The host copies the GPU's value to guest memory, as far as the exposure still stands.
	void Land(uint64_t id, uint64_t address, std::vector<uint8_t> value) {
		exposed.Land(id, address, value, [this](uint64_t to, std::span<const uint8_t> part) {
			std::copy(part.begin(), part.end(), bytes.begin() + static_cast<long>(to - Base));
		});
	}

	Parts UploadParts(uint64_t address, uint64_t size) {
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

const Parts WholePage     = {{0x1000, 0x1000}};
const Parts AroundCounter = {{0x1000, 0x100}, {0x1104, 0xefc}};

void TestNothingExposed() {
	Guest guest;
	Check(guest.exposed.Empty() && !guest.exposed.Intersects(0x1000, 0x1000), "a new set is empty");
	Check(guest.UploadParts(0x1000, 0x1000) == WholePage,
	      "with nothing exposed the whole range is uploaded");
	Check(!guest.exposed.PendingTick(0x1000, 0x1000), "and nothing is on its way");
}

void TestUploadLeavesExposedBytesOut() {
	Guest guest;
	(void)guest.Expose(0x1100, 4);
	Check(guest.exposed.Intersects(0x1000, 0x1000) && guest.exposed.Intersects(0x1103, 1) &&
	          !guest.exposed.Intersects(0x1104, 8) && !guest.exposed.Intersects(0x10f0, 0x10),
	      "the exposed bytes are found by address");
	Check(guest.UploadParts(0x1000, 0x1000) == AroundCounter,
	      "an upload of the page goes around the exposed bytes");
	Check(guest.UploadParts(0x1100, 4).empty(),
	      "an upload of only the exposed bytes uploads nothing");
	Check(guest.UploadParts(0x10fe, 4) == Parts {{0x10fe, 2}},
	      "an upload that ends inside the exposed bytes stops in front of them");
	Check(guest.UploadParts(0x1102, 6) == Parts {{0x1104, 4}},
	      "an upload that starts inside the exposed bytes starts behind them");

	(void)guest.Expose(0x1104, 8); // touches the first range
	Check(guest.UploadParts(0x1000, 0x1000) == Parts {{0x1000, 0x100}, {0x110c, 0xef4}},
	      "two ranges that touch are both left out");
}

void TestGuestWriteAfterTheValueArrived() {
	Guest      guest;
	const auto id = guest.Expose(0x1100, 4, 7);
	Check(guest.exposed.PendingTick(0x1000, 0x1000) == 7u, "the GPU's value is on its way");
	guest.Land(id, 0x1100, {1, 2, 3, 4});
	Check(guest.At(0x1100, 4)[3] == 4, "the value arrives in guest memory");
	Check(!guest.exposed.PendingTick(0x1000, 0x1000), "nothing is on its way any more");
	Check(guest.UploadParts(0x1000, 0x1000) == AroundCounter,
	      "guest memory holds the GPU's value: the GPU still owns the bytes");

	guest.Write(0x1100, {1, 2, 3, 5});
	Check(guest.UploadParts(0x1102, 6) == Parts {{0x1102, 6}},
	      "the guest wrote the bytes: an upload takes them, also one that starts inside them");
	Check(!guest.exposed.Intersects(0x1100, 4), "and the bytes are not exposed any more");
}

void TestGuestWriteBeforeTheValueArrives() {
	Guest guest;
	guest.Write(0x1100, {5, 5, 5, 5});
	const auto id = guest.Expose(0x1100, 4);
	Check(guest.UploadParts(0x1000, 0x1000) == AroundCounter,
	      "guest memory is unchanged since the exposure: the GPU owns the bytes");

	guest.Write(0x1100, {9, 9, 9, 9});
	Check(guest.UploadParts(0x1000, 0x1000) == WholePage,
	      "the guest wrote before the GPU's value arrived: the upload takes the guest's bytes");
	guest.Land(id, 0x1100, {1, 2, 3, 4});
	Check(guest.At(0x1100, 4)[0] == 9, "and the late GPU value does not overwrite them");
}

void TestNewGpuWriteDropsTheOldValue() {
	Guest      guest;
	const auto first = guest.Expose(0x1100, 4);
	// The GPU writes again, the page is taken and exposed again: a new exposure.
	guest.exposed.Forget(0x1100, 4);
	const auto second = guest.Expose(0x1100, 4);
	guest.Land(first, 0x1100, {1, 1, 1, 1});
	Check(guest.At(0x1100, 4)[0] == 0, "the value of the older exposure is not written");
	Check(guest.exposed.PendingTick(0x1100, 4).has_value(), "the newer value is still on its way");
	guest.Land(second, 0x1100, {2, 2, 2, 2});
	Check(guest.At(0x1100, 4)[0] == 2 && guest.UploadParts(0x1000, 0x1000) == AroundCounter,
	      "the value of the newer exposure arrives, and the GPU keeps the bytes");
}

void TestPartOfARangeIsForgotten() {
	Guest guest;
	guest.Write(0x1100, {1, 2, 3, 4, 5, 6, 7, 8});
	const auto id = guest.Expose(0x1100, 8);
	guest.exposed.Forget(0x1102, 2); // a new GPU write covers the middle
	Check(guest.UploadParts(0x1000, 0x1000) ==
	          Parts {{0x1000, 0x100}, {0x1102, 2}, {0x1108, 0xef8}},
	      "the parts on both sides stay exposed");
	guest.Land(id, 0x1100, {11, 12, 13, 14, 15, 16, 17, 18});
	const auto now = guest.At(0x1100, 8);
	Check(now[0] == 11 && now[1] == 12 && now[2] == 3 && now[3] == 4 && now[4] == 15 &&
	          now[7] == 18,
	      "the value arrives in the parts that are still exposed, and only there");

	guest.Write(0x1106, {99});
	Check(guest.UploadParts(0x1000, 0x1000) == Parts {{0x1000, 0x100}, {0x1102, 0xefe}},
	      "a guest write to one part takes that part back, the other part stays");
	guest.exposed.Clear();
	Check(guest.UploadParts(0x1000, 0x1000) == WholePage, "a cleared set leaves nothing out");
}

} // namespace

int main() {
	TestNothingExposed();
	TestUploadLeavesExposedBytesOut();
	TestGuestWriteAfterTheValueArrived();
	TestGuestWriteBeforeTheValueArrives();
	TestNewGpuWriteDropsTheOldValue();
	TestPartOfARangeIsForgotten();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("exposed gpu bytes tests passed\n");
	return 0;
}
