#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Guest bytes whose current value only the host GPU has, inside pages the guest may read and
// write without a fault.
//
// The usual way to keep such bytes safe is to take the whole page away from the guest: its next
// access faults, and the fault waits until the GPU's bytes are back in guest memory. That is right
// when the guest wants those bytes. It is costly when a shader writes a counter of a few bytes
// into a page of the guest's own variables: every read of a neighbour then waits for the GPU.
//
// So a few GPU-owned bytes can be exposed instead. The page goes back to the guest. The GPU's
// value is copied to guest memory when the GPU has produced it, without anyone waiting, as on the
// real machine. Until the GPU writes the bytes again, two rules keep both sides right:
// - an upload of the page to the GPU leaves the exposed bytes out, so the GPU's value stays;
// - when guest memory holds something else than the value that was copied there, the guest wrote
//   the bytes itself: they are the guest's again, and the upload takes them.
//
// The ranges are used on the GPU thread only. `Landed` may come from any thread.
class ExposedGpuBytes final {
public:
	// The host GPU owns `size` bytes at `address`, and the guest can reach them without a fault.
	void Expose(uint64_t address, uint64_t size) {
		m_ranges.Add(address, size);
		std::lock_guard lock(m_mutex);
		// A value copied earlier is not what the GPU has now.
		m_landed.erase(address);
	}

	// The bytes in the range are not exposed any more: the page was taken from the guest again,
	// or guest memory is the current value.
	void Forget(uint64_t address, uint64_t size) {
		if (!m_ranges.Intersects(address, size)) {
			return;
		}
		m_ranges.Subtract(address, size);
		std::lock_guard lock(m_mutex);
		for (auto item = m_landed.lower_bound(address);
		     item != m_landed.end() && item->first < address + size;) {
			item = m_landed.erase(item);
		}
	}

	void Clear() {
		m_ranges.Clear();
		std::lock_guard lock(m_mutex);
		m_landed.clear();
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		return m_ranges.Intersects(address, size);
	}

	[[nodiscard]] bool Empty() const { return m_ranges.Empty(); }

	// The GPU's value of an exposed range was copied to guest memory: `bytes` at `address`.
	void Landed(uint64_t address, std::span<const uint8_t> bytes) {
		std::lock_guard lock(m_mutex);
		m_landed[address].assign(bytes.begin(), bytes.end());
	}

	// Splits the guest range that is about to be uploaded. Calls `part(address, size)` for each
	// piece the upload may take from guest memory, in address order. `read_guest(address, out)`
	// fills `out` with the guest bytes at `address`.
	template <typename Read, typename Part>
	void ForEachUploadPart(uint64_t address, uint64_t size, Read&& read_guest, Part&& part) {
		if (!m_ranges.Intersects(address, size)) {
			part(address, size);
			return;
		}
		// Collected first: a range the guest took back is removed from the set.
		std::vector<std::pair<uint64_t, uint64_t>> exposed;
		m_ranges.ForEachInRange(
		    address, size, [&](uint64_t begin, uint64_t end) { exposed.emplace_back(begin, end); });
		uint64_t next = address;
		for (const auto& [begin, end]: exposed) {
			if (GuestWrote(begin, end - begin, read_guest)) {
				Forget(begin, end - begin);
				continue;
			}
			if (next < begin) {
				part(next, begin - next);
			}
			next = end;
		}
		if (next < address + size) {
			part(next, address + size - next);
		}
	}

private:
	// True when guest memory holds something else than the GPU value that was copied there.
	// False while no value was copied yet, or when only a part of the range is asked for: then
	// nothing shows that the guest wrote, and the GPU keeps the bytes.
	template <typename Read>
	[[nodiscard]] bool GuestWrote(uint64_t address, uint64_t size, Read&& read_guest) {
		std::lock_guard lock(m_mutex);
		const auto      landed = m_landed.find(address);
		if (landed == m_landed.end() || landed->second.size() != size) {
			return false;
		}
		m_scratch.resize(size);
		read_guest(address, std::span<uint8_t> {m_scratch});
		return std::memcmp(m_scratch.data(), landed->second.data(), size) != 0;
	}

	RangeSet                                 m_ranges;
	std::mutex                               m_mutex;
	std::map<uint64_t, std::vector<uint8_t>> m_landed; // start of a range -> value copied there
	std::vector<uint8_t>                     m_scratch;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_
