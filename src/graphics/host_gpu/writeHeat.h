#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEHEAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEHEAT_H_

#include "common/bitArray.h"

#include <array>
#include <cstddef>
#include <cstdint>

// Which pages of a tracked region the guest writes again and again.
//
// A clean page is write-protected, so that the first write to it is seen. A game that rewrites a
// buffer many times per frame then pays, for each page and each time: a fault, a protection
// change in the fault handler, and another one when the renderer has uploaded the page. A page
// that keeps coming back like this is "hot": it stays open and counts as written, and the
// renderer uploads it at every use. That is always correct, since an upload of unchanged bytes
// changes nothing; it trades the faults for copies.
//
// The heat is a lease of LeasePasses upload passes (RegionManager::NoteUploadPass: one pass for
// each draw or dispatch that reads memory through device addresses). At the first upload after
// the lease, the page is clean again: it is protected before its bytes are copied, so a later
// write faults and is seen. A game that writes a ring buffer touches each page once per lap; a
// lease keeps such a page from being uploaded at every pass for seconds after its last write.
//
// Heat is also forgotten from time to time (CoolAll), for pages that no pass uploads.

namespace Libs::Graphics {

template <size_t Pages>
class WriteHeat final {
public:
	using Bits = Common::BitArray<Pages>;

	// Writes to a clean page, since the last cooling, that make it hot.
	static constexpr uint8_t HotAfter = 2;
	// Upload passes a page stays hot for.
	static constexpr uint32_t LeasePasses = 2;

	// The CPU wrote to pages [start, end) that the tracker had as clean, during upload pass
	// `pass`.
	constexpr void NoteWrite(size_t start, size_t end, uint32_t pass) {
		for (size_t page = start; page < end; page++) {
			if (m_writes[page] < HotAfter && ++m_writes[page] == HotAfter) {
				m_hot.Set(page);
				m_lease_end[page] = pass + LeasePasses;
			}
		}
	}

	[[nodiscard]] constexpr bool IsHot(size_t page) const { return m_hot.Get(page); }

	// After the caller cleared [start, end) in `dirty` for an upload in pass `pass`: the hot
	// pages stay dirty. A page whose lease has ended loses its heat and stays clean.
	constexpr void KeepHotDirty(Bits& dirty, size_t start, size_t end, uint32_t pass) {
		if (m_hot.None()) {
			return;
		}
		for (size_t page = start; page < end; page++) {
			if (!m_hot.Get(page)) {
				continue;
			}
			// The counter wraps; the difference tells whether `pass` reached the end.
			if (static_cast<int32_t>(pass - m_lease_end[page]) >= 0) {
				m_hot.Unset(page);
				m_writes[page] = 0;
			} else {
				dirty.Set(page);
			}
		}
	}

	// The pages are no longer the CPU's to write at will, for example because the GPU wrote
	// them. Returns true when any of them was hot.
	constexpr bool Cool(size_t start, size_t end) {
		bool any = false;
		for (size_t page = start; page < end; page++) {
			any            = any || m_hot.Get(page);
			m_writes[page] = 0;
		}
		if (any) {
			m_hot.UnsetRange(start, end);
		}
		return any;
	}

	constexpr void CoolAll() {
		m_writes.fill(0);
		m_hot.Clear();
	}

private:
	std::array<uint8_t, Pages>  m_writes {};
	std::array<uint32_t, Pages> m_lease_end {};
	Bits                        m_hot;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEHEAT_H_
