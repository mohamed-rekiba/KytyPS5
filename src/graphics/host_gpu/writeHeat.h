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
// Heat is forgotten from time to time (CoolAll), so that a page the guest has stopped writing is
// protected again and its uploads stop.

namespace Libs::Graphics {

template <size_t Pages>
class WriteHeat final {
public:
	using Bits = Common::BitArray<Pages>;

	// Writes to a clean page, since the last cooling, that make it hot.
	static constexpr uint8_t HotAfter = 2;

	// The CPU wrote to pages [start, end) that the tracker had as clean.
	constexpr void NoteWrite(size_t start, size_t end) {
		for (size_t page = start; page < end; page++) {
			if (m_writes[page] < HotAfter && ++m_writes[page] == HotAfter) {
				m_hot.Set(page);
			}
		}
	}

	[[nodiscard]] constexpr bool IsHot(size_t page) const { return m_hot.Get(page); }

	// After the caller cleared [start, end) in `dirty`: the hot pages stay dirty.
	constexpr void KeepHotDirty(Bits& dirty, size_t start, size_t end) const {
		if (m_hot.None()) {
			return;
		}
		for (size_t page = start; page < end; page++) {
			if (m_hot.Get(page)) {
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
	std::array<uint8_t, Pages> m_writes {};
	Bits                       m_hot;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_WRITEHEAT_H_
