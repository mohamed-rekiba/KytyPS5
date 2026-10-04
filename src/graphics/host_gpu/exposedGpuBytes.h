#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
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
// - when guest memory holds something else than it would without a guest write, the guest wrote
//   the bytes itself: they are the guest's again, the upload takes them, and a GPU value that is
//   still on its way is dropped.
//
// `Land` may come from any thread; everything else is called on the GPU thread.
class ExposedGpuBytes final {
public:
	// The host GPU owns the bytes at `address`, and the guest can reach them without a fault.
	// `guest_now` is what guest memory holds there at this moment. `tick` names the host work
	// after which the GPU's value is copied to guest memory. Returns the name of this exposure,
	// for `Land`.
	uint64_t Expose(uint64_t address, std::span<const uint8_t> guest_now, uint64_t tick) {
		std::lock_guard lock(m_mutex);
		RemoveLocked(address, guest_now.size());
		const auto id      = ++m_last_id;
		m_entries[address] = {.size      = guest_now.size(),
		                      .id        = id,
		                      .tick      = tick,
		                      .pending   = true,
		                      .reference = {guest_now.begin(), guest_now.end()}};
		m_count.store(m_entries.size(), std::memory_order_relaxed);
		return id;
	}

	// The bytes in the range are not exposed any more: the page was taken from the guest again,
	// or guest memory is the current value. Exposed bytes next to the range stay exposed.
	void Forget(uint64_t address, uint64_t size) {
		if (Empty()) {
			return;
		}
		std::lock_guard lock(m_mutex);
		RemoveLocked(address, size);
		m_count.store(m_entries.size(), std::memory_order_relaxed);
	}

	void Clear() {
		std::lock_guard lock(m_mutex);
		m_entries.clear();
		m_count.store(0, std::memory_order_relaxed);
	}

	[[nodiscard]] bool Empty() const { return m_count.load(std::memory_order_relaxed) == 0; }

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		if (Empty()) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return FirstLocked(address) != m_entries.end() &&
		       FirstLocked(address)->first < address + size;
	}

	// The host work to wait for, so that guest memory holds the GPU's value of every exposed
	// byte in the range. Nothing when it holds them all.
	[[nodiscard]] std::optional<uint64_t> PendingTick(uint64_t address, uint64_t size) const {
		std::optional<uint64_t> tick;
		if (Empty()) {
			return tick;
		}
		std::lock_guard lock(m_mutex);
		for (auto entry = FirstLocked(address);
		     entry != m_entries.end() && entry->first < address + size; ++entry) {
			if (entry->second.pending) {
				tick = std::max(tick.value_or(0), entry->second.tick);
			}
		}
		return tick;
	}

	// The GPU's value of exposure `id` is ready: `bytes` belong at `address`. Calls
	// `write(address, bytes)` for each part that is still exposed from that exposure, so the
	// caller puts it into guest memory. A part that was forgotten, exposed anew, or taken back
	// by the guest in the meantime is not written: its value is older than what is there.
	template <typename Write>
	void Land(uint64_t id, uint64_t address, std::span<const uint8_t> bytes, Write&& write) {
		std::lock_guard lock(m_mutex);
		for (auto entry = FirstLocked(address);
		     entry != m_entries.end() && entry->first < address + bytes.size(); ++entry) {
			auto& item = entry->second;
			if (item.id != id || entry->first < address ||
			    entry->first + item.size > address + bytes.size()) {
				continue;
			}
			const auto part = bytes.subspan(entry->first - address, item.size);
			write(entry->first, part);
			item.reference.assign(part.begin(), part.end());
			item.pending = false;
		}
	}

	// Splits the guest range that is about to be uploaded. Calls `part(address, size)` for each
	// piece the upload may take from guest memory, in address order. `read_guest(address, out)`
	// fills `out` with the guest bytes at `address`.
	template <typename Read, typename Part>
	void ForEachUploadPart(uint64_t address, uint64_t size, Read&& read_guest, Part&& part) {
		if (Empty()) {
			part(address, size);
			return;
		}
		uint64_t next = address;
		{
			std::lock_guard lock(m_mutex);
			for (auto entry = FirstLocked(address);
			     entry != m_entries.end() && entry->first < address + size;) {
				auto& item = entry->second;
				// Guest memory holds something else than the bytes that were there when the
				// GPU took them, or than the GPU value that was copied there since: the guest
				// wrote them. They are the guest's again.
				m_scratch.resize(item.size);
				read_guest(entry->first, std::span<uint8_t> {m_scratch});
				if (m_scratch != item.reference) {
					entry = m_entries.erase(entry);
					continue;
				}
				const auto begin = std::max(entry->first, address);
				const auto end   = std::min(entry->first + item.size, address + size);
				if (next < begin) {
					part(next, begin - next);
				}
				next = end;
				++entry;
			}
			m_count.store(m_entries.size(), std::memory_order_relaxed);
		}
		if (next < address + size) {
			part(next, address + size - next);
		}
	}

private:
	struct Entry {
		uint64_t size = 0;
		uint64_t id   = 0; // the exposure it came from
		uint64_t tick = 0;
		// The GPU's value is not in guest memory yet.
		bool pending = true;
		// What guest memory holds while the guest does not write the bytes.
		std::vector<uint8_t> reference;
	};
	using Entries = std::map<uint64_t, Entry>;

	// The first entry that reaches past `address`.
	[[nodiscard]] Entries::const_iterator FirstLocked(uint64_t address) const {
		auto entry = m_entries.upper_bound(address);
		if (entry != m_entries.begin() &&
		    std::prev(entry)->first + std::prev(entry)->second.size > address) {
			--entry;
		}
		return entry;
	}
	[[nodiscard]] Entries::iterator FirstLocked(uint64_t address) {
		auto entry = m_entries.upper_bound(address);
		if (entry != m_entries.begin() &&
		    std::prev(entry)->first + std::prev(entry)->second.size > address) {
			--entry;
		}
		return entry;
	}

	// Takes the range out of the entries. An entry that reaches over an end of the range keeps
	// the part outside, with its share of the reference bytes.
	void RemoveLocked(uint64_t address, uint64_t size) {
		const auto end = address + size;
		for (auto entry = FirstLocked(address); entry != m_entries.end() && entry->first < end;) {
			const auto begin = entry->first;
			auto       item  = std::move(entry->second);
			entry            = m_entries.erase(entry);
			if (begin < address) {
				auto head = item;
				head.size = address - begin;
				head.reference.resize(head.size);
				m_entries[begin] = std::move(head);
			}
			if (begin + item.size > end) {
				auto tail = item;
				tail.size = begin + item.size - end;
				tail.reference.assign(item.reference.end() - static_cast<long>(tail.size),
				                      item.reference.end());
				entry = std::next(m_entries.insert_or_assign(end, std::move(tail)).first);
			}
		}
	}

	mutable std::mutex   m_mutex;
	Entries              m_entries; // start address -> exposed range; ranges do not overlap
	std::atomic<size_t>  m_count {0};
	uint64_t             m_last_id = 0;
	std::vector<uint8_t> m_scratch;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_EXPOSEDGPUBYTES_H_
