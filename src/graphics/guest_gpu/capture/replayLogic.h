#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_REPLAYLOGIC_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_REPLAYLOGIC_H_

// The decisions of the capture player that need no renderer and no guest memory, so they can be
// tested on their own.

#include "graphics/guest_gpu/capture/captureFile.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::Capture {

// Calls `func(offset, size)` for each run of bytes in which `wanted` differs from `current`.
// Runs with fewer than `merge_gap` equal bytes between them are reported as one run.
template <typename Func>
void ForEachChangedRun(std::span<const uint8_t> current, std::span<const uint8_t> wanted,
                       size_t merge_gap, Func&& func) {
	const size_t size = std::min(current.size(), wanted.size());
	// Equal stretches are skipped a word at a time.
	const auto first_difference = [&](size_t from) {
		size_t i = from;
		for (; i + sizeof(uint64_t) <= size; i += sizeof(uint64_t)) {
			if (std::memcmp(current.data() + i, wanted.data() + i, sizeof(uint64_t)) != 0) {
				break;
			}
		}
		while (i < size && current[i] == wanted[i]) {
			i++;
		}
		return i;
	};
	for (size_t begin = first_difference(0); begin < size;) {
		size_t end = begin + 1;
		for (;;) {
			const size_t next = first_difference(end);
			if (next >= size || next - end >= merge_gap) {
				func(begin, end - begin);
				begin = next;
				break;
			}
			end = next + 1;
		}
	}
}

// The waits on guest memory of one pass, as the live run saw them pass. The command stream of a
// replay is the same as in the live run, so the waits on one address come in the same order: the
// next wait the GPU thread meets on an address is the oldest recorded one that has not passed.
class WaitSchedule final {
public:
	struct Wait {
		uint64_t address = 0;
		// The wait failed its test in the live run before it passed.
		bool had_blocked = false;
	};

	// Starts a pass: every wait of it, in recorded order.
	void Arm(std::span<const Wait> waits) {
		m_waits.assign(waits.begin(), waits.end());
		m_released.assign(m_waits.size(), false);
		m_order.clear();
		for (size_t i = 0; i < m_waits.size(); i++) {
			m_order[m_waits[i].address].push_back(i);
		}
	}

	// The replay has reached the point of the stream where the live run saw wait `index` pass.
	void Release(size_t index) {
		if (index < m_released.size()) {
			m_released[index] = true;
		}
	}

	// A wait on `address` is about to be tested. Returns the recorded wait whose memory must be
	// in place before the test: one that passed at once in the live run.
	[[nodiscard]] std::optional<size_t> BeforeTest(uint64_t address) const {
		const auto next = Next(address);
		return next && !m_waits[*next].had_blocked ? next : std::nullopt;
	}

	// A wait on `address` failed its test. Returns the recorded wait to serve now, or nothing
	// when the wait has to stay blocked: the capture has no wait left for the address, or the
	// replay is not yet where the live run released it.
	[[nodiscard]] std::optional<size_t> Blocked(uint64_t address) const {
		const auto next = Next(address);
		return next && (!m_waits[*next].had_blocked || m_released[*next]) ? next : std::nullopt;
	}

	// A wait on `address` passed. Returns the recorded wait it was, and moves on to the next one
	// on that address.
	[[nodiscard]] std::optional<size_t> Passed(uint64_t address) {
		const auto order = m_order.find(address);
		if (order == m_order.end() || order->second.empty()) {
			return std::nullopt;
		}
		const size_t index = order->second.front();
		order->second.pop_front();
		return index;
	}

private:
	[[nodiscard]] std::optional<size_t> Next(uint64_t address) const {
		const auto order = m_order.find(address);
		if (order == m_order.end() || order->second.empty()) {
			return std::nullopt;
		}
		return order->second.front();
	}

	std::vector<Wait>                                m_waits;
	std::vector<bool>                                m_released;
	std::unordered_map<uint64_t, std::deque<size_t>> m_order;
};

// A wait that the live run saw pass can fail its test in a replay for two reasons. The guest
// wrote the awaited bytes: a replay has no guest, so the player must write them. Or another
// queue of the GPU thread writes them, and has not come to that point yet: then the player must
// keep out, or the waiting queue runs ahead of the work it waited for. The two are told apart by
// what the GPU thread does between two tests of the wait: it tests a blocked wait again and
// again, and in between it runs whatever else can run. When it comes back to the wait and has
// done no work at all, nothing else can run, and only the player can release the wait.
class StalledWaits final {
public:
	// The GPU thread did some work of the stream: a draw, a dispatch, a submission start, a
	// wait that passed, a frame end.
	void Progress() { m_progress++; }

	// Wait `index` failed its test again. True when no work was done since it failed last time.
	[[nodiscard]] bool NothingElseRan(size_t index) {
		const auto [seen, first] = m_seen.try_emplace(index, m_progress);
		if (first) {
			return false;
		}
		const bool stalled = seen->second == m_progress;
		seen->second       = m_progress;
		return stalled;
	}

	// Starts a pass.
	void Reset() {
		m_seen.clear();
		m_progress = 0;
	}

private:
	std::unordered_map<size_t, uint64_t> m_seen; // work counter when the wait failed last
	uint64_t                             m_progress = 0;
};

// How far two pictures of the same size are apart.
struct PictureDifference {
	uint64_t bytes   = 0; // bytes that differ
	uint32_t largest = 0; // largest difference of one byte
};

[[nodiscard]] inline PictureDifference ComparePictures(std::span<const uint8_t> a,
                                                       std::span<const uint8_t> b) {
	PictureDifference difference;
	const size_t      size = std::min(a.size(), b.size());
	for (size_t i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			difference.bytes++;
			const uint32_t delta = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
			difference.largest   = std::max(difference.largest, delta);
		}
	}
	difference.bytes += std::max(a.size(), b.size()) - size;
	return difference;
}

inline constexpr size_t NoOwner = static_cast<size_t>(-1);

// The records at which the recorder stored what had changed in guest memory.
[[nodiscard]] constexpr bool IsMemoryPoint(RecordType type) {
	return type == RecordType::Submission || type == RecordType::WaitBytes ||
	       type == RecordType::FrameEnd || type == RecordType::ReadPoint;
}

// A memory record (`Pages`, `GuestWrites`, `Mapping`) belongs to the next point of the stream at
// which the recorder stored guest memory: a submission start, a wait that passed, a frame end, or a
// read of changed memory by the renderer. Returns, for each record of a frame sequence, the index
// of the record it belongs to; `NoOwner` for a record that is not a memory record or has no such
// point after it.
[[nodiscard]] inline std::vector<size_t> MemoryRecordOwners(std::span<const RecordType> types) {
	std::vector<size_t> owners(types.size(), NoOwner);
	size_t              owner = NoOwner;
	for (size_t i = types.size(); i-- > 0;) {
		const auto type = types[i];
		if (IsMemoryPoint(type)) {
			owner = i;
		} else if (type == RecordType::Pages || type == RecordType::GuestWrites ||
		           type == RecordType::Mapping) {
			owners[i] = owner;
		}
	}
	return owners;
}

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_REPLAYLOGIC_H_
