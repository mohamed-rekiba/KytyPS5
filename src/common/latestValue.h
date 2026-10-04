#ifndef EMULATOR_SRC_COMMON_LATESTVALUE_H_
#define EMULATOR_SRC_COMMON_LATESTVALUE_H_

#include <mutex>
#include <optional>
#include <utility>

namespace Common {

// A slot for a value that one thread produces often and another thread shows when it gets to
// it. The producer never waits: a new value replaces one that was not taken yet. `Put` tells the
// producer when to ask the consumer to come: only when none is on its way.
template <typename T>
class LatestValue final {
public:
	// Stores `value`. True when the caller has to schedule a consumer that calls `Take`.
	[[nodiscard]] bool Put(T value) {
		std::lock_guard lock(m_mutex);
		m_value = std::move(value);
		return !std::exchange(m_consumer_pending, true);
	}

	// The consumer's call: the newest value, if one was put since the last call.
	[[nodiscard]] std::optional<T> Take() {
		std::lock_guard lock(m_mutex);
		m_consumer_pending = false;
		return std::exchange(m_value, std::nullopt);
	}

private:
	std::mutex       m_mutex;
	std::optional<T> m_value;
	bool             m_consumer_pending = false;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LATESTVALUE_H_
