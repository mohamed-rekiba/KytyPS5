#ifndef EMULATOR_SRC_COMMON_WAITBUDGET_H_
#define EMULATOR_SRC_COMMON_WAITBUDGET_H_

#include <algorithm>
#include <chrono>

namespace Common {

// How long a thread may still wait for background work without becoming a visible stop.
// It may wait up to `capacity` at once, and what it used comes back at `capacity` per `period`.
// Short waits cost nothing noticeable and keep results exact; when many waits come together,
// the budget runs out and the caller has to go on without the result.
class WaitBudget final {
public:
	using Clock    = std::chrono::steady_clock;
	using Duration = Clock::duration;

	WaitBudget(Duration capacity, Duration period)
	    : m_capacity(capacity), m_period(period), m_left(capacity) {}

	// What may be waited at `now`.
	[[nodiscard]] Duration Left(Clock::time_point now) {
		Refill(now);
		return m_left;
	}

	// The caller waited `used`, ending at `now`.
	void Spend(Duration used, Clock::time_point now) {
		Refill(now);
		m_left -= std::min(used, m_left);
	}

private:
	void Refill(Clock::time_point now) {
		if (m_last != Clock::time_point {} && now > m_last) {
			const auto back = std::chrono::duration_cast<Duration>(
			    (now - m_last) *
			    (static_cast<double>(m_capacity.count()) / static_cast<double>(m_period.count())));
			m_left = std::min(m_capacity, m_left + back);
		}
		m_last = std::max(m_last, now);
	}

	Duration          m_capacity;
	Duration          m_period;
	Duration          m_left;
	Clock::time_point m_last {};
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_WAITBUDGET_H_
