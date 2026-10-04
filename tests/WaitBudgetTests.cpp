#include "common/waitBudget.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace std::chrono_literals;
using Common::WaitBudget;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WaitBudgetTests: failed: %s\n", message);
		std::abort();
	}
}

void TestRunsOutAndComesBack() {
	WaitBudget       budget(8ms, 100ms);
	const auto       start = WaitBudget::Clock::time_point {} + 1s;
	Check(budget.Left(start) == 8ms, "a new budget is not full");
	budget.Spend(3ms, start + 3ms);
	Check(budget.Left(start + 3ms) < 6ms && budget.Left(start + 3ms) > 4ms,
	      "a short wait did not take its time from the budget");
	budget.Spend(20ms, start + 23ms);
	Check(budget.Left(start + 23ms) == 0ms, "a long wait left budget over");
	const auto half = budget.Left(start + 73ms);
	Check(half > 3ms && half < 5ms, "half a period did not give half the budget back");
	Check(budget.Left(start + 10s) == 8ms, "the budget grew past its capacity or stayed empty");
}

void TestTimeDoesNotRunBackwards() {
	WaitBudget budget(8ms, 100ms);
	const auto start = WaitBudget::Clock::time_point {} + 1s;
	budget.Spend(8ms, start);
	Check(budget.Left(start - 50ms) == 0ms, "an earlier time gave budget back");
}

} // namespace

int main() {
	TestRunsOutAndComesBack();
	TestTimeDoesNotRunBackwards();
	std::puts("WaitBudgetTests: all cases passed");
	return 0;
}
