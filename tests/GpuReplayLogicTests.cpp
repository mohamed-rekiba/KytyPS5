// Tests for the decisions of the capture player that need no renderer: which bytes to write,
// which recorded wait a wait of the replay is, when a blocked wait may be released, which point
// of the stream a memory record belongs to, and how far two pictures are apart.

#include "graphics/guest_gpu/capture/replayLogic.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::Capture;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		g_failures++;
	}
}

using Runs = std::vector<std::pair<size_t, size_t>>;

Runs ChangedRuns(const std::vector<uint8_t>& current, const std::vector<uint8_t>& wanted,
                 size_t merge_gap) {
	Runs runs;
	ForEachChangedRun(current, wanted, merge_gap,
	                  [&runs](size_t offset, size_t size) { runs.emplace_back(offset, size); });
	return runs;
}

void TestChangedRuns() {
	const std::vector<uint8_t> current(200, 7);
	Check(ChangedRuns(current, current, 64).empty(), "equal bytes give no run");

	auto wanted = current;
	wanted[0]   = 1;
	wanted[199] = 1;
	Check(ChangedRuns(current, wanted, 64) == Runs {{0, 1}, {199, 1}},
	      "the first and the last byte are found, far apart: two runs");

	wanted     = current;
	wanted[10] = 1;
	wanted[13] = 1;
	wanted[90] = 1;
	Check(ChangedRuns(current, wanted, 64) == Runs {{10, 4}, {90, 1}},
	      "differences closer than the gap are one run, the far one is its own run");
	Check(ChangedRuns(current, wanted, 1) == Runs {{10, 1}, {13, 1}, {90, 1}},
	      "with a gap of one, every difference is its own run");
	Check(ChangedRuns(current, wanted, 80) == Runs {{10, 81}},
	      "a gap larger than the distance joins all of them");

	wanted = current;
	for (size_t i = 17; i < 60; i++) {
		wanted[i] = 9;
	}
	Check(ChangedRuns(current, wanted, 64) == Runs {{17, 43}},
	      "a long difference that crosses word boundaries is one exact run");

	// Applying the runs to `current` must give `wanted`.
	wanted = current;
	for (size_t i = 0; i < wanted.size(); i += 3 + i % 11) {
		wanted[i] ^= 0x5a;
	}
	auto patched = current;
	ForEachChangedRun(current, wanted, 8, [&](size_t offset, size_t size) {
		std::copy_n(wanted.begin() + static_cast<long>(offset), size,
		            patched.begin() + static_cast<long>(offset));
	});
	Check(patched == wanted, "writing the runs turns the current bytes into the wanted ones");
}

void TestWaitThatPassedAtOnce() {
	WaitSchedule                          schedule;
	const std::vector<WaitSchedule::Wait> waits = {{0x1000, false}};
	schedule.Arm(waits);
	Check(schedule.BeforeTest(0x1000) == 0u, "its memory is put in place before the test");
	Check(schedule.Blocked(0x1000) == 0u, "if it still fails the test, it is served at once");
	Check(schedule.Passed(0x1000) == 0u, "it is the recorded wait");
	Check(!schedule.BeforeTest(0x1000) && !schedule.Blocked(0x1000) && !schedule.Passed(0x1000),
	      "no wait is left on the address");
}

void TestWaitThatBlockedIsHeldUntilReleased() {
	WaitSchedule                          schedule;
	const std::vector<WaitSchedule::Wait> waits = {{0x1000, true}};
	schedule.Arm(waits);
	Check(!schedule.BeforeTest(0x1000), "memory is not put in place early for a wait that blocked");
	Check(!schedule.Blocked(0x1000), "it stays blocked until the replay reaches its release");
	schedule.Release(0);
	Check(schedule.Blocked(0x1000) == 0u, "after the release it is served");
}

void TestWaitsOnOneAddressKeepTheirOrder() {
	WaitSchedule                          schedule;
	const std::vector<WaitSchedule::Wait> waits = {
	    {0x1000, true}, {0x2000, false}, {0x1000, false}};
	schedule.Arm(waits);
	// The release of the first wait must not serve the second one on the same address.
	Check(!schedule.Blocked(0x1000), "first wait on the address: blocked");
	Check(schedule.BeforeTest(0x2000) == 1u, "another address is not affected");
	schedule.Release(0);
	Check(schedule.Passed(0x1000) == 0u, "the first wait passes first");
	Check(schedule.BeforeTest(0x1000) == 2u, "then the second wait on the address is next");
	Check(schedule.Passed(0x1000) == 2u, "and passes as itself");
	Check(!schedule.Blocked(0x3000), "a wait the capture does not have stays blocked");

	schedule.Arm(waits);
	Check(!schedule.Blocked(0x1000) && schedule.Passed(0x1000) == 0u,
	      "arming again starts the next pass from the first wait, not released");
}

void TestBlockedWaitIsOnlyReleasedWhenNothingElseCanRun() {
	StalledWaits stalled;
	Check(!stalled.NothingElseRan(7), "the first failed test of a wait proves nothing");
	stalled.Progress(); // another queue ran a draw
	Check(!stalled.NothingElseRan(7), "work was done since the last test: another queue runs");
	Check(stalled.NothingElseRan(7), "no work since the last test: nothing else can run");

	// Two waits that both wait for the player: the GPU thread tests them in turn.
	stalled.Reset();
	Check(!stalled.NothingElseRan(1) && !stalled.NothingElseRan(2), "first round: no proof yet");
	Check(stalled.NothingElseRan(1), "second round, no work in between: the first is released");
	stalled.Progress(); // it passed
	Check(!stalled.NothingElseRan(2), "that was work, so the second waits one more round");
	Check(stalled.NothingElseRan(2), "then it is released, too");
}

void TestPictureDifference() {
	const std::vector<uint8_t> live       = {10, 20, 30, 40, 50};
	auto                       difference = ComparePictures(live, live);
	Check(difference.bytes == 0 && difference.largest == 0, "equal pictures have no difference");
	const std::vector<uint8_t> replayed = {10, 23, 30, 31, 50};
	difference                          = ComparePictures(live, replayed);
	Check(difference.bytes == 2, "the differing bytes are counted");
	Check(difference.largest == 9, "the largest difference is found, whichever side is larger");
	const std::vector<uint8_t> shorter = {10, 20, 30};
	Check(ComparePictures(live, shorter).bytes == 2, "bytes one picture lacks count as differing");
}

void TestMemoryRecordsBelongToTheNextScanPoint() {
	using T                             = RecordType;
	const std::vector<RecordType> types = {
	    T::Pages,       // 0 -> submission 1
	    T::Submission,  // 1
	    T::GuestWrites, // 2 -> wait 4
	    T::Pages,       // 3 -> wait 4
	    T::WaitBytes,   // 4
	    T::Pages,       // 5 -> frame end 8 (a picture and shaders in between do not own memory)
	    T::Shaders,     // 6
	    T::Picture,     // 7
	    T::FrameEnd,    // 8
	    T::Pages,       // 9 -> read point 11
	    T::Mapping,     // 10 -> read point 11
	    T::ReadPoint,   // 11
	    T::Pages,       // 12 -> nothing follows
	};
	const auto owners = MemoryRecordOwners(types);
	Check(owners.size() == types.size(), "one entry per record");
	Check(owners[0] == 1, "pages before a submission belong to it");
	Check(owners[2] == 4 && owners[3] == 4, "writes and pages before a wait belong to the wait");
	Check(owners[5] == 8, "pages after the last wait belong to the frame end");
	Check(owners[9] == 11, "pages the renderer read after a change belong to that read");
	Check(owners[10] == 11, "a change of the mapped ranges belongs to the next point, too");
	Check(owners[12] == NoOwner, "pages with no point after them have no owner");
	Check(owners[1] == NoOwner && owners[4] == NoOwner && owners[6] == NoOwner &&
	          owners[8] == NoOwner,
	      "other records own nothing");
}

} // namespace

int main() {
	TestChangedRuns();
	TestWaitThatPassedAtOnce();
	TestWaitThatBlockedIsHeldUntilReleased();
	TestWaitsOnOneAddressKeepTheirOrder();
	TestBlockedWaitIsOnlyReleasedWhenNothingElseCanRun();
	TestMemoryRecordsBelongToTheNextScanPoint();
	TestPictureDifference();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("gpu replay logic tests passed\n");
	return 0;
}
