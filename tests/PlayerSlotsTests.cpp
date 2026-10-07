#include "libs/playerSlots.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::PlayerSlots;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PlayerSlotsTests: failed: %s\n", message);
		std::abort();
	}
}

bool NextEventIs(PlayerSlots& slots, bool login, int slot) {
	PlayerSlots::Event event;
	return slots.TakeEvent(&event) && event.login == login && event.slot == slot;
}

bool NoEvent(PlayerSlots& slots) {
	PlayerSlots::Event event;
	return !slots.TakeEvent(&event);
}

void TestPlayerOneIsThereFromTheStart() {
	PlayerSlots slots(1000);
	Check(slots.IsLoggedIn(0) && slots.SlotOfUser(1000) == 0, "player 1 is not logged in at start");
	Check(!slots.IsLoggedIn(1) && slots.SlotOfUser(1001) == PlayerSlots::NoSlot,
	      "player 2 is logged in with no gamepad");
	Check(NextEventIs(slots, true, 0) && NoEvent(slots),
	      "the game does not get exactly one login, of player 1, at start");
}

void TestGamepadsTakeTheLowestFreeSlot() {
	PlayerSlots slots(1000);
	Check(slots.Connect(7) == 0 && slots.Connect(9) == 1 && slots.Connect(3) == 2 &&
	          slots.Connect(5) == 3,
	      "gamepads did not get slots in the order they connected");
	Check(slots.Connect(9) == 1, "a gamepad that connects twice changed its slot");
	Check(slots.Connect(11) == PlayerSlots::NoSlot && slots.SlotOf(11) == PlayerSlots::NoSlot,
	      "a fifth gamepad got a slot");
	Check(slots.Disconnect(11) == PlayerSlots::NoSlot, "a gamepad with no slot left a slot");
	Check(slots.Disconnect(9) == 1 && slots.SlotOf(3) == 2 && slots.SlotOf(5) == 3,
	      "a gamepad that left moved the other gamepads");
	Check(slots.Connect(11) == 1 && slots.GamepadOf(1) == 11, "a free slot was not used again");
}

void TestPlayersLogInAndOutWithTheirGamepad() {
	PlayerSlots slots(1000);
	Check(NextEventIs(slots, true, 0), "no login of player 1");
	slots.Connect(7);
	Check(NoEvent(slots), "the gamepad of player 1 logged player 1 in again");
	slots.Connect(9);
	Check(slots.IsLoggedIn(1) && slots.SlotOfUser(slots.UserOf(1)) == 1 &&
	          NextEventIs(slots, true, 1) && NoEvent(slots),
	      "the second gamepad did not log player 2 in");
	slots.Connect(9);
	Check(NoEvent(slots), "a gamepad that connects twice logged its player in twice");
	slots.Disconnect(9);
	Check(!slots.IsLoggedIn(1) && slots.SlotOfUser(slots.UserOf(1)) == PlayerSlots::NoSlot &&
	          NextEventIs(slots, false, 1) && NoEvent(slots),
	      "player 2 stayed logged in with no gamepad");
	slots.Disconnect(7);
	Check(slots.IsLoggedIn(0) && slots.SlotOfUser(1000) == 0 && NoEvent(slots),
	      "player 1 was logged out with their gamepad");
	Check(slots.Disconnect(PlayerSlots::NoGamepad) == PlayerSlots::NoSlot && NoEvent(slots),
	      "no gamepad was taken for a gamepad");
}

void TestEveryPlayerHasTheirOwnUser() {
	PlayerSlots slots(0xfd);
	Check(slots.UserOf(0) == 0xfd && slots.UserOf(1) == 0x100 && slots.UserOf(2) == 0x101 &&
	          slots.UserOf(3) == 0x102,
	      "a player got a user id the system keeps for itself, or two players share one");
}

} // namespace

int main() {
	TestPlayerOneIsThereFromTheStart();
	TestGamepadsTakeTheLowestFreeSlot();
	TestPlayersLogInAndOutWithTheirGamepad();
	TestEveryPlayerHasTheirOwnUser();
	std::puts("PlayerSlotsTests: all cases passed");
	return 0;
}
