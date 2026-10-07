// Tests for closing the game from a gamepad: a hold of Back and Start, or of the Guide button.

#include "graphics/presentation/window/gamepadQuit.h"

#include <iostream>

namespace {

using Libs::Graphics::GamepadQuitHold;
using Libs::Graphics::QUIT_HOLD_MS;
using Libs::Graphics::QuitButton;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

void TestBackAndStartHeldCloseTheGame() {
	GamepadQuitHold hold;
	Check(!hold.Remaining(0).has_value(), "nothing held: no close runs");
	hold.Press(4, QuitButton::Back, true, 100);
	Check(!hold.Remaining(100).has_value(), "Back alone does not close the game");
	hold.Press(4, QuitButton::Start, true, 300);
	Check(hold.Remaining(300) == QUIT_HOLD_MS, "Back and Start together start the hold");
	Check(hold.Remaining(300 + QUIT_HOLD_MS - 1) == 1, "the hold counts down");
	Check(hold.Remaining(300 + QUIT_HOLD_MS) == 0, "held long enough: close now");
}

void TestLettingGoStopsTheHold() {
	GamepadQuitHold hold;
	hold.Press(4, QuitButton::Back, true, 0);
	hold.Press(4, QuitButton::Start, true, 0);
	hold.Press(4, QuitButton::Start, false, 1500);
	Check(!hold.Remaining(5000).has_value(), "a released Start stops the hold");
	hold.Press(4, QuitButton::Start, true, 6000);
	Check(hold.Remaining(6000) == QUIT_HOLD_MS, "pressing again starts the hold over");
}

void TestTheGuideButtonAloneClosesTheGame() {
	GamepadQuitHold hold;
	hold.Press(4, QuitButton::Guide, true, 0);
	Check(hold.Remaining(QUIT_HOLD_MS) == 0, "the Guide (PS) button held closes the game");
}

void TestButtonsOfTwoPadsDoNotCombine() {
	GamepadQuitHold hold;
	hold.Press(4, QuitButton::Back, true, 0);
	hold.Press(5, QuitButton::Start, true, 0);
	Check(!hold.Remaining(QUIT_HOLD_MS).has_value(),
	      "Back on one pad and Start on another do not close the game");
}

void TestARemovedPadIsForgotten() {
	GamepadQuitHold hold;
	hold.Press(4, QuitButton::Back, true, 0);
	hold.Press(4, QuitButton::Start, true, 0);
	hold.Forget(4);
	Check(!hold.Remaining(QUIT_HOLD_MS).has_value(), "a pad that left holds nothing");
}

} // namespace

int main() {
	TestBackAndStartHeldCloseTheGame();
	TestLettingGoStopsTheHold();
	TestTheGuideButtonAloneClosesTheGame();
	TestButtonsOfTwoPadsDoNotCombine();
	TestARemovedPadIsForgotten();
	if (g_failures != 0) {
		std::cerr << g_failures << " check(s) failed\n";
		return 1;
	}
	return 0;
}
