// Tests for the patch plan that makes a game trust the stand-in for EA's game server.

#include "libs/fakeEa/fakeEaPatchPlan.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace Libs::FakeEa;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

// The plan names the game it is made for: whatever game is loaded, not one fixed title.
void TestThePlanTakesItsGameFromItsArguments() {
	const std::vector<uint8_t> new_modulus(AuthorityModulus().size(), 0xab);
	for (const auto* title: {"PPSA34015", "CUSA12345"}) {
		const auto text = MakePatchPlan(title, "01.002.003", "eboot.bin", new_modulus);
		const auto plan = nlohmann::json::parse(text, nullptr, false);
		Check(plan.is_object(), "the plan is JSON");
		Check(plan.value("id", "") == title, "the title id comes from the argument");
		Check(plan.value("version", "") == "01.002.003", "the version comes from the argument");
		Check(plan.value("process", "") == "eboot.bin", "the process comes from the argument");
	}
}

// One write: EA's authority key, wherever the game keeps it, becomes ours.
void TestTheWriteReplacesTheAuthorityKeyWhereverItIs() {
	const std::vector<uint8_t> new_modulus(AuthorityModulus().size(), 0xab);
	const auto plan = nlohmann::json::parse(MakePatchPlan("T", "1", "eboot.bin", new_modulus), nullptr, false);
	Check(plan.is_object() && plan["mods"].size() == 1 && plan["mods"][0]["memory"].size() == 1,
	      "one mod with one write");
	const auto& write = plan["mods"][0]["memory"][0];
	Check(write["off"].get<std::string>().substr(0, 4) == "cc6d", "the write looks for EA's authority key");
	Check(write["off"].get<std::string>().size() == AuthorityModulus().size() * 2, "the whole key is the pattern");
	std::string expected;
	for (size_t i = 0; i < AuthorityModulus().size(); i++) expected += "ab";
	Check(write["on"].get<std::string>() == expected, "our key is written");
	Check(write["offset"].get<std::string>() == "0x0", "no address: the key is found by search");
}

// A key of another size would shift the bytes after it.
void TestAKeyOfAnotherSizeMakesNoPlan() {
	const std::vector<uint8_t> short_modulus(128, 0x01);
	Check(MakePatchPlan("T", "1", "eboot.bin", short_modulus).empty(), "a key of another size is refused");
}

void TestTheAuthorityKeyIsAnRsa2048Modulus() {
	const auto& modulus = AuthorityModulus();
	Check(modulus.size() == 256, "256 bytes");
	Check(modulus[0] == 0xcc && modulus[1] == 0x6d, "starts as EA's GOS 2015 key does");
}

// A game that holds EA's key is not always one that needs the server: FC 25 reaches its menus and a
// match without it, and with it stays on its main menu. Only a title with a built-in reply starts it
// by itself; any other title asks for it with KYTY_FAKE_EA=1.
void TestOnlyAKnownTitleStartsByItself() {
	Check(ShouldStartServer("PPSA34015", nullptr), "FC 27 starts by itself");
	Check(!ShouldStartServer("PPSA20049", nullptr), "another title does not start by itself");
	Check(ShouldStartServer("PPSA20049", "1"), "another title starts when asked");
	Check(!ShouldStartServer("PPSA34015", "0"), "KYTY_FAKE_EA=0 turns it off for every title");
	Check(!ShouldStartServer("PPSA20049", "0"), "KYTY_FAKE_EA=0 stays off");
	Check(!ShouldStartServer("", nullptr), "no title, no server");
}

} // namespace

int main() {
	TestOnlyAKnownTitleStartsByItself();
	TestThePlanTakesItsGameFromItsArguments();
	TestTheWriteReplacesTheAuthorityKeyWhereverItIs();
	TestAKeyOfAnotherSizeMakesNoPlan();
	TestTheAuthorityKeyIsAnRsa2048Modulus();
	if (g_failures != 0) {
		std::cerr << g_failures << " failure(s)\n";
		return 1;
	}
	std::cout << "fake EA patch plan tests passed\n";
	return 0;
}
