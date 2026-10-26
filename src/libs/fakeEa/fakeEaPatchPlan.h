#ifndef EMULATOR_SRC_LIBS_FAKE_EA_PATCH_PLAN_H_
#define EMULATOR_SRC_LIBS_FAKE_EA_PATCH_PLAN_H_

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::FakeEa {

// The modulus (RSA 2048) of "GOS 2015 Certificate Authority", the authority that EA's Blaze client
// trusts. A game that talks to EA's game servers has it in its data.
const std::vector<uint8_t>& AuthorityModulus();

// A patch plan (GoldHEN-style JSON, see loader/gamePatch.h) for the game that is loaded: it
// replaces EA's authority key, wherever the game keeps it, by `new_modulus`. The title, the version
// and the process are the game's own, so the plan fits any game and any version. Empty when
// `new_modulus` is not the size of EA's key.
std::string MakePatchPlan(const std::string& title_id, const std::string& version, const std::string& process,
                          const std::vector<uint8_t>& new_modulus);

// Whether the server starts for a game that holds EA's key. `switch_value` is KYTY_FAKE_EA (null:
// not set). A title with a built-in Util.preAuth reply (FC 27) starts it by itself. Another EA
// game may not need it (FC 25 reaches its menus and a match without it, and with it stays on its
// main menu) and has no built-in reply: it starts with KYTY_FAKE_EA=1. "0" turns it off.
bool ShouldStartServer(const std::string& title_id, const char* switch_value);

// Whether Util.preAuth has a reply built in for this title.
bool HasBuiltInPreAuth(const std::string& title_id);

std::string Hex(const std::vector<uint8_t>& bytes);

} // namespace Libs::FakeEa

#endif // EMULATOR_SRC_LIBS_FAKE_EA_PATCH_PLAN_H_
