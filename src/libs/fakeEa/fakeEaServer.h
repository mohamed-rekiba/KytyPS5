#ifndef EMULATOR_INCLUDE_EMULATOR_FAKE_EA_SERVER_H_
#define EMULATOR_INCLUDE_EMULATOR_FAKE_EA_SERVER_H_

#include <cstdint>
#include <string>

// A stand-in for EA's Blaze game server, inside the emulator. A game of EA asks EA's servers for a
// game server, connects to it with its own TLS client, and may not leave its first screen until
// that works (FC 27: "Let's Play!"). The emulator answers the question with this machine, runs a
// TLS 1.2 server on a port of its own, and replies to the few calls the game makes before its
// offline menus.
//
// The game trusts a server by a certificate authority in its own data. The emulator makes a
// certificate authority with the name of EA's "GOS 2015" authority, and a patch (a GoldHEN-style
// plan, see loader/gamePatch.h) puts its key in the game's list. The game says whether it needs
// this: it holds EA's authority key (AuthorityModulus), and the patch finds it by search. The
// title, version and process of the patch are the loaded game's own.
//
// Only the reply to Util.preAuth is a recorded one, and only FC 27's is built in. Another title
// needs its reply in KYTY_FAKE_EA_DIR, recorded with KYTY_FAKE_EA_UPSTREAM=all.
//
// Switches (environment variables):
//   KYTY_FAKE_EA=0            turns the server off. =1 starts it for an EA game that has no built-in
//                             Util.preAuth reply (FC 27 starts it by itself).
//   KYTY_FAKE_EA_TRACE=1      writes every request and reply, decoded, to the log.
//   KYTY_FAKE_EA_UPSTREAM=1   sends a call the server does not answer to EA's real server, and
//                             gives the game the real reply. "all" does it for Util.preAuth too.
//   KYTY_FAKE_EA_DIR=folder   replies in files (0009_0007_type6.bin: component, command, client
//                             type) that are used before the built-in ones; a reply from the real
//                             server is written there, with its secret values blanked.
namespace Libs::FakeEa {

// Starts the server for the loaded game. Call it for a game that holds EA's authority key, before
// the guest runs. False: it was started already, ShouldStartServer says no for this title and
// KYTY_FAKE_EA, or it did not start.
bool StartForGame(const std::string& title_id);

// The port of the server on this machine, or 0. On Linux and macOS nothing listens on it: the
// socket layer calls OpenConnection when the game connects to it. On Windows the server listens.
uint16_t Port();

// A connected socket for the game, with the server on its other end (no port, no accept thread), or
// -1 (no server, or Windows). The caller owns the socket.
int OpenConnection();

// The patch plan (JSON) that makes the loaded game (its version and main module's file name) trust
// this server, or an empty string.
std::string PatchPlanJson(const std::string& version, const std::string& process);

// The local user looks signed in to PSN, because the server runs (see KYTY_PSN_SIGNED_IN).
bool SignedInForTitle();

} // namespace Libs::FakeEa

#endif // EMULATOR_INCLUDE_EMULATOR_FAKE_EA_SERVER_H_
