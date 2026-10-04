#ifndef EMULATOR_SRC_LIBS_PLAYERSLOTS_H_
#define EMULATOR_SRC_LIBS_PLAYERSLOTS_H_

#include <array>
#include <cstdint>
#include <deque>

namespace Libs {

// Who plays: up to four players, each with one gamepad and one user.
// A gamepad takes the lowest free slot when it connects and keeps it until it disconnects.
// Player 1 (slot 0) is logged in from the start and stays logged in, also with no gamepad: the
// keyboard plays in that slot. The other players are logged in while their gamepad is connected.
class PlayerSlots final {
public:
	static constexpr int MaxPlayers = 4;
	static constexpr int NoSlot     = -1;
	static constexpr int NoGamepad  = 0; // never a host gamepad id

	struct Event {
		bool login = false;
		int  slot  = NoSlot;
	};

	// `first_user` is the user id of player 1. The other players get the ids that follow it,
	// without the two ids the system keeps for itself.
	explicit PlayerSlots(int32_t first_user) {
		int32_t user = first_user;
		for (auto& id: m_users) {
			while (user == UserEveryone || user == UserSystem) {
				++user;
			}
			id = user++;
		}
		m_events.push_back({.login = true, .slot = 0});
	}

	// The slot of the gamepad, or NoSlot when all four are taken.
	int Connect(int gamepad) {
		if (const int slot = SlotOf(gamepad); slot != NoSlot) {
			return slot;
		}
		const int slot = SlotOf(NoGamepad);
		if (slot != NoSlot) {
			m_gamepads[slot] = gamepad;
			if (slot != 0) {
				m_events.push_back({.login = true, .slot = slot});
			}
		}
		return slot;
	}

	// The slot the gamepad had, or NoSlot when it had none.
	int Disconnect(int gamepad) {
		const int slot = gamepad != NoGamepad ? SlotOf(gamepad) : NoSlot;
		if (slot != NoSlot) {
			m_gamepads[slot] = NoGamepad;
			if (slot != 0) {
				m_events.push_back({.login = false, .slot = slot});
			}
		}
		return slot;
	}

	[[nodiscard]] int SlotOf(int gamepad) const {
		for (int slot = 0; slot < MaxPlayers; ++slot) {
			if (m_gamepads[slot] == gamepad) {
				return slot;
			}
		}
		return NoSlot;
	}

	[[nodiscard]] int GamepadOf(int slot) const { return m_gamepads[slot]; }

	[[nodiscard]] bool IsLoggedIn(int slot) const {
		return slot == 0 || (slot > 0 && slot < MaxPlayers && m_gamepads[slot] != NoGamepad);
	}

	[[nodiscard]] int32_t UserOf(int slot) const { return m_users[slot]; }

	// The slot of a logged-in user, or NoSlot.
	[[nodiscard]] int SlotOfUser(int32_t user) const {
		for (int slot = 0; slot < MaxPlayers; ++slot) {
			if (m_users[slot] == user && IsLoggedIn(slot)) {
				return slot;
			}
		}
		return NoSlot;
	}

	// The oldest login or logout the game has not seen yet. The first one is the login of player 1.
	bool TakeEvent(Event* event) {
		if (m_events.empty()) {
			return false;
		}
		*event = m_events.front();
		m_events.pop_front();
		return true;
	}

private:
	static constexpr int32_t UserEveryone = 0xfe;
	static constexpr int32_t UserSystem   = 0xff;

	std::array<int, MaxPlayers>     m_gamepads {};
	std::array<int32_t, MaxPlayers> m_users {};
	std::deque<Event>               m_events;
};

} // namespace Libs

#endif /* EMULATOR_SRC_LIBS_PLAYERSLOTS_H_ */
