#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_GAMEPADQUIT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_GAMEPADQUIT_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

// Closing the game from a gamepad, for a player away from the keyboard: hold Back and Start
// together, or the Guide (PS) button, on one gamepad. The hold is long, so a press of Start that
// the game reads as Options never closes it. Some gamepads keep their logo button to themselves
// (a Logitech Cordless RumblePad 2 sends nothing for it), so Back and Start work on every gamepad.

namespace Libs::Graphics {

inline constexpr uint64_t QUIT_HOLD_MS = 2000;

enum class QuitButton : uint8_t { Back, Start, Guide };

class GamepadQuitHold {
public:
	// A button of gamepad `pad` went down or up at `now_ms`.
	void Press(int pad, QuitButton button, bool down, uint64_t now_ms) {
		auto& held = Of(pad);
		switch (button) {
			case QuitButton::Back: held.back = down; break;
			case QuitButton::Start: held.start = down; break;
			case QuitButton::Guide: held.guide = down; break;
		}
		const bool holding = (held.back && held.start) || held.guide;
		if (!holding) {
			held.since.reset();
		} else if (!held.since) {
			held.since = now_ms;
		}
	}

	// The gamepad left: what it held is forgotten.
	void Forget(int pad) {
		std::erase_if(m_pads, [pad](const Held& held) { return held.pad == pad; });
	}

	// Milliseconds left until a hold closes the game (0: close now), or none when no hold runs.
	[[nodiscard]] std::optional<uint64_t> Remaining(uint64_t now_ms) const {
		std::optional<uint64_t> left;
		for (const auto& held: m_pads) {
			if (held.since) {
				const auto elapsed = now_ms > *held.since ? now_ms - *held.since : 0;
				const auto pad_left = elapsed < QUIT_HOLD_MS ? QUIT_HOLD_MS - elapsed : 0;
				left = left ? std::min(*left, pad_left) : pad_left;
			}
		}
		return left;
	}

private:
	struct Held {
		int                     pad   = 0;
		bool                    back  = false;
		bool                    start = false;
		bool                    guide = false;
		std::optional<uint64_t> since;
	};

	Held& Of(int pad) {
		const auto it =
		    std::ranges::find_if(m_pads, [pad](const Held& held) { return held.pad == pad; });
		if (it != m_pads.end()) {
			return *it;
		}
		return m_pads.emplace_back(Held {.pad = pad});
	}

	std::vector<Held> m_pads;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_GAMEPADQUIT_H_ */
