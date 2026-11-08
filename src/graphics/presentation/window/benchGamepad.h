#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHGAMEPAD_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHGAMEPAD_H_

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// The virtual gamepad that the pad and axis steps of a bench script drive. SDL sends its events as
// for a pad on the host, so a replay goes through the same input path as the player's pad did.
class BenchGamepad {
public:
	BenchGamepad() = default;
	~BenchGamepad() { Detach(); }

	BenchGamepad(const BenchGamepad&)            = delete;
	BenchGamepad& operator=(const BenchGamepad&) = delete;

	// False when SDL cannot attach a virtual gamepad; SDL_GetError() names the cause.
	[[nodiscard]] bool Attach() {
		if (m_joystick != nullptr) {
			return true;
		}
		// SDL drops gamepad input while no window of the app has keyboard focus, and a replay
		// runs with the game window in the background, as key steps do.
		SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
		SDL_VirtualJoystickDesc desc;
		SDL_INIT_INTERFACE(&desc);
		desc.type     = SDL_JOYSTICK_TYPE_GAMEPAD;
		desc.naxes    = SDL_GAMEPAD_AXIS_COUNT;
		desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
		// With every button and axis present, SDL numbers them in the order of the gamepad
		// enums, so a joystick index is the gamepad button or axis.
		desc.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1u;
		desc.axis_mask   = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1u;
		desc.name        = "Bench script gamepad";
		m_id             = SDL_AttachVirtualJoystick(&desc);
		if (m_id == 0) {
			return false;
		}
		m_joystick = SDL_OpenJoystick(m_id);
		if (m_joystick == nullptr) {
			SDL_DetachVirtualJoystick(m_id);
			m_id = 0;
			return false;
		}
		return true;
	}

	void Detach() {
		if (m_joystick != nullptr) {
			SDL_CloseJoystick(m_joystick);
			SDL_DetachVirtualJoystick(m_id);
			m_joystick = nullptr;
			m_id       = 0;
		}
	}

	[[nodiscard]] SDL_JoystickID Id() const { return m_id; }

	void SetButton(SDL_GamepadButton button, bool down) {
		SDL_SetJoystickVirtualButton(m_joystick, button, down);
	}

	// `value` is the gamepad value: -32768 to 32767, a trigger 0 to 32767.
	void SetAxis(SDL_GamepadAxis axis, int value) {
		SDL_SetJoystickVirtualAxis(m_joystick, axis, JoystickAxisValue(axis, value));
	}

	// SDL maps the full joystick range of a virtual trigger to the gamepad's 0 to 32767, rounding
	// down; rounding up here makes the gamepad report the value that was set.
	[[nodiscard]] static int16_t JoystickAxisValue(SDL_GamepadAxis axis, int value) {
		if (axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
			constexpr int64_t full = int64_t {SDL_JOYSTICK_AXIS_MAX} - SDL_JOYSTICK_AXIS_MIN;
			const int64_t     at   = std::clamp(value, 0, int {SDL_JOYSTICK_AXIS_MAX});
			value =
			    static_cast<int>((at * full + SDL_JOYSTICK_AXIS_MAX - 1) / SDL_JOYSTICK_AXIS_MAX +
			                     SDL_JOYSTICK_AXIS_MIN);
		}
		return static_cast<int16_t>(
		    std::clamp(value, int {SDL_JOYSTICK_AXIS_MIN}, int {SDL_JOYSTICK_AXIS_MAX}));
	}

private:
	SDL_JoystickID m_id       = 0;
	SDL_Joystick*  m_joystick = nullptr;
};

// While a bench script runs, the presses, moves, touches and motion of the pads on the host are
// dropped: a touch took player 1 from a replay, or closed the run with the quit hold. The script's
// virtual pad (`bench_pad`, 0 when it has none) is not dropped. The caller lets pads arrive and
// leave.
[[nodiscard]] inline bool BenchDropsPadInput(bool bench_running, SDL_JoystickID bench_pad,
                                             SDL_JoystickID from) {
	return bench_running && from != bench_pad;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHGAMEPAD_H_ */
