#include "libs/controller.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/dualSenseHaptics.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"
#include "libs/playerSlots.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Controller {

LIB_NAME("Pad", "Pad");

constexpr int PAD_ERROR_INVALID_ARG    = -2137915391; /* 0x80920001 */
constexpr int PAD_ERROR_INVALID_HANDLE = -2137915389; /* 0x80920003 */

constexpr uint32_t RUMBLE_DURATION_MS = 0xffff;
constexpr uint32_t RELEASE_FLUSH_MS   = 50;

struct PadControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int      device_class;
	uint8_t  reserve[8];
};

struct PadLightBarParam {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t reserve;
};

struct PadVibrationParam {
	uint8_t large_motor;
	uint8_t small_motor;
};

struct PadTriggerEffectCommand {
	uint32_t mode;
	uint8_t  reserve[4];
	uint8_t  data[48];
};

struct PadTriggerEffectParam {
	uint8_t                 trigger_mask;
	uint8_t                 reserve[7];
	PadTriggerEffectCommand command[2];
};

static_assert(sizeof(PadTriggerEffectCommand) == 56);
static_assert(sizeof(PadTriggerEffectParam) == 120);

struct DualSenseEffects {
	uint8_t enable_bits;
	uint8_t reserve[9];
	uint8_t right_trigger[11];
	uint8_t left_trigger[11];
};

static_assert(sizeof(DualSenseEffects) == 32);

struct ControllerState {
	struct Touch {
		uint8_t  id   = 0;
		bool     down = false;
		uint16_t x    = 0;
		uint16_t y    = 0;
	};

	uint64_t time                                  = 0;
	uint32_t buttons                               = 0;
	int      axes[static_cast<int>(Axis::AxisMax)] = {128, 128, 128, 128, 0, 0};
	Touch    touch[2];
	std::array<float, 3> accel {0.0f, 1.0f, 0.0f};
	std::array<float, 3> gyro {};
	std::array<float, 4> orientation {0.0f, 0.0f, 0.0f, 1.0f};
};

// The pad of one player, as the game reads it.
struct Pad {
	static constexpr uint32_t STATES_MAX = 64;

	void AddState();
	// A gamepad joined or left this player: nothing of the old input is kept.
	void ForgetInput();
	// The gamepad left: what the game asked it to do is not sent to the next one.
	void ForgetOutput() {
		vibration       = {};
		vibration_until = 0;
		trigger_effect  = {};
	}

	int      connected_count = 0;
	bool     motion_enabled  = true;
	uint64_t gyro_time       = 0;
	// Accelerometer gravity direction in the reset frame.
	std::array<float, 3> up_reference {0.0f, 1.0f, 0.0f};
	bool                 up_reference_valid = false;
	ControllerState      state;
	ControllerState      states[STATES_MAX];
	bool                 obtained[STATES_MAX] {};
	uint32_t             states_num    = 0;
	uint32_t             first_state   = 0;
	uint8_t              next_touch_id = 1;
	// The game's last requests, re-sent when their intensity setting changes.
	std::array<uint8_t, 2> vibration {};
	uint64_t               vibration_until = 0;
	PadTriggerEffectParam  trigger_effect {};
};

class GameController {
public:
	GameController(): m_slots(Config::GetUserId()) { m_pads[0].connected_count = 1; }

	KYTY_CLASS_NO_COPY(GameController);

	void Connect(int id);
	void Disconnect(int id);
	void Button(int id, uint32_t button, bool down);
	void Axis(int id, Axis axis, int value);
	void RightStick(int id, int x, int y);
	void TouchPad(int id, int finger, bool down, float x, float y);
	void Motion(int id, Sensor sensor, const float* data, uint64_t time_us);
	void SetMotionSensorState(int slot, bool enable);
	void ResetOrientation(int slot);
	void ResetInputState();
	void ReleaseHostPads();
	void GetConnectionInfo(int slot, bool* flag, int* count);
	void SetVibration(int slot, uint8_t large_motor, uint8_t small_motor);
	int  GetGamepadOfPlayerOne();
	void SetLightBar(int slot, uint8_t r, uint8_t g, uint8_t b);
	bool SetTriggerEffect(int slot, const PadTriggerEffectParam& param);
	void ReadState(int slot, ControllerState* state, bool* flag, int* count);
	int  ReadStates(int slot, ControllerState* states, int states_num, bool* flag, int* count);
	int  GetSlotOfUser(int user_id);
	int  GetUserOfSlot(int slot);
	void GetLoggedInUsers(int* users);
	bool TakeEvent(PlayerSlots::Event* event);
	void  CycleSetting(Setting setting);
	float GetSettingScale(Setting setting) const;

private:
	// Gives the gamepad a player, when one is free. Called at its first button press.
	void Seat(int id);
	// The pad that input from this host device goes to. The keyboard plays as player 1.
	Pad* InputPad(int id);
	// The host gamepad of the player, or nullptr.
	SDL_Gamepad* HostPad(int slot);
	bool         IsConnected(int slot) const { return m_slots.IsLoggedIn(slot); }
	// Apply cached output without changing its game-requested lifetime; caller holds m_mutex.
	void ApplyVibration(int slot);
	bool SendTriggerEffect(int slot, const PadTriggerEffectParam& param);

	Common::Mutex    m_mutex;
	std::vector<int> m_host_pads; // every open gamepad, with a player or not
	PlayerSlots      m_slots;
	std::array<Pad, PlayerSlots::MaxPlayers> m_pads;
	// Setting changes share the output lock; audio only needs an atomic scale snapshot.
	std::array<std::atomic<uint32_t>, 3> m_setting_steps {};
};

static GameController* g_controller = nullptr;

static void pad_fill_data(PadData* data, const ControllerState& state, bool connected,
                          int connected_count) {
	EXIT_IF(data == nullptr);

	std::memset(data, 0, sizeof(*data));

	data->buttons           = state.buttons;
	data->left_stick_x      = state.axes[static_cast<int>(Axis::LeftX)];
	data->left_stick_y      = state.axes[static_cast<int>(Axis::LeftY)];
	data->right_stick_x     = state.axes[static_cast<int>(Axis::RightX)];
	data->right_stick_y     = state.axes[static_cast<int>(Axis::RightY)];
	data->analog_buttons_l2 = state.axes[static_cast<int>(Axis::TriggerLeft)];
	data->analog_buttons_r2 = state.axes[static_cast<int>(Axis::TriggerRight)];
	data->acceleration_x     = state.accel[0];
	data->acceleration_y     = state.accel[1];
	data->acceleration_z     = state.accel[2];
	data->angular_velocity_x = state.gyro[0];
	data->angular_velocity_y = state.gyro[1];
	data->angular_velocity_z = state.gyro[2];
	data->orientation_x      = state.orientation[0];
	data->orientation_y      = state.orientation[1];
	data->orientation_z      = state.orientation[2];
	data->orientation_w      = state.orientation[3];
	for (const auto& touch: state.touch) {
		if (touch.down) {
			auto& output = data->touch_data.touch[data->touch_data.touch_num++];
			output.x     = touch.x;
			output.y     = touch.y;
			output.id    = touch.id;
		}
	}
	data->connected              = connected;
	data->timestamp              = state.time;
	data->connected_count        = static_cast<uint8_t>(std::min(connected_count, 255));
	data->device_unique_data_len = 0;
}

// In cycle order, starting at the PS5 default.
constexpr std::array SPEAKER_VOLUME = {1.0f, 0.0f, 0.02f, 0.12f, 0.42f};
constexpr std::array INTENSITY      = {1.0f, 0.0f, 0.33f, 0.66f};

void CycleSetting(Setting setting) {
	if (g_controller != nullptr) {
		g_controller->CycleSetting(setting);
	}
}

float GetSettingScale(Setting setting) {
	return g_controller != nullptr ? g_controller->GetSettingScale(setting) : 1.0f;
}

void GameController::CycleSetting(Setting setting) {
	Common::LockGuard lock(m_mutex);
	auto&             step = m_setting_steps[static_cast<size_t>(setting)];
	const auto count = setting == Setting::SpeakerVolume ? SPEAKER_VOLUME.size() : INTENSITY.size();
	step.store((step.load(std::memory_order_relaxed) + 1) % count, std::memory_order_relaxed);
	// The setting is one for every player.
	for (int slot = 0; slot < PlayerSlots::MaxPlayers; slot++) {
		if (setting == Setting::VibrationIntensity) {
			ApplyVibration(slot);
		} else if (setting == Setting::TriggerEffectIntensity) {
			(void)SendTriggerEffect(slot, m_pads[slot].trigger_effect);
		}
	}
}

float GameController::GetSettingScale(Setting setting) const {
	const auto step = m_setting_steps[static_cast<size_t>(setting)].load(std::memory_order_relaxed);
	if (setting == Setting::SpeakerVolume) {
		// The old speaker level matches the PS5 slider's midpoint. Give the upper
		// half of the global slider up to 6 dB of extra software gain.
		return SPEAKER_VOLUME[step] * (Config::GetControllerSpeakerVolume() / 50.0f);
	}
	if (setting == Setting::VibrationIntensity) {
		return INTENSITY[step] * (Config::GetControllerVibrationIntensity() / 100.0f);
	}
	return INTENSITY[step];
}

static uint8_t Scale(uint8_t value, float scale) {
	if (value == 0 || scale == 0.0f) {
		return 0;
	}
	return static_cast<uint8_t>(std::max(1L, std::lround(value * scale)));
}

static bool trigger_effect_zones(uint8_t* effect, const uint8_t* strengths, float scale,
                                 uint8_t type, uint8_t frequency = 0) {
	uint16_t active = 0;
	uint32_t packed = 0;
	for (int i = 0; i < 10; i++) {
		if (strengths[i] > 8) {
			return false;
		}
		const auto strength = Scale(strengths[i], scale);
		if (strength != 0) {
			active |= static_cast<uint16_t>(1u << i);
			packed |= static_cast<uint32_t>(strength - 1u) << (3 * i);
		}
	}

	effect[0] = active != 0 && (type != 0x26 || frequency != 0) ? type : 0x05;
	effect[1] = static_cast<uint8_t>(active);
	effect[2] = static_cast<uint8_t>(active >> 8u);
	effect[3] = static_cast<uint8_t>(packed);
	effect[4] = static_cast<uint8_t>(packed >> 8u);
	effect[5] = static_cast<uint8_t>(packed >> 16u);
	effect[6] = static_cast<uint8_t>(packed >> 24u);
	effect[9] = frequency;
	return true;
}

static bool trigger_effect_to_dualsense(const PadTriggerEffectCommand& command, uint8_t* effect,
                                        float scale) {
	std::memset(effect, 0, 11);
	effect[0] = 0x05;

	uint8_t strengths[10] = {};
	switch (command.mode) {
		case 0: return true;
		case 1:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, scale, 0x21);
		case 2: {
			const auto start    = command.data[0];
			const auto end      = command.data[1];
			const auto strength = command.data[2];
			if (start < 2 || start > 7 || end <= start || end > 8 || strength > 8) {
				return false;
			}
			const auto scaled = Scale(strength, scale);
			if (scaled == 0) {
				return true;
			}
			const uint16_t zones = static_cast<uint16_t>((1u << start) | (1u << end));
			effect[0]            = 0x25;
			effect[1]            = static_cast<uint8_t>(zones);
			effect[2]            = static_cast<uint8_t>(zones >> 8u);
			effect[3]            = scaled - 1u;
			return true;
		}
		case 3:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, scale, 0x26, command.data[2]);
		case 4: return trigger_effect_zones(effect, command.data, scale, 0x21);
		case 5: {
			const int start          = command.data[0];
			const int end            = command.data[1];
			const int start_strength = command.data[2];
			const int end_strength   = command.data[3];
			if (start > 8 || end <= start || end > 9 || start_strength < 1 || start_strength > 8 ||
			    end_strength < 1 || end_strength > 8) {
				return false;
			}
			for (int i = start; i < 10; i++) {
				const int delta  = (end_strength - start_strength) * (i - start);
				const int length = end - start;
				strengths[i]     = static_cast<uint8_t>(
				    i >= end ? end_strength
				             : start_strength +
				                   (delta + (delta < 0 ? -length / 2 : length / 2)) / length);
			}
			return trigger_effect_zones(effect, strengths, scale, 0x21);
		}
		case 6: return trigger_effect_zones(effect, command.data + 1, scale, 0x26, command.data[0]);
		default: return false;
	}
}

void Initialize() {
	EXIT_IF(g_controller != nullptr);

	g_controller = new GameController;
}

void Shutdown() {
	EmergencyShutdown();
	delete g_controller;
	g_controller = nullptr;
}

void EmergencyShutdown() {
	if (g_controller != nullptr) {
		g_controller->ReleaseHostPads();
	}
}

void Pad::AddState() {
	if (states_num >= STATES_MAX) {
		states_num  = STATES_MAX - 1;
		first_state = (first_state + 1) % STATES_MAX;
	}

	const auto index = (first_state + states_num) % STATES_MAX;
	states[index]    = state;
	obtained[index]  = false;
	states_num++;
}

void Pad::ForgetInput() {
	state              = {};
	gyro_time          = 0;
	up_reference_valid = false;
	states_num         = 0;
	first_state        = 0;
	next_touch_id      = 1;
}

Pad* GameController::InputPad(int id) {
	const int slot = id == HOST_INPUT_CONTROLLER_ID ? 0 : m_slots.SlotOf(id);
	return slot != PlayerSlots::NoSlot ? &m_pads[slot] : nullptr;
}

SDL_Gamepad* GameController::HostPad(int slot) {
	const int id = m_slots.GamepadOf(slot);
	return id != PlayerSlots::NoGamepad ? SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id))
	                                    : nullptr;
}

// The gamepad's name and SDL's id for it, for the log. SDL counts ids up from the first device
// it opened and never reuses one, so the id alone says nothing about the player.
static std::string GamepadLabel(int id) {
	const char* name = SDL_GetGamepadNameForID(static_cast<SDL_JoystickID>(id));
	return fmt::format("\"{}\" (id {})", name != nullptr ? name : "gamepad", id);
}

// Where SDL found the gamepad, for the log: its vendor, product, device path, and the driver that
// opened it ('m' for Apple's GameController framework). One device can show up as two gamepads:
// that framework reports a Logitech Cordless RumblePad 2 twice. Only the copy that sends input
// takes a player (Button).
static std::string GamepadSource(int id) {
	const auto  jid     = static_cast<SDL_JoystickID>(id);
	const char* path    = SDL_GetGamepadPathForID(jid);
	const auto  guid    = SDL_GetGamepadGUIDForID(jid);
	Uint16      vendor  = 0;
	Uint16      product = 0;
	SDL_GetJoystickGUIDInfo(guid, &vendor, &product, nullptr, nullptr);
	const char driver =
	    guid.data[14] >= ' ' && guid.data[14] < 127 ? static_cast<char>(guid.data[14]) : '-';
	return fmt::format("vendor {:04x} product {:04x}, driver '{}', path {}", vendor, product,
	                   driver, path != nullptr ? path : "none");
}

void GameController::Seat(int id) {
	const int slot = m_slots.Connect(id);
	if (slot == PlayerSlots::NoSlot) {
		LOGF("Gamepad %s has no player: all %d players have a gamepad\n", GamepadLabel(id).c_str(),
		     PlayerSlots::MaxPlayers);
		return;
	}
	auto& pad = m_pads[slot];
	pad.ForgetInput();
	if (slot != 0) {
		pad.connected_count++;
	}
	LOGF("Gamepad %s plays as player %d\n", GamepadLabel(id).c_str(), slot + 1);
}

void GameController::Connect(int id) {
	Common::LockGuard lock(m_mutex);

	if (std::find(m_host_pads.begin(), m_host_pads.end(), id) != m_host_pads.end()) {
		return;
	}

	m_host_pads.push_back(id);

	if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id)); pad != nullptr) {
		if (const auto& color = Config::GetControllerColor()) {
			(void)SDL_SetGamepadLED(pad, (*color)[0], (*color)[1], (*color)[2]);
		}
		for (auto sensor: {SDL_SENSOR_ACCEL, SDL_SENSOR_GYRO}) {
			if (SDL_GamepadHasSensor(pad, sensor) &&
			    !SDL_SetGamepadSensorEnabled(pad, sensor, true)) {
				LOGF("\t enabling controller sensor failed: %s\n", SDL_GetError());
			}
		}
	}

	// It plays from its first button press (Button), as a console's controller logs in with a
	// press of its PS button: a gamepad that is only plugged in takes no player and logs in no user.
	LOGF("Gamepad %s connected: press a button on it to play\n", GamepadLabel(id).c_str());
	LOGF("\t %s\n", GamepadSource(id).c_str());
}

void GameController::Disconnect(int id) {
	Common::LockGuard lock(m_mutex);

	const auto it = std::find(m_host_pads.begin(), m_host_pads.end(), id);
	EXIT_IF(it == m_host_pads.end());

	m_host_pads.erase(it);

	const int slot = m_slots.Disconnect(id);
	if (slot == PlayerSlots::NoSlot) {
		return;
	}
	m_pads[slot].ForgetInput();
	m_pads[slot].ForgetOutput();
	LOGF("Gamepad %s left: player %d has no gamepad\n", GamepadLabel(id).c_str(), slot + 1);
}

void GameController::Button(int id, uint32_t button, bool down) {
	Common::LockGuard lock(m_mutex);

	if (down && id != HOST_INPUT_CONTROLLER_ID && m_slots.SlotOf(id) == PlayerSlots::NoSlot &&
	    std::find(m_host_pads.begin(), m_host_pads.end(), id) != m_host_pads.end()) {
		Seat(id);
	}
	if (button == PAD_BUTTON_NONE) {
		return; // only seats the gamepad
	}

	if (auto* pad = InputPad(id); pad != nullptr) {
		pad->state.time = LibKernel::KernelGetProcessTime();

		pad->state.buttons = down ? pad->state.buttons | button : pad->state.buttons & ~button;

		pad->AddState();
	}
}

void GameController::Axis(int id, Controller::Axis axis, int value) {
	Common::LockGuard lock(m_mutex);

	if (auto* pad = InputPad(id); pad != nullptr) {
		auto& state = pad->state;
		state.time  = LibKernel::KernelGetProcessTime();

		int axis_id = static_cast<int>(axis);

		EXIT_IF(axis_id < 0 || axis_id >= static_cast<int>(Controller::Axis::AxisMax));

		state.axes[axis_id] = value;

		uint32_t trigger = 0;
		if (axis == Controller::Axis::TriggerLeft) {
			trigger = PAD_BUTTON_L2;
		} else if (axis == Controller::Axis::TriggerRight) {
			trigger = PAD_BUTTON_R2;
		}
		if (trigger != 0) {
			state.buttons = value > 0 ? state.buttons | trigger : state.buttons & ~trigger;
		}

		pad->AddState();
	}
}

void GameController::RightStick(int id, int x, int y) {
	Common::LockGuard lock(m_mutex);

	if (auto* pad = InputPad(id); pad != nullptr) {
		pad->state.time                                 = LibKernel::KernelGetProcessTime();
		pad->state.axes[static_cast<int>(Axis::RightX)] = x;
		pad->state.axes[static_cast<int>(Axis::RightY)] = y;
		pad->AddState();
	}
}

void GameController::TouchPad(int id, int finger, bool down, float x, float y) {
	if (finger < 0 || finger >= 2) {
		return;
	}

	Common::LockGuard lock(m_mutex);
	if (auto* pad = InputPad(id); pad != nullptr) {
		auto& state = pad->state;
		auto& touch = state.touch[finger];
		state.time  = LibKernel::KernelGetProcessTime();
		if (down && !touch.down) {
			touch.id           = pad->next_touch_id;
			pad->next_touch_id = pad->next_touch_id == 127 ? 1 : pad->next_touch_id + 1;
		}
		touch.down = down;
		touch.x    = static_cast<uint16_t>(std::clamp(x, 0.0f, 1.0f) * 1920.0f);
		touch.y    = static_cast<uint16_t>(std::clamp(y, 0.0f, 1.0f) * 943.0f);
		if (id == HOST_INPUT_CONTROLLER_ID) {
			state.buttons =
			    down ? state.buttons | PAD_BUTTON_TOUCH_PAD : state.buttons & ~PAD_BUTTON_TOUCH_PAD;
		}
		pad->AddState();
	}
}

namespace {

using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>; // x, y, z, w

Vec3 QuatRotate(const Quat& q, const Vec3& v) {
	const Vec3 t {2.0f * (q[1] * v[2] - q[2] * v[1]), 2.0f * (q[2] * v[0] - q[0] * v[2]),
	              2.0f * (q[0] * v[1] - q[1] * v[0])};
	return {v[0] + q[3] * t[0] + (q[1] * t[2] - q[2] * t[1]),
	        v[1] + q[3] * t[1] + (q[2] * t[0] - q[0] * t[2]),
	        v[2] + q[3] * t[2] + (q[0] * t[1] - q[1] * t[0])};
}

Quat QuatConjugate(const Quat& q) {
	return {-q[0], -q[1], -q[2], q[3]};
}

float Length(const Vec3& v) {
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} // namespace

void GameController::Motion(int id, Sensor sensor, const float* data, uint64_t time_us) {
	Common::LockGuard lock(m_mutex);
	const int         slot = m_slots.SlotOf(id);
	if (slot == PlayerSlots::NoSlot || !m_pads[slot].motion_enabled) {
		return;
	}
	auto& pad = m_pads[slot];

	// Convert acceleration to G; angular velocity is already in rad/s.
	if (sensor == Sensor::Accel) {
		for (int i = 0; i < 3; i++) {
			pad.state.accel[i] = data[i] / SDL_STANDARD_GRAVITY;
		}
		// Capture gravity near 1 G to limit interference from linear acceleration.
		const float magnitude = Length(pad.state.accel);
		if (!pad.up_reference_valid && magnitude > 0.9f && magnitude < 1.1f) {
			const Vec3 up {pad.state.accel[0] / magnitude, pad.state.accel[1] / magnitude,
			               pad.state.accel[2] / magnitude};
			pad.up_reference       = QuatRotate(pad.state.orientation, up);
			pad.up_reference_valid = true;
		}
	} else {
		std::copy_n(data, 3, pad.state.gyro.begin());
		// Mahony tilt correction; heading is unobservable and reported gyro values stay raw.
		auto        rate      = pad.state.gyro;
		const float magnitude = Length(pad.state.accel);
		if (pad.up_reference_valid && magnitude > 0.8f && magnitude < 1.2f) {
			const Vec3 measured {pad.state.accel[0] / magnitude, pad.state.accel[1] / magnitude,
			                     pad.state.accel[2] / magnitude};
			const Vec3 predicted =
			    QuatRotate(QuatConjugate(pad.state.orientation), pad.up_reference);
			rate[0] += measured[1] * predicted[2] - measured[2] * predicted[1];
			rate[1] += measured[2] * predicted[0] - measured[0] * predicted[2];
			rate[2] += measured[0] * predicted[1] - measured[1] * predicted[0];
		}
		// Do not extrapolate a single sample across lost reports (e.g. loss of window focus).
		constexpr uint64_t max_gyro_interval_us = 100000;
		if (pad.gyro_time != 0 && time_us > pad.gyro_time &&
		    time_us - pad.gyro_time <= max_gyro_interval_us) {
			const float dt         = static_cast<float>(time_us - pad.gyro_time) * 0.000001f;
			const float speed      = Length(rate);
			const float half_angle = speed * dt * 0.5f;
			const float scale      = speed > 0.0f ? std::sin(half_angle) / speed : 0.0f;
			const float x          = rate[0] * scale;
			const float y          = rate[1] * scale;
			const float z          = rate[2] * scale;
			const float w          = std::cos(half_angle);
			const auto  q          = pad.state.orientation;
			// Accumulate body-local rotation relative to connection / orientation reset.
			pad.state.orientation = {q[3] * x + q[0] * w + q[1] * z - q[2] * y,
			                         q[3] * y - q[0] * z + q[1] * w + q[2] * x,
			                         q[3] * z + q[0] * y - q[1] * x + q[2] * w,
			                         q[3] * w - q[0] * x - q[1] * y - q[2] * z};
			float length        = 0.0f;
			for (float value: pad.state.orientation) {
				length += value * value;
			}
			length = std::sqrt(length);
			for (float& value: pad.state.orientation) {
				value /= length;
			}
		}
		pad.gyro_time = time_us;
	}
	pad.state.time = LibKernel::KernelGetProcessTime();
	pad.AddState();
}

void GameController::SetMotionSensorState(int slot, bool enable) {
	Common::LockGuard lock(m_mutex);
	auto&             pad = m_pads[slot];
	if (pad.motion_enabled != enable) {
		pad.motion_enabled     = enable;
		pad.gyro_time          = 0;
		pad.state.accel        = {0.0f, 1.0f, 0.0f};
		pad.state.gyro         = {};
		pad.up_reference_valid = false;
		pad.state.time         = LibKernel::KernelGetProcessTime();
		pad.AddState();
	}
}

void GameController::ResetOrientation(int slot) {
	Common::LockGuard lock(m_mutex);
	auto&             pad  = m_pads[slot];
	pad.state.orientation  = {0.0f, 0.0f, 0.0f, 1.0f};
	pad.gyro_time          = 0;
	pad.up_reference_valid = false;
	pad.state.time         = LibKernel::KernelGetProcessTime();
	pad.AddState();
}

void GameController::ResetInputState() {
	Common::LockGuard lock(m_mutex);
	for (auto& pad: m_pads) {
		pad.state.buttons = 0;
		std::fill_n(pad.state.axes, 4, 128);
		std::fill_n(pad.state.axes + 4, 2, 0);
		for (auto& touch: pad.state.touch) {
			touch = {};
		}
		pad.state.time    = LibKernel::KernelGetProcessTime();
		pad.states_num    = 0;
		pad.first_state   = 0;
		pad.next_touch_id = 1;
		pad.AddState();
	}
}

void GameController::ReleaseHostPads() {
	Common::LockGuard lock(m_mutex);
	DualSenseHaptics::Shutdown();

	std::vector<SDL_Gamepad*> pads;
	for (const auto id: m_host_pads) {
		if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id));
		    pad != nullptr) {
			if (SDL_GetGamepadType(pad) == SDL_GAMEPAD_TYPE_PS5) {
				DualSenseEffects effect {};
				effect.enable_bits     = 0x0c;
				effect.right_trigger[0] = 0x05;
				effect.left_trigger[0]  = 0x05;
				(void)SDL_SendGamepadEffect(pad, &effect, sizeof(effect));
			}
			(void)SDL_RumbleGamepad(pad, 0, 0, 0);
			(void)SDL_SetGamepadLED(pad, 0, 0, 0);
			pads.push_back(pad);
		}
	}

	if (!pads.empty()) {
		SDL_Delay(RELEASE_FLUSH_MS);
		for (auto* pad: pads) {
			SDL_CloseGamepad(pad);
		}
	}

	// A released gamepad plays as no one, and nothing the game asked of it is sent again.
	for (const auto id: m_host_pads) {
		if (const int slot = m_slots.Disconnect(id); slot != PlayerSlots::NoSlot) {
			m_pads[slot].ForgetInput();
			m_pads[slot].ForgetOutput();
		}
	}
	m_host_pads.clear();
}

void GameController::SetVibration(int slot, uint8_t large_motor, uint8_t small_motor) {
	Common::LockGuard lock(m_mutex);
	auto& pad           = m_pads[slot];
	pad.vibration       = {large_motor, small_motor};
	pad.vibration_until = SDL_GetTicks() + RUMBLE_DURATION_MS;
	ApplyVibration(slot);
}

void GameController::ApplyVibration(int slot) {
	const int id = m_slots.GamepadOf(slot);
	if (id == PlayerSlots::NoGamepad) {
		return;
	}
	const auto& output = m_pads[slot];
	const auto  now    = SDL_GetTicks();
	const auto  duration =
	    output.vibration_until > now ? static_cast<uint32_t>(output.vibration_until - now) : 0;
	const auto scale       = duration != 0 ? GetSettingScale(Setting::VibrationIntensity) : 0.0f;
	const auto large_motor = Scale(output.vibration[0], scale);
	const auto small_motor = Scale(output.vibration[1], scale);
	// The haptics path drives one gamepad, the one of player 1. A second gamepad on it would stop
	// the motors of the first.
	if (slot == 0 && DualSenseHaptics::SetVibration(id, large_motor, small_motor, duration)) {
		return;
	}

	auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id));
	if (pad == nullptr) {
		return;
	}

	const auto large = static_cast<uint16_t>(large_motor * 0x101U);
	const auto small = static_cast<uint16_t>(small_motor * 0x101U);
	if (!SDL_RumbleGamepad(pad, large, small, duration)) {
		LOGF("\t rumble failed: %s\n", SDL_GetError());
	}
}

int GameController::GetGamepadOfPlayerOne() {
	Common::LockGuard lock(m_mutex);
	const int         id = m_slots.GamepadOf(0);
	return id != PlayerSlots::NoGamepad ? id : HOST_INPUT_CONTROLLER_ID;
}

void GameController::SetLightBar(int slot, uint8_t r, uint8_t g, uint8_t b) {
	Common::LockGuard lock(m_mutex);
	if (const auto& color = Config::GetControllerColor()) {
		r = (*color)[0];
		g = (*color)[1];
		b = (*color)[2];
	}
	if (auto* pad = HostPad(slot); pad != nullptr) {
		(void)SDL_SetGamepadLED(pad, r, g, b);
	}
}

bool GameController::SetTriggerEffect(int slot, const PadTriggerEffectParam& param) {
	Common::LockGuard lock(m_mutex);
	if (!SendTriggerEffect(slot, param)) {
		return false;
	}
	auto& kept = m_pads[slot].trigger_effect;
	for (int i = 0; i < 2; i++) {
		if ((param.trigger_mask & (1u << i)) != 0) {
			kept.trigger_mask |= static_cast<uint8_t>(1u << i);
			kept.command[i] = param.command[i];
		}
	}
	return true;
}

bool GameController::SendTriggerEffect(int slot, const PadTriggerEffectParam& param) {
	if ((param.trigger_mask & ~0x03u) != 0) {
		return false;
	}

	DualSenseEffects effect {};
	const auto       scale = GetSettingScale(Setting::TriggerEffectIntensity);
	if ((param.trigger_mask & 0x01u) != 0) {
		effect.enable_bits |= 0x08;
		if (!trigger_effect_to_dualsense(param.command[0], effect.left_trigger, scale)) {
			return false;
		}
	}
	if ((param.trigger_mask & 0x02u) != 0) {
		effect.enable_bits |= 0x04;
		if (!trigger_effect_to_dualsense(param.command[1], effect.right_trigger, scale)) {
			return false;
		}
	}
	if (effect.enable_bits == 0) {
		return true;
	}

	auto* pad = HostPad(slot);
	if (pad != nullptr && SDL_GetGamepadType(pad) == SDL_GAMEPAD_TYPE_PS5) {
		(void)SDL_SendGamepadEffect(pad, &effect, sizeof(effect));
	}
	return true;
}

void GameController::GetConnectionInfo(int slot, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = IsConnected(slot);
	*count = m_pads[slot].connected_count;
}

void GameController::ReadState(int slot, ControllerState* state, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(state == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = IsConnected(slot);
	*count = m_pads[slot].connected_count;
	*state = m_pads[slot].state;
}

int GameController::ReadStates(int slot, ControllerState* states, int states_num, bool* flag,
                               int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(states == nullptr);
	EXIT_IF(states_num < 1 || states_num > static_cast<int>(Pad::STATES_MAX));

	Common::LockGuard lock(m_mutex);

	auto& pad = m_pads[slot];
	*flag     = IsConnected(slot);
	*count    = pad.connected_count;

	int ret_num = 0;

	if (*flag) {
		for (uint32_t i = 0; i < pad.states_num; i++) {
			if (ret_num >= states_num) {
				break;
			}
			auto index = (pad.first_state + i) % Pad::STATES_MAX;
			if (!pad.obtained[index]) {
				pad.obtained[index] = true;

				states[ret_num++] = pad.states[index];
			}
		}
	}

	return ret_num;
}

int GameController::GetSlotOfUser(int user_id) {
	Common::LockGuard lock(m_mutex);
	return m_slots.SlotOfUser(user_id);
}

int GameController::GetUserOfSlot(int slot) {
	Common::LockGuard lock(m_mutex);
	return m_slots.UserOf(slot);
}

void GameController::GetLoggedInUsers(int* users) {
	Common::LockGuard lock(m_mutex);
	for (int slot = 0; slot < PlayerSlots::MaxPlayers; ++slot) {
		users[slot] = m_slots.IsLoggedIn(slot) ? m_slots.UserOf(slot) : -1;
	}
}

bool GameController::TakeEvent(PlayerSlots::Event* event) {
	Common::LockGuard lock(m_mutex);
	return m_slots.TakeEvent(event);
}

void Connect(int id) {
	g_controller->Connect(id);
}

void Disconnect(int id) {
	g_controller->Disconnect(id);
}

void SetButton(int id, uint32_t button, bool down) {
	g_controller->Button(id, button, down);
}

void SetAxis(int id, Axis axis, int value) {
	g_controller->Axis(id, axis, value);
}

void SetRightStick(int id, int x, int y) {
	g_controller->RightStick(id, x, y);
}

void SetTouchPad(int id, int finger, bool down, float x, float y) {
	g_controller->TouchPad(id, finger, down, x, y);
}

void SetSensor(int id, Sensor sensor, const float* data, uint64_t time_us) {
	g_controller->Motion(id, sensor, data, time_us);
}

void ResetInputState() {
	g_controller->ResetInputState();
}

int GetGamepadOfPlayerOne() {
	return g_controller != nullptr ? g_controller->GetGamepadOfPlayerOne()
	                               : HOST_INPUT_CONTROLLER_ID;
}

int GetPlayerOfUser(int user_id) {
	return g_controller->GetSlotOfUser(user_id);
}

void GetLoggedInUsers(int* users) {
	g_controller->GetLoggedInUsers(users);
}

bool TakePlayerEvent(bool* login, int* user_id) {
	PlayerSlots::Event event;
	if (!g_controller->TakeEvent(&event)) {
		return false;
	}
	*login   = event.login;
	*user_id = g_controller->GetUserOfSlot(event.slot);
	return true;
}

int KYTY_SYSV_ABI PadInit() {
	PRINT_NAME();

	return OK;
}

// A pad port of a player. The standard port reads the player's gamepad. The special port is for
// special controllers (arcade sticks and the like), which are not emulated: it opens, and reads as
// no controller. Games that read both would otherwise see every press twice.
struct PadPort {
	int  player  = PlayerSlots::NoSlot;
	bool special = false;
};

// The port opened with these arguments; its player is NoSlot when the port cannot be opened.
static PadPort PortOf(int user_id, int type, int index) {
	constexpr int user_id_system     = 0xff;
	constexpr int port_type_standard = 0;
	constexpr int port_type_special  = 2;
	constexpr int port_type_remote   = 16;
	if (index != 0) {
		return {};
	}
	if (user_id == user_id_system && type == port_type_remote) {
		return {.player = 0};
	}
	if (type == port_type_standard || type == port_type_special) {
		return {.player = g_controller->GetSlotOfUser(user_id),
		        .special = type == port_type_special};
	}
	return {};
}

// Each port has one handle: 1 to 4 for the standard ports, 5 to 8 for the special ones. It stays
// valid while the player has no controller: the pad then reads as not connected.
static int HandleOf(const PadPort& port) {
	return port.player + 1 + (port.special ? PlayerSlots::MaxPlayers : 0);
}

static PadPort PortOfHandle(int handle) {
	if (handle >= 1 && handle <= 2 * PlayerSlots::MaxPlayers) {
		const bool special = handle > PlayerSlots::MaxPlayers;
		return {.player = (handle - 1) % PlayerSlots::MaxPlayers, .special = special};
	}
	return {};
}

static int PlayerOfHandle(int handle) {
	return PortOfHandle(handle).player;
}

static bool IsSpecialHandle(int handle) {
	return PortOfHandle(handle).special;
}

bool IsPadHandle(int handle) {
	return PlayerOfHandle(handle) != PlayerSlots::NoSlot;
}

int KYTY_SYSV_ABI PadOpen(int user_id, int type, int index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	const auto port = PortOf(user_id, type, index);
	if (port.player == PlayerSlots::NoSlot) {
		return pad_error_invalid_arg;
	}

	return HandleOf(port);
}

int KYTY_SYSV_ABI PadGetHandle(int user_id, int type, int index) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n",
	     user_id, type, index);

	constexpr int pad_error_device_no_handle = -2137915384; /* 0x80920008 */

	const auto port = PortOf(user_id, type, index);
	if (port.player == PlayerSlots::NoSlot) {
		return pad_error_device_no_handle;
	}

	return HandleOf(port);
}

int KYTY_SYSV_ABI PadSetMotionSensorState(int handle, bool enable) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));
	if (!IsSpecialHandle(handle)) {
		g_controller->SetMotionSensorState(player, enable);
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandState(int handle, bool enable) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadResetOrientation(int handle) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	if (!IsSpecialHandle(handle)) {
		g_controller->ResetOrientation(player);
	}
	return OK;
}

int KYTY_SYSV_ABI PadGetControllerInformation(int handle, PadControllerInformation* info) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int  connected_count = 0;
	bool connected       = false;

	if (!IsSpecialHandle(handle)) {
		g_controller->GetConnectionInfo(player, &connected, &connected_count);
	}

	std::memset(info, 0, sizeof(*info));

	info->touch_pixel_density   = 44.86f;
	info->touch_resolution_x    = 1920;
	info->touch_resolution_y    = 943;
	info->stick_dead_zone_left  = controller_get_axis(-32768, 32767, 8000) - 128;
	info->stick_dead_zone_right = controller_get_axis(-32768, 32767, 8000) - 128;
	info->connection_type       = 0;
	info->connected_count       = static_cast<uint8_t>(std::min(connected_count, 255));
	info->connected             = connected;
	info->device_class          = 0;

	return OK;
}

int KYTY_SYSV_ABI PadIsRemoteController(int handle, bool* is_remote) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (is_remote == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	*is_remote = false;
	return OK;
}

int KYTY_SYSV_ABI PadReadState(int handle, PadData* data) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState state;

	if (!IsSpecialHandle(handle)) {
		g_controller->ReadState(player, &state, &connected, &connected_count);
	}

	pad_fill_data(data, state, connected, connected_count);

	return OK;
}

int KYTY_SYSV_ABI PadRead(int handle, PadData* data, int num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(num < 1 || num > 64);
	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(PadData) * static_cast<size_t>(num));

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState states[64]      = {};

	int ret_num = IsSpecialHandle(handle) ? 0
	                                      : g_controller->ReadStates(player, states, num, &connected,
	                                                                 &connected_count);

	if (!connected || ret_num == 0) {
		if (connected) {
			g_controller->ReadState(player, &states[0], &connected, &connected_count);
		}
		ret_num = 1;
	}

	for (int i = 0; i < ret_num; i++) {
		pad_fill_data(&data[i], states[i], connected, connected_count);
	}

	return ret_num;
}

int KYTY_SYSV_ABI PadSetVibration(int handle, const PadVibrationParam* param) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	LOGF("\t large_motor = %d\n"
	     "\t small_motor = %d\n",
	     static_cast<int>(param->large_motor), static_cast<int>(param->small_motor));

	if (!IsSpecialHandle(handle)) {
		g_controller->SetVibration(player, param->large_motor, param->small_motor);
	}

	return OK;
}

int KYTY_SYSV_ABI PadResetLightBar(int handle) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetLightBar(int handle, const PadLightBarParam* param) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	if (!IsSpecialHandle(handle)) {
		g_controller->SetLightBar(player, param->r, param->g, param->b);
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetTriggerEffect(int handle, const PadTriggerEffectParam* param) {
	PRINT_NAME();

	const int player = PlayerOfHandle(handle);
	if (player == PlayerSlots::NoSlot) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	if (IsSpecialHandle(handle)) {
		return OK;
	}
	return g_controller->SetTriggerEffect(player, *param) ? OK : PAD_ERROR_INVALID_ARG;
}

} // namespace Libs::Controller
