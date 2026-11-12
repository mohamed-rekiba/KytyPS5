#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHSCRIPT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHSCRIPT_H_

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// A benchmark script (--bench-script): key and gamepad presses replayed without a window in
// front, so a measured run takes the same route every time, also while the screen is locked.
//
// One step per line; '#' starts a comment. A step starts with when it is due:
//   f<frame>   at the first presented frame >= <frame>
//   +<ms>      <ms> milliseconds after the step before it
// and then names the input:
//   <key> down|up           an SDL key name ("J", "W", "Left Shift"): the keyboard mapping applies
//   pad <button> down|up    an SDL gamepad button name ("a", "start", "dpup")
//   axis <axis> <value>     an SDL gamepad axis name ("leftx", "righttrigger") and its value,
//                           -32768 to 32767 (a trigger 0 to 32767)
//   quit                    close the emulator
// Pad and axis steps drive a virtual gamepad (benchGamepad.h). KYTY_INPUT_LOG writes a player's
// input in this form.
struct BenchStep {
	enum class When { Frame, AfterMs };
	enum class Input { Key, PadButton, PadAxis };
	When        when  = When::Frame;
	uint64_t    value = 0;
	Input       input = Input::Key;
	std::string key; // the key, button or axis name
	bool        down       = false;
	int         axis_value = 0;
	bool        quit       = false;
};

namespace BenchScriptDetail {

inline std::string_view Trim(std::string_view s) {
	while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
		s.remove_prefix(1);
	}
	while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
		s.remove_suffix(1);
	}
	return s;
}

template <typename T>
inline bool ParseNumber(std::string_view s, T& value) {
	const auto* end    = s.data() + s.size();
	const auto  result = std::from_chars(s.data(), end, value);
	return !s.empty() && result.ec == std::errc {} && result.ptr == end;
}

// Removes `word` and the blanks after it from the start of s, when s starts with it.
inline bool TakeWord(std::string_view& s, std::string_view word) {
	if (!s.starts_with(word) || s.size() == word.size() ||
	    (s[word.size()] != ' ' && s[word.size()] != '\t')) {
		return false;
	}
	s = Trim(s.substr(word.size()));
	return true;
}

} // namespace BenchScriptDetail

// The steps of a script, or nullopt with the line that is wrong in `error`.
inline std::optional<std::vector<BenchStep>> ParseBenchScript(std::string_view text,
                                                              std::string&     error) {
	using BenchScriptDetail::ParseNumber;
	using BenchScriptDetail::TakeWord;
	using BenchScriptDetail::Trim;
	std::vector<BenchStep> steps;
	uint32_t               line_number = 0;
	while (!text.empty()) {
		const auto       eol  = text.find('\n');
		std::string_view line = text.substr(0, eol);
		text.remove_prefix(eol == std::string_view::npos ? text.size() : eol + 1);
		line_number++;
		if (const auto hash = line.find('#'); hash != std::string_view::npos) {
			line = line.substr(0, hash);
		}
		line = Trim(line);
		if (line.empty()) {
			continue;
		}
		const auto fail = [&](const char* what) {
			error = "line " + std::to_string(line_number) + ": " + what;
			return std::nullopt;
		};
		const auto when_end = line.find_first_of(" \t");
		const auto when     = line.substr(0, when_end);
		BenchStep  step;
		if (when.size() > 1 && when[0] == 'f') {
			step.when = BenchStep::When::Frame;
		} else if (when.size() > 1 && when[0] == '+') {
			step.when = BenchStep::When::AfterMs;
		} else {
			return fail("a step starts with f<frame> or +<ms>");
		}
		if (!ParseNumber(when.substr(1), step.value)) {
			return fail("the frame or delay is not a number");
		}
		auto rest =
		    when_end == std::string_view::npos ? std::string_view {} : Trim(line.substr(when_end));
		if (rest == "quit") {
			step.quit = true;
			steps.push_back(std::move(step));
			continue;
		}
		if (TakeWord(rest, "axis")) {
			step.input             = BenchStep::Input::PadAxis;
			const auto value_start = rest.find_last_of(" \t");
			if (value_start == std::string_view::npos) {
				return fail("an axis step names an axis and a value");
			}
			if (!ParseNumber(rest.substr(value_start + 1), step.axis_value) ||
			    step.axis_value < -32768 || step.axis_value > 32767) {
				return fail("the axis value is not a number from -32768 to 32767");
			}
			step.key = std::string(Trim(rest.substr(0, value_start)));
			steps.push_back(std::move(step));
			continue;
		}
		if (TakeWord(rest, "pad")) {
			step.input = BenchStep::Input::PadButton;
		}
		const auto action_start = rest.find_last_of(" \t");
		if (action_start == std::string_view::npos) {
			return fail("a step names a key and down or up, or quit");
		}
		const auto action = rest.substr(action_start + 1);
		if (action == "down") {
			step.down = true;
		} else if (action != "up") {
			return fail("the action is not down or up");
		}
		step.key = std::string(Trim(rest.substr(0, action_start)));
		steps.push_back(std::move(step));
	}
	return steps;
}

// Which steps are due. The window loop calls Advance with the presented frame count and a
// millisecond clock, and waits no longer than WaitMs between calls.
class BenchScript {
public:
	explicit BenchScript(std::vector<BenchStep> steps): m_steps(std::move(steps)) {}

	// Runs `fire` for each due step, in order. A delay counts from the time the step before it
	// was due (a frame step: when it fired); the first delayed step counts from the first call.
	template <typename Fire>
	void Advance(uint64_t frame, uint64_t now_ms, Fire&& fire) {
		if (!m_started) {
			m_started = true;
			m_last_ms = now_ms;
		}
		while (m_next < m_steps.size()) {
			const auto& step = m_steps[m_next];
			const bool due = step.when == BenchStep::When::Frame ? frame >= step.value
			                                                     : now_ms >= m_last_ms + step.value;
			if (!due) {
				return;
			}
			// A delay counts from when the step before it was due, so a late step does not push the
			// steps after it: a recorded log replays at its own times.
			m_last_ms = step.when == BenchStep::When::Frame ? now_ms : m_last_ms + step.value;
			m_next++;
			fire(step);
		}
	}

	[[nodiscard]] bool Done() const { return m_next >= m_steps.size(); }

	// True when a step presses a gamepad button or moves an axis.
	[[nodiscard]] bool UsesGamepad() const {
		return std::any_of(m_steps.begin(), m_steps.end(), [](const BenchStep& step) {
			return !step.quit && step.input != BenchStep::Input::Key;
		});
	}

	// How long the loop may wait before the next step can be due: the delay left for a delayed
	// step, `frame_poll_ms` for a step that waits for a frame, `max_ms` when the script is done.
	[[nodiscard]] int WaitMs(uint64_t now_ms, int frame_poll_ms, int max_ms) const {
		if (Done()) {
			return max_ms;
		}
		const auto& step = m_steps[m_next];
		if (step.when == BenchStep::When::Frame || !m_started) {
			return std::min(frame_poll_ms, max_ms);
		}
		const auto due = m_last_ms + step.value;
		return due <= now_ms ? 0 : static_cast<int>(std::min<uint64_t>(due - now_ms, max_ms));
	}

private:
	std::vector<BenchStep> m_steps;
	size_t                 m_next    = 0;
	uint64_t               m_last_ms = 0;
	bool                   m_started = false;
};

// The lines that KYTY_INPUT_LOG writes: a player's input as bench script steps, so a log replays
// as it is. Each step's delay counts from the step before it; the first from `start_ms`. The
// presented frame and the gamepad are in a comment.
class InputLogFormat {
public:
	// A stick sends an event for each small move. A move smaller than this is not written, and a
	// value this close to rest is rest (0): a stick at rest seldom reads exactly 0.
	static constexpr int AxisStep = 2048;

	explicit InputLogFormat(uint64_t start_ms): m_last_ms(start_ms) {}

	// Each returns the line to write, or an empty string when there is nothing to write.
	std::string Key(std::string_view name, bool down, uint64_t now_ms, uint64_t frame) {
		if (name.empty() || name.find('#') != std::string_view::npos) {
			// '#' starts a comment in a script, so the step cannot be written.
			return "# key \"" + std::string(name) + "\" cannot be a step\n";
		}
		return Line(std::string(name) + (down ? " down" : " up"), now_ms, frame, -1);
	}

	std::string Button(std::string_view name, bool down, uint64_t now_ms, uint64_t frame,
	                   int gamepad) {
		return Line("pad " + std::string(name) + (down ? " down" : " up"), now_ms, frame, gamepad);
	}

	std::string Axis(std::string_view name, int value, uint64_t now_ms, uint64_t frame,
	                 int gamepad) {
		const int at   = (value > -AxisStep && value < AxisStep) ? 0 : value;
		auto      last = m_axes.find(name);
		const int was  = last == m_axes.end() ? 0 : last->second;
		if (at == was || (at != 0 && at - was < AxisStep && was - at < AxisStep)) {
			return {};
		}
		m_axes[std::string(name)] = at;
		return Line("axis " + std::string(name) + " " + std::to_string(at), now_ms, frame, gamepad);
	}

private:
	std::string Line(const std::string& step, uint64_t now_ms, uint64_t frame, int gamepad) {
		std::string line = "+" + std::to_string(now_ms >= m_last_ms ? now_ms - m_last_ms : 0) +
		                   " " + step + " # frame " + std::to_string(frame);
		if (gamepad >= 0) {
			line += ", gamepad " + std::to_string(gamepad);
		}
		m_last_ms = now_ms;
		return line + "\n";
	}

	uint64_t                                m_last_ms;
	std::map<std::string, int, std::less<>> m_axes;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_BENCHSCRIPT_H_ */
