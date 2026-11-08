// Tests for the benchmark script: key and gamepad presses replayed at frames
// or after delays.

#include "graphics/presentation/window/benchGamepad.h"
#include "graphics/presentation/window/benchScript.h"

#include <SDL3/SDL.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::BenchGamepad;
using Libs::Graphics::BenchScript;
using Libs::Graphics::BenchStep;
using Libs::Graphics::ParseBenchScript;

int g_failures = 0;

void Check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    g_failures++;
  }
}

std::vector<BenchStep> Parse(const char *text) {
  std::string error;
  auto steps = ParseBenchScript(text, error);
  Check(steps.has_value(), "a valid script parses");
  return steps.value_or(std::vector<BenchStep>{});
}

void TestParse() {
  const auto steps = Parse("# menu\n"
                           "f2500 J down\n"
                           "+200 J up   # release\n"
                           "\n"
                           "+50 Left Shift down\n"
                           "f9000 quit\n");
  Check(steps.size() == 4, "comments and blank lines are skipped");
  if (steps.size() != 4) {
    return;
  }
  Check(steps[0].when == BenchStep::When::Frame && steps[0].value == 2500 &&
            steps[0].key == "J" && steps[0].down,
        "a frame step names its frame, key and action");
  Check(steps[1].when == BenchStep::When::AfterMs && steps[1].value == 200 &&
            !steps[1].down,
        "a delayed step names its delay");
  Check(steps[2].key == "Left Shift", "a key name may hold a space");
  Check(steps[3].quit && steps[3].value == 9000,
        "quit ends the run at a frame");
}

void TestParseGamepad() {
  const auto steps = Parse("+10 pad a down # frame 5, gamepad 3\n"
                           "+20 pad dpup up\n"
                           "+30 axis leftx -32768\n"
                           "+40 axis righttrigger 32767\n"
                           "+50 Keypad 1 down\n");
  Check(steps.size() == 5, "pad and axis steps parse");
  if (steps.size() != 5) {
    return;
  }
  Check(steps[0].input == BenchStep::Input::PadButton && steps[0].key == "a" &&
            steps[0].down && steps[0].value == 10,
        "a pad step names its button and action");
  Check(steps[1].input == BenchStep::Input::PadButton &&
            steps[1].key == "dpup" && !steps[1].down,
        "a pad step can release a button");
  Check(steps[2].input == BenchStep::Input::PadAxis &&
            steps[2].key == "leftx" && steps[2].axis_value == -32768,
        "an axis step names its axis and a negative value");
  Check(steps[3].input == BenchStep::Input::PadAxis &&
            steps[3].key == "righttrigger" && steps[3].axis_value == 32767,
        "an axis step takes the largest value");
  Check(steps[4].input == BenchStep::Input::Key && steps[4].key == "Keypad 1",
        "a key name that holds \"pad\" stays a key");
  Check(BenchScript(Parse("+1 pad a down\n")).UsesGamepad() &&
            BenchScript(Parse("+1 axis leftx 5\n")).UsesGamepad() &&
            !BenchScript(Parse("+1 J down\n+1 quit\n")).UsesGamepad(),
        "a script uses the gamepad when a step presses or moves it");
}

void TestParseErrors() {
  for (const char *bad :
       {"2500 J down", "fx J down", "f10 J press", "f10 J", "+ J down",
        "+1 axis leftx", "+1 axis leftx 32768", "+1 axis leftx -32769",
        "+1 axis leftx half", "+1 pad a"}) {
    std::string error;
    Check(!ParseBenchScript(bad, error).has_value() &&
              error.starts_with("line 1: "),
          "a wrong step is refused with its line");
  }
  std::string error;
  Check(!ParseBenchScript("f1 J down\n\nf2 J sideways\n", error).has_value() &&
            error.starts_with("line 3: "),
        "the error names the line of the wrong step");
}

void TestFramesAndDelays() {
  BenchScript script(Parse("f10 J down\n+100 J up\n+50 quit\n"));
  std::vector<std::string> fired;
  const auto fire = [&](const BenchStep &step) {
    fired.push_back(step.quit ? "quit"
                              : step.key + (step.down ? " down" : " up"));
  };
  script.Advance(5, 1000, fire);
  Check(fired.empty(), "nothing fires before its frame");
  Check(script.WaitMs(1000, 5, 1000) == 5, "a frame step is polled");
  script.Advance(12, 1010, fire);
  Check(fired.size() == 1 && fired[0] == "J down",
        "a frame step fires at a later frame too");
  Check(script.WaitMs(1050, 5, 1000) == 60,
        "the loop waits for the delay that is left");
  script.Advance(13, 1109, fire);
  Check(fired.size() == 1, "a delay counts from the step before it");
  script.Advance(13, 1110, fire);
  Check(fired.size() == 2 && fired[1] == "J up",
        "a delayed step fires when its delay ends");
  script.Advance(13, 2000, fire);
  Check(fired.size() == 3 && fired[2] == "quit", "quit fires like a step");
  Check(script.Done() && script.WaitMs(2000, 5, 1000) == 1000,
        "a finished script stops waking");
}

void TestStepsDueTogetherFireInOrder() {
  BenchScript script(Parse("f10 J down\n+0 J up\nf10 I down\n"));
  std::vector<std::string> fired;
  script.Advance(10, 0, [&](const BenchStep &step) {
    fired.push_back(step.key + (step.down ? " down" : " up"));
  });
  Check(fired == std::vector<std::string>{"J down", "J up", "I down"},
        "steps due at one call fire in script order");
}

// The virtual pad reports what a step sets through the SDL gamepad API, as
// a pad on the host would: every button, both sticks and both triggers.
void TestVirtualGamepad() {
  if (!SDL_Init(SDL_INIT_GAMEPAD)) {
    Check(false, "SDL starts its gamepad subsystem");
    return;
  }
  {
    BenchGamepad bench;
    Check(bench.Attach(), "the virtual gamepad attaches");
    SDL_Gamepad *pad = SDL_OpenGamepad(bench.Id());
    Check(pad != nullptr, "the virtual gamepad opens as a gamepad");
    if (pad != nullptr) {
      for (int button = 0; button < SDL_GAMEPAD_BUTTON_COUNT; ++button) {
        const auto b = static_cast<SDL_GamepadButton>(button);
        bench.SetButton(b, true);
        SDL_UpdateJoysticks();
        const bool down = SDL_GetGamepadButton(pad, b);
        bench.SetButton(b, false);
        SDL_UpdateJoysticks();
        Check(down && !SDL_GetGamepadButton(pad, b),
              "a button step presses and releases that button");
      }
      const auto axis_round_trip = [&](SDL_GamepadAxis axis, int value) {
        bench.SetAxis(axis, value);
        SDL_UpdateJoysticks();
        return SDL_GetGamepadAxis(pad, axis);
      };
      for (const auto axis :
           {SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY,
            SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY}) {
        Check(axis_round_trip(axis, -32768) == -32768 &&
                  axis_round_trip(axis, 1000) == 1000 &&
                  axis_round_trip(axis, 32767) == 32767 &&
                  axis_round_trip(axis, 0) == 0,
              "a stick step sets that stick to its value");
      }
      for (const auto axis :
           {SDL_GAMEPAD_AXIS_LEFT_TRIGGER, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER}) {
        bool exact = SDL_GetGamepadAxis(pad, axis) == 0;
        for (int value = 0; value <= 32767 && exact; ++value) {
          exact = axis_round_trip(axis, value) == value;
        }
        Check(exact, "a trigger step sets that trigger to its value");
      }
      SDL_CloseGamepad(pad);
    }
  }
  SDL_Quit();
}

} // namespace

int main() {
  TestParse();
  TestParseGamepad();
  TestParseErrors();
  TestVirtualGamepad();
  TestFramesAndDelays();
  TestStepsDueTogetherFireInOrder();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "bench_script: all checks passed\n";
  return 0;
}
