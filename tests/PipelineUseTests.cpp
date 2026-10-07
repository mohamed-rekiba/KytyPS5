// Tests for the rule that decides what a draw does when its pipeline is not built yet.

#include "graphics/host_gpu/pipelineUse.h"

#include <iostream>

namespace {

using Libs::Graphics::DrawEffects;
using Libs::Graphics::PipelineUse;
using Libs::Graphics::PlanPipelineUse;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

void TestAReadyPipelineDraws() {
	for (const bool always_wait: {false, true}) {
		Check(PlanPipelineUse({.writes_memory = true, .clears_depth = true, .writes_fresh_target = true},
		                      true, always_wait) == PipelineUse::Draw,
		      "a built pipeline is used at once, whatever the draw does");
	}
}

void TestAVisualDrawIsLeftOutWhileItsPipelineBuilds() {
	Check(PlanPipelineUse({}, false, false) == PipelineUse::LeaveOut,
	      "a draw that only changes the picture is left out while its pipeline builds");
}

void TestADrawWithEffectsBeyondThePictureWaits() {
	Check(PlanPipelineUse({.writes_memory = true}, false, false) == PipelineUse::Wait,
	      "a draw whose shaders write buffers or images waits for its pipeline");
	Check(PlanPipelineUse({.clears_depth = true}, false, false) == PipelineUse::Wait,
	      "a draw that clears its depth or stencil target waits for its pipeline");
	Check(PlanPipelineUse({.writes_fresh_target = true}, false, false) == PipelineUse::Wait,
	      "a draw into a target the game did not draw last frame waits: it may never be drawn "
	      "again");
}

void TestEveryDrawWaitsWhenAsked() {
	Check(PlanPipelineUse({}, false, true) == PipelineUse::Wait,
	      "with --pipeline-wait a visual draw waits too");
}

} // namespace

int main() {
	TestAReadyPipelineDraws();
	TestAVisualDrawIsLeftOutWhileItsPipelineBuilds();
	TestADrawWithEffectsBeyondThePictureWaits();
	TestEveryDrawWaitsWhenAsked();
	if (g_failures != 0) {
		std::cerr << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "pipeline use: all checks passed\n";
	return 0;
}
