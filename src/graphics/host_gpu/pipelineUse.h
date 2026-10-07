#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINEUSE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINEUSE_H_

#include <cstdint>

// What a draw does when its graphics pipeline is not built yet.
//
// A pipeline is built on a worker thread. With the driver's caches warm that takes about a
// millisecond; on a machine that sees the game for the first time the host has to compile the
// shaders, which takes 100 ms to 2 s per pipeline, and a level asks for several new pipelines
// per second. A draw that waited would stop the picture for each of them.
//
// So a draw that only changes the picture is left out until its pipeline is built: the object or
// effect appears a few frames late, once per pipeline on this machine. A draw that changes more
// than the picture waits: what its shaders write to memory, and the depth clear it carries, are
// read by later work, and leaving them out would change what the game computes.
//
// A picture the game draws once and reads for the rest of the level (a colour table, a light
// probe, an atlas) is the same: left out once, it stays missing. The game's own frames tell the
// two apart: a target drawn in the previous frame as well is drawn again anyway, so a draw into
// it may be left out; a draw into any other target waits.
//
// What this cannot see: a draw into one region of a target the game otherwise redraws (that
// region stays missing), and a target the game later copies into memory it reads (the bytes
// miss the draw). Both need knowledge of what the game does next, which the draw does not have.

namespace Libs::Graphics {

struct DrawEffects {
	// A shader of the draw writes buffers or images, or uses atomics on them.
	bool writes_memory = false;
	// The draw clears its depth or stencil target.
	bool clears_depth = false;
	// A target of the draw was not drawn in the previous frame.
	bool writes_fresh_target = false;
};

enum class PipelineUse : uint8_t {
	// The pipeline is built: draw.
	Draw,
	// Wait for the build, then draw.
	Wait,
	// Leave the draw out; it is drawn again by the game once the pipeline is built.
	LeaveOut,
};

// `always_wait`: every draw waits (--pipeline-wait), so the picture is exact from the first frame.
[[nodiscard]] constexpr PipelineUse PlanPipelineUse(const DrawEffects& effects, bool ready,
                                                    bool always_wait) noexcept {
	if (ready) {
		return PipelineUse::Draw;
	}
	if (always_wait || effects.writes_memory || effects.clears_depth ||
	    effects.writes_fresh_target) {
		return PipelineUse::Wait;
	}
	return PipelineUse::LeaveOut;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINEUSE_H_ */
