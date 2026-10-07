#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_SUBMITPLAN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_SUBMITPLAN_H_

#include <cstdint>

namespace Libs::Graphics {

// When the open host command buffer is submitted.
//
// Every submission costs the host driver a fixed amount of work before the GPU starts it (on
// MoltenVK about 2 ms of encoding and driver processing per command buffer, for a median of 0.1 ms
// of GPU work when every guest command buffer was submitted on its own). Guest command buffers are
// therefore merged into one host submission until one of these holds:
//   - a completion callback waits for the open submission (an end-of-pipe interrupt, a flip);
//   - the GPU thread is about to wait for the guest, so nothing more would join;
//   - the open submission holds enough draws and dispatches, or its first command is old enough,
//     so that the GPU is not left idle while the CPU records.
// Points that need the submission for correctness (a read-back of bytes the open submission writes,
// a wait for its tick, Finish) submit directly and do not come here.
struct SubmitState {
	// Draws and dispatches recorded in the open command buffer.
	uint32_t open_work = 0;
	// Completion callbacks queued for the open command buffer's tick.
	uint32_t open_callbacks = 0;
	// Any command was recorded in the open command buffer (transfers count, draws count).
	bool open_used = false;
	// Microseconds since the first command went into the open command buffer.
	uint64_t open_age_us = 0;
	// The GPU thread has nothing more to record right now and is about to sleep.
	bool about_to_wait = false;
};

enum class SubmitReason : uint8_t {
	None,
	// A completion callback waits for the open submission.
	Callbacks,
	// The GPU thread is about to wait; what is recorded goes to the GPU now.
	Idle,
	// Enough draws and dispatches are recorded.
	Work,
	// The first command has waited long enough.
	Age,
};

// Starting values, measured on an M4 Max with MoltenVK 1.4.2 (about 25 us of recording per draw):
// 4 ms of recording is about 160 draws, two to three guest command buffers of a typical frame.
inline constexpr uint32_t SUBMIT_WORK_LIMIT  = 512;
inline constexpr uint64_t SUBMIT_AGE_LIMIT_US = 4000;

[[nodiscard]] constexpr SubmitReason PlanSubmit(const SubmitState& state) noexcept {
	if (!state.open_used && state.open_callbacks == 0) {
		return SubmitReason::None;
	}
	if (state.open_callbacks != 0) {
		return SubmitReason::Callbacks;
	}
	if (state.about_to_wait) {
		return SubmitReason::Idle;
	}
	if (state.open_work >= SUBMIT_WORK_LIMIT) {
		return SubmitReason::Work;
	}
	if (state.open_age_us >= SUBMIT_AGE_LIMIT_US) {
		return SubmitReason::Age;
	}
	return SubmitReason::None;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_SUBMITPLAN_H_ */
