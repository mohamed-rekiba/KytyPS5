#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_

#include <cstdint>
#include <span>

namespace Libs::Graphics {

// Sees the guest GPU stream in the order the GPU thread executes it. The capture recorder and the
// capture player attach here. Every call is made on the GPU thread.
class GuestGpuObserver {
public:
	enum class SubmissionKind { Graphics, Compute, CpuFlip };

	virtual ~GuestGpuObserver() = default;

	// A submission is about to run its first slice. `queue` is the guest's queue number for a
	// compute submission and 0 otherwise.
	virtual void OnSubmissionStart(SubmissionKind kind, uint32_t queue,
	                               std::span<const uint32_t> commands,
	                               std::span<const uint32_t> constant_commands) = 0;
	// The guest's suspend point number `frame_id` was processed: the graphics command processor
	// is reset.
	virtual void OnSuspendPoint(uint32_t frame_id) = 0;
	// A wait on guest memory is about to be tested, for the first time or again.
	virtual void OnBeforeWait(uint64_t address, uint32_t size) = 0;
	// A wait on guest memory failed its test; the submission blocks and retries later.
	virtual void OnWaitBlocked(uint64_t address, uint32_t size) = 0;
	// A wait on guest memory passes. `had_blocked`: it failed its test at least once before.
	virtual void OnWaitPassed(uint64_t address, uint32_t size, bool had_blocked) = 0;
	// A flip of display buffer `index` on video-out port `handle` was recorded from the command
	// stream.
	virtual void OnFlip(int handle, int index) = 0;
	// A submission blocked on something other than a wait on guest memory; `what` names it.
	virtual void OnOtherBlock(const char* what) = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_
