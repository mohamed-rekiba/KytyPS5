#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_

#include <cstdint>
#include <span>

namespace Libs::Graphics {

// Sees the guest GPU stream in the order the GPU thread executes it. The capture recorder and the
// capture player attach here. Every call is made on the GPU thread, except `OnGuestWrite`.
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
	// Guest memory was written: `size` bytes at `address`. `Buffers`, `Images`, `Both`: the guest
	// wrote, and that cache of the renderer was told. `Host`: the renderer itself copied bytes
	// the host GPU held to guest memory, and no cache was told. Called on the thread that wrote,
	// which is a guest thread inside a page fault most of the time: no waiting, and no call back
	// into the renderer.
	enum class WriteTarget { Buffers = 1, Images = 2, Both = 3, Host = 4 };
	virtual void OnGuestWrite(WriteTarget target, uint64_t address, uint64_t size) noexcept = 0;
	// The renderer is about to copy `size` bytes of guest memory at `address` to the host, in
	// the middle of its work: nothing that touches the renderer may be done here.
	virtual void OnGuestRead(uint64_t address, uint64_t size) = 0;
	// The renderer handed a draw or a dispatch to the host GPU. Called in the middle of that
	// work: nothing that touches the renderer may be done here.
	virtual void OnHostWork() = 0;
	// The GPU thread is between two commands of the stream, where it also runs what other
	// threads ask of it. The renderer may be used here.
	virtual void OnBetweenCommands() = 0;
	// A submission blocked on something other than a wait on guest memory; `what` names it.
	virtual void OnOtherBlock(const char* what) = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUOBSERVER_H_
