#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_STATELAYOUT_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_STATELAYOUT_H_

#include "graphics/guest_gpu/capture/captureFile.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <type_traits>

namespace Libs::Graphics::Capture {

// Raise this when `CommandProcessorState` or `GuestGpu::StartedSubmission` changes the meaning of
// a field without changing size: a capture stores both as raw bytes, and the sizes alone would
// then let an older capture through.
inline constexpr uint32_t StateLayoutVersion = 1;

static_assert(std::is_trivially_copyable_v<CommandProcessorState>);
static_assert(std::is_trivially_copyable_v<GuestGpu::StartedSubmission>);

// The layout of the structures this build stores in a capture.
[[nodiscard]] inline StateLayout CurrentStateLayout() {
	return {.version                 = StateLayoutVersion,
	        .processor_state_size    = sizeof(CommandProcessorState),
	        .started_submission_size = sizeof(GuestGpu::StartedSubmission)};
}

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_STATELAYOUT_H_
