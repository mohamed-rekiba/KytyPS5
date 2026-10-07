#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HOSTLOWERING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HOSTLOWERING_H_

#include "graphics/host_gpu/hostCapabilities.h"

#include <cstdint>

// How the renderer does a thing on the host at hand, where hosts differ. Each choice is a pure
// function of the host and of the work, so it can be tested without a GPU.

namespace Libs::Graphics {

// How a pipeline's descriptors reach the device.
enum class DescriptorDelivery : uint8_t {
	// vkCmdPushDescriptorSet: no descriptor set to allocate.
	Pushed,
	// A descriptor set from a pool, bound with vkCmdBindDescriptorSets.
	Sets,
};

struct DescriptorLayoutShape {
	uint32_t descriptors     = 0; // every element of every binding
	uint32_t storage_buffers = 0; // of which storage buffers
};

// Descriptors are pushed when the device can push them, the set fits its limit, and the driver's
// defect does not hit the set: a pushed storage buffer has no size, and every storage buffer of a
// guest shader is read with a bounds check on its size.
[[nodiscard]] constexpr DescriptorDelivery ChooseDescriptorDelivery(const HostGpu& host,
                                                                    uint32_t max_push_descriptors,
                                                                    DescriptorLayoutShape shape) {
	if (!host.capabilities.push_descriptors || shape.descriptors > max_push_descriptors) {
		return DescriptorDelivery::Sets;
	}
	if (host.faults.pushed_buffers_have_no_size && shape.storage_buffers != 0) {
		return DescriptorDelivery::Sets;
	}
	return DescriptorDelivery::Pushed;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HOSTLOWERING_H_ */
