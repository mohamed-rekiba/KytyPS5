#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_DEVICESUITABILITY_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_DEVICESUITABILITY_H_

#include <string>
#include <vector>

// Decides whether a Vulkan device can run the emulator and which optional capabilities it offers.
// The decision uses plain data only, so it can be tested without a GPU.

namespace Libs::Graphics {

// What the physical device reports. A feature of an extension is only meaningful when the
// extension is listed in `extensions`.
struct DeviceFacts {
	std::vector<std::string> extensions;
	// The driver is MoltenVK (VkPhysicalDeviceVulkan12Properties::driverID).
	bool driver_is_moltenvk = false;

	// Core 1.0 features.
	bool depth_clamp                               = false;
	bool depth_bias_clamp                          = false;
	bool sample_rate_shading                       = false;
	bool shader_clip_distance                      = false;
	bool shader_cull_distance                      = false;
	bool large_points                              = false;
	bool multi_viewport                            = false;
	bool fill_mode_non_solid                       = false;
	bool fragment_stores_and_atomics               = false;
	bool vertex_pipeline_stores_and_atomics        = false;
	bool sampler_anisotropy                        = false;
	bool robust_buffer_access                      = false;
	bool depth_bounds                              = false;
	bool shader_storage_image_write_without_format = false;
	bool shader_image_gather_extended              = false;
	bool independent_blend                         = false;
	bool dual_src_blend                            = false;
	bool tessellation_shader                       = false;
	bool shader_int64                              = false;

	// Vulkan 1.2 features.
	bool sampler_mirror_clamp_to_edge = false;
	bool timeline_semaphore           = false;
	bool shader_output_layer          = false;
	bool shader_output_viewport_index = false;
	bool buffer_device_address        = false;
	bool shader_buffer_int64_atomics  = false;
	bool shader_shared_int64_atomics  = false;

	// Vulkan 1.3 features.
	bool robust_image_access = false;
	bool dynamic_rendering   = false;
	bool synchronization2    = false;

	// Features of extensions.
	bool color_write_enable               = false; // VK_EXT_color_write_enable
	bool depth_clip_control               = false; // VK_EXT_depth_clip_control
	bool depth_clip_enable                = false; // VK_EXT_depth_clip_enable
	bool image_view_min_lod               = false; // VK_EXT_image_view_min_lod
	bool fragment_shader_barycentric      = false; // VK_KHR_fragment_shader_barycentric
	bool workgroup_memory_explicit_layout = false; // VK_KHR_workgroup_memory_explicit_layout
};

// Platform policy: which features a device must have. A platform that cannot offer a feature
// says so here, in one place.
struct DeviceRequirements {
	bool color_write_enable = true;
	bool depth_clip_enable  = true;
	bool depth_clamp        = true;

	[[nodiscard]] static DeviceRequirements ForCurrentPlatform();
};

// Optional capabilities. Each one is on only when the device reports it. A guest shader or texture
// that needs a capability that is off stops the emulator with a message that names it.
struct DeviceCapabilities {
	bool image_view_min_lod          = false;
	bool shader_cull_distance        = false;
	bool shader_buffer_int64_atomics = false;
	// Native 64-bit LDS atomics need shaderSharedInt64Atomics and the explicit workgroup layout.
	bool shader_shared_int64_atomics = false;
	bool fragment_shader_barycentric = false;
	bool depth_bounds                = false;
	// Descriptors can be pushed with vkCmdPushDescriptorSet. MoltenVK binds no buffer sizes for
	// them, so a shader that asks for a buffer length reads a null buffer.
	bool push_descriptors = false;
	// Fragment barycentrics interpolated at the centroid. MoltenVK cannot translate them to Metal.
	bool centroid_barycentric = false;
	// A pixel shader input with the raw values of the three vertices of its triangle. MoltenVK
	// cannot translate it to Metal.
	bool per_vertex_attributes = false;
};

struct DeviceDecision {
	// Why the device cannot run the emulator. Empty when it can.
	std::vector<std::string> rejections;
	// Optional capabilities the device lacks, one line each, for the log.
	std::vector<std::string> unavailable;
	DeviceCapabilities       capabilities;

	[[nodiscard]] bool Accepted() const { return rejections.empty(); }
};

[[nodiscard]] DeviceDecision
EvaluateDeviceSuitability(const DeviceFacts& facts, const DeviceRequirements& requirements,
                          const std::vector<std::string>& required_extensions);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_WINDOW_DEVICESUITABILITY_H_ */
