#include "graphics/presentation/window/deviceSuitability.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

constexpr const char* kMinLodExtension          = "VK_EXT_image_view_min_lod";
constexpr const char* kBarycentricExtension     = "VK_KHR_fragment_shader_barycentric";
constexpr const char* kWorkgroupLayoutExtension = "VK_KHR_workgroup_memory_explicit_layout";
constexpr const char* kPushDescriptorExtension  = "VK_KHR_push_descriptor";

bool HasExtension(const std::vector<std::string>& extensions, const std::string& name) {
	return std::find(extensions.begin(), extensions.end(), name) != extensions.end();
}

} // namespace

DeviceRequirements DeviceRequirements::ForCurrentPlatform() {
	DeviceRequirements requirements;
#if defined(__APPLE__)
	// MoltenVK lacks VK_EXT_color_write_enable and VK_EXT_depth_clip_enable, and the renderer
	// falls back to static color-write masks and default depth clipping there. It does not
	// require depthClamp either.
	requirements.color_write_enable = false;
	requirements.depth_clip_enable  = false;
	requirements.depth_clamp        = false;
#endif
	return requirements;
}

DeviceDecision EvaluateDeviceSuitability(const DeviceFacts&              facts,
                                         const DeviceRequirements&       requirements,
                                         const std::vector<std::string>& required_extensions) {
	DeviceDecision decision;

	const auto require = [&](bool supported, const char* name) {
		if (!supported) {
			decision.rejections.push_back(std::string(name) + " is not supported");
		}
	};

	require(facts.depth_clip_control, "depthClipControl");
	if (requirements.color_write_enable) {
		require(facts.color_write_enable, "colorWriteEnable");
	}
	if (requirements.depth_clip_enable) {
		require(facts.depth_clip_enable, "depthClipEnable");
	}
	if (requirements.depth_clamp) {
		require(facts.depth_clamp, "depthClamp");
	}

	require(facts.sampler_mirror_clamp_to_edge, "samplerMirrorClampToEdge");
	require(facts.timeline_semaphore, "timelineSemaphore");
	require(facts.shader_output_layer, "shaderOutputLayer");
	require(facts.shader_output_viewport_index, "shaderOutputViewportIndex");
	require(facts.buffer_device_address, "bufferDeviceAddress");
	require(facts.robust_image_access, "robustImageAccess");
	require(facts.dynamic_rendering, "dynamicRendering");
	require(facts.synchronization2, "synchronization2");
	require(facts.sample_rate_shading, "sampleRateShading");
	require(facts.depth_bias_clamp, "depthBiasClamp");
	require(facts.shader_clip_distance, "shaderClipDistance");
	require(facts.large_points, "largePoints");
	require(facts.multi_viewport, "multiViewport");
	require(facts.fill_mode_non_solid, "fillModeNonSolid");
	require(facts.fragment_stores_and_atomics, "fragmentStoresAndAtomics");
	require(facts.sampler_anisotropy, "samplerAnisotropy");
	require(facts.robust_buffer_access, "robustBufferAccess");
	require(facts.shader_storage_image_write_without_format,
	        "shaderStorageImageWriteWithoutFormat");
	require(facts.shader_image_gather_extended, "shaderImageGatherExtended");
	require(facts.independent_blend, "independentBlend");
	require(facts.tessellation_shader, "tessellationShader");
	// Device creation needs these. Checking them here rejects a device before it is chosen.
	require(facts.shader_int64, "shaderInt64");
	require(facts.vertex_pipeline_stores_and_atomics, "vertexPipelineStoresAndAtomics");
	require(facts.dual_src_blend, "dualSrcBlend");

	for (const auto& extension: required_extensions) {
		if (!HasExtension(facts.extensions, extension)) {
			decision.rejections.push_back(extension + " is not supported");
		}
	}

	// Optional capabilities: on only when the device reports the feature (and the extension that
	// carries it).
	auto& caps = decision.capabilities;
	caps.image_view_min_lod =
	    facts.image_view_min_lod && HasExtension(facts.extensions, kMinLodExtension);
	caps.shader_cull_distance        = facts.shader_cull_distance;
	caps.shader_buffer_int64_atomics = facts.shader_buffer_int64_atomics;
	caps.shader_shared_int64_atomics = facts.shader_shared_int64_atomics &&
	                                   facts.workgroup_memory_explicit_layout &&
	                                   HasExtension(facts.extensions, kWorkgroupLayoutExtension);
	caps.fragment_shader_barycentric =
	    facts.fragment_shader_barycentric && HasExtension(facts.extensions, kBarycentricExtension);
	caps.depth_bounds = facts.depth_bounds;
	caps.push_descriptors =
	    HasExtension(facts.extensions, kPushDescriptorExtension) && !facts.driver_is_moltenvk;
	caps.centroid_barycentric = caps.fragment_shader_barycentric && !facts.driver_is_moltenvk;

	const auto note = [&](bool available, const char* line) {
		if (!available) {
			decision.unavailable.push_back(line);
		}
	};
	note(caps.image_view_min_lod,
	     "image view minLod is unavailable: a texture with a non-zero minimum LOD stops the "
	     "emulator");
	note(caps.shader_cull_distance,
	     "shaderCullDistance is unavailable: a shader that exports cull distances stops the "
	     "emulator");
	note(caps.shader_buffer_int64_atomics,
	     "shaderBufferInt64Atomics is unavailable: a shader with 64-bit buffer atomics stops the "
	     "emulator");
	note(caps.shader_shared_int64_atomics,
	     "64-bit LDS atomics are unavailable: a shader that uses them stops the emulator");
	note(caps.fragment_shader_barycentric,
	     "fragmentShaderBarycentric is unavailable: a shader that reads barycentrics or raw vertex "
	     "attributes stops the emulator");
	note(caps.depth_bounds,
	     "depthBounds is unavailable: a draw that enables the depth bounds test stops the "
	     "emulator");
	// With the extension present but the driver unable to interpolate at the centroid.
	if (caps.fragment_shader_barycentric && !caps.centroid_barycentric) {
		decision.unavailable.push_back("centroid barycentrics are unavailable: a multisampled "
		                               "shader that reads them stops the "
		                               "emulator");
	}
	// Not an error: the renderer uses regular descriptor sets instead.
	note(caps.push_descriptors,
	     "push descriptors are not used: MoltenVK binds no buffer sizes for them, so the "
	     "renderer uses regular descriptor sets");

	return decision;
}

} // namespace Libs::Graphics
