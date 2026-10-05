#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_

#include <cstdint>

namespace Libs::Graphics {

// Optional device capabilities that a compiled shader may rely on. The renderer supplies what the
// device enabled. The default is "everything is available", which is what offline compilation and
// the tests assume. A shader that needs a capability that is off stops with a message that names
// it.
struct ShaderHostFeatures {
	bool buffer_int64_atomics        = true;
	bool shared_int64_atomics        = true;
	bool cull_distance               = true;
	bool fragment_shader_barycentric = true;
	bool float64                     = true;
	// Barycentrics interpolated at the centroid (a multisampled shader that reads them).
	bool centroid_barycentric = true;
	// A pixel shader input that holds the raw values of the three vertices of its triangle
	// (PerVertexKHR). Without it the values come through three more flat inputs, which
	// tessellation shaders made for the draw fill in (see triangleVertexValueShader.h).
	bool per_vertex_attributes = true;
	// A texture view can carry a minimum LOD (VK_EXT_image_view_min_lod). Without it each
	// texture is read through a sampler of its own, which applies the minimum.
	bool image_view_min_lod = true;
	// Stages that may use subgroup operations: VkPhysicalDeviceVulkan11Properties::
	// subgroupSupportedStages, as VkShaderStageFlagBits values. All bits set means every stage.
	uint32_t subgroup_supported_stages = ~0u;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_ */
