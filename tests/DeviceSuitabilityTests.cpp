// Tests for the pure device-suitability decision: what a Vulkan device reports goes in,
// accept/reject reasons and the optional capabilities that are enabled come out.
// No Vulkan device is needed.

#include "graphics/presentation/window/deviceSuitability.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

bool Contains(const std::vector<std::string>& lines, const std::string& needle) {
	return std::any_of(lines.begin(), lines.end(), [&](const std::string& line) {
		return line.find(needle) != std::string::npos;
	});
}

const std::vector<std::string> kRequiredExtensions = {
    "VK_KHR_swapchain", "VK_EXT_depth_clip_control", "VK_KHR_maintenance1"};

// A device that reports every feature and extension the emulator can use.
DeviceFacts FullFacts() {
	DeviceFacts f;
	f.extensions                                      = {"VK_KHR_swapchain",
	                                                     "VK_EXT_depth_clip_control",
	                                                     "VK_KHR_push_descriptor",
	                                                     "VK_KHR_maintenance1",
	                                                     "VK_EXT_image_view_min_lod",
	                                                     "VK_KHR_fragment_shader_barycentric",
	                                                     "VK_KHR_workgroup_memory_explicit_layout",
	                                                     "VK_EXT_color_write_enable",
	                                                     "VK_EXT_depth_clip_enable",
	                                                     "VK_EXT_shader_image_atomic_int64",
	                                                     "VK_EXT_mesh_shader",
	                                                     "VK_KHR_compute_shader_derivatives"};
	f.subgroup_supported_stages                       = ~0u;
	f.subgroup_supported_operations                   = ~0u;
	f.depth_clamp                                     = true;
	f.depth_bias_clamp                                = true;
	f.sample_rate_shading                             = true;
	f.shader_clip_distance                            = true;
	f.shader_cull_distance                            = true;
	f.large_points                                    = true;
	f.multi_viewport                                  = true;
	f.fill_mode_non_solid                             = true;
	f.fragment_stores_and_atomics                     = true;
	f.vertex_pipeline_stores_and_atomics              = true;
	f.sampler_anisotropy                              = true;
	f.robust_buffer_access                            = true;
	f.depth_bounds                                    = true;
	f.shader_storage_image_write_without_format       = true;
	f.shader_image_gather_extended                    = true;
	f.independent_blend                               = true;
	f.dual_src_blend                                  = true;
	f.tessellation_shader                             = true;
	f.shader_int64                                    = true;
	f.shader_float64                                  = true;
	f.storage_buffer_16bit_access                     = true;
	f.sampler_mirror_clamp_to_edge                    = true;
	f.timeline_semaphore                              = true;
	f.shader_output_layer                             = true;
	f.shader_output_viewport_index                    = true;
	f.buffer_device_address                           = true;
	f.storage_buffer_8bit_access                      = true;
	f.shader_sampled_image_array_non_uniform_indexing = true;
	f.shader_buffer_int64_atomics                     = true;
	f.shader_shared_int64_atomics                     = true;
	f.float64_preserves_special_values                = true;
	f.robust_image_access                             = true;
	f.dynamic_rendering                               = true;
	f.synchronization2                                = true;
	f.color_write_enable                              = true;
	f.depth_clip_control                              = true;
	f.depth_clip_enable                               = true;
	f.image_view_min_lod                              = true;
	f.fragment_shader_barycentric                     = true;
	f.workgroup_memory_explicit_layout                = true;
	f.shader_image_int64_atomics                      = true;
	f.mesh_shader                                     = true;
	f.compute_derivative_group_quads                  = true;
	return f;
}

// A report like the one of MoltenVK 1.4.2 on an Apple M4 Max. Measured there: the device has
// every mandatory feature and lacks minLod, shaderBufferInt64Atomics and shaderCullDistance. The
// other missing features are set as its documentation states them.
DeviceFacts MoltenVkFacts() {
	auto f       = FullFacts();
	f.driver     = HostDriver::MoltenVk;
	f.extensions = {"VK_KHR_swapchain",          "VK_EXT_depth_clip_control",
	                "VK_KHR_push_descriptor",    "VK_KHR_maintenance1",
	                "VK_KHR_portability_subset", "VK_KHR_fragment_shader_barycentric"};
	f.subgroup_supported_stages        = 0x20 | 0x10; // compute and fragment
	f.shader_cull_distance             = false;
	f.depth_bounds                     = false;
	f.shader_float64                   = false;
	f.shader_buffer_int64_atomics      = false;
	f.shader_shared_int64_atomics      = false;
	f.color_write_enable               = false;
	f.depth_clip_enable                = false;
	f.image_view_min_lod               = false;
	f.workgroup_memory_explicit_layout = false;
	f.shader_image_int64_atomics       = false;
	f.mesh_shader                      = false;
	f.compute_derivative_group_quads   = false;
	return f;
}

void TestMoltenVkLikeDeviceIsAccepted() {
	const auto decision = EvaluateDeviceSuitability(MoltenVkFacts(), kRequiredExtensions);
	Check(decision.Accepted(), "MoltenVK-like device must be accepted");
	const auto& c = decision.capabilities;
	Check(!c.image_view_min_lod, "minLod must be off without the extension");
	Check(!c.cull_distance, "cull distance must be off");
	Check(!c.buffer_int64_atomics, "buffer int64 atomics must be off");
	Check(!c.shared_int64_atomics, "shared int64 atomics must be off");
	Check(!c.image_int64_atomics, "image int64 atomics must be off");
	Check(!c.depth_bounds, "depth bounds must be off");
	Check(!c.depth_clip_enable && !c.color_write_enable && !c.mesh_shader && !c.float64 &&
	          !c.compute_derivatives,
	      "capabilities the device does not report must be off");
	Check(c.depth_clamp, "depth clamp must be on: the device reports it");
	Check(c.fragment_barycentric,
	      "barycentrics must be on: MoltenVK offers the extension and the feature");
	Check(c.push_descriptors, "push descriptors must be on: the device has the extension");
	Check(c.subgroup_supported_stages == (0x20u | 0x10u), "subgroup stages must be passed on");
	Check(c.subgroup_supported_operations == ~0u, "subgroup operations must be passed on");
	Check(Contains(decision.unavailable, "image view minLod"),
	      "an unavailable minLod must be reported");
	Check(Contains(decision.unavailable, "shaderCullDistance"),
	      "an unavailable cull distance must be reported");
	Check(Contains(decision.unavailable, "shaderBufferInt64Atomics"),
	      "unavailable 64-bit buffer atomics must be reported");
}

void TestDriverDefectsAreNotCapabilities() {
	const auto moltenvk = EvaluateDeviceSuitability(MoltenVkFacts(), kRequiredExtensions);
	Check(moltenvk.faults.pushed_buffers_have_no_size && moltenvk.faults.no_per_vertex_inputs &&
	          moltenvk.faults.no_centroid_barycentric,
	      "MoltenVK's defects must be reported");
	Check(Contains(moltenvk.unavailable, "driver defect"), "a driver defect must reach the log");

	// The same report from another driver: same capabilities, no defects.
	auto facts       = MoltenVkFacts();
	facts.driver     = HostDriver::Other;
	const auto other = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(other.capabilities == moltenvk.capabilities,
	      "capabilities must not depend on the driver's name");
	Check(other.faults == DriverFaults {}, "an unknown driver has no known defects");
}

void TestMissingRequiredFeatureIsRejectedByName() {
	auto facts              = MoltenVkFacts();
	facts.dynamic_rendering = false;
	const auto decision     = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(!decision.Accepted(), "a device without dynamicRendering must be rejected");
	Check(Contains(decision.rejections, "dynamicRendering is not supported"),
	      "the rejection must name dynamicRendering");
}

void TestEveryRejectionIsReported() {
	auto facts               = FullFacts();
	facts.dynamic_rendering  = false;
	facts.timeline_semaphore = false;
	facts.extensions         = {"VK_KHR_swapchain"};
	const auto decision      = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(Contains(decision.rejections, "dynamicRendering is not supported"),
	      "dynamicRendering must be listed");
	Check(Contains(decision.rejections, "timelineSemaphore is not supported"),
	      "timelineSemaphore must be listed");
	Check(Contains(decision.rejections, "VK_KHR_maintenance1 is not supported"),
	      "a missing required extension must be listed");
}

void TestFullDeviceKeepsEveryCapability() {
	const auto decision = EvaluateDeviceSuitability(FullFacts(), kRequiredExtensions);
	Check(decision.Accepted(), "a full-feature device must be accepted");
	Check(decision.unavailable.empty(), "a full-feature device has nothing unavailable");
	Check(decision.capabilities == HostCapabilities::Full(),
	      "every optional capability must stay enabled when the device has it");
	Check(decision.faults == DriverFaults {}, "a full-feature device has no defects");
}

void TestOptionalFeaturesDoNotRejectADevice() {
	// Intel HD 630 report: no fragment barycentrics, no depth bounds.
	auto facts                        = FullFacts();
	facts.fragment_shader_barycentric = false;
	facts.depth_bounds                = false;
	auto decision                     = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(decision.Accepted(), "a device lacking only barycentrics and depth bounds is accepted");
	Check(!decision.capabilities.fragment_barycentric, "barycentrics must be off");
	Check(!decision.capabilities.depth_bounds, "depth bounds must be off");

	facts                    = FullFacts();
	facts.color_write_enable = false;
	facts.depth_clip_enable  = false;
	facts.depth_clamp        = false;
	decision                 = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(decision.Accepted(), "colour write, depth clip and depth clamp are optional");
	Check(!decision.capabilities.color_write_enable && !decision.capabilities.depth_clip_enable &&
	          !decision.capabilities.depth_clamp,
	      "the three capabilities must be off");
}

void TestCapabilityNeedsBothFeatureAndExtension() {
	auto facts           = FullFacts();
	facts.extensions     = kRequiredExtensions; // drops the optional extensions
	auto        decision = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	const auto& c        = decision.capabilities;
	Check(!c.image_view_min_lod,
	      "minLod feature without the extension must not enable the capability");
	Check(!c.fragment_barycentric,
	      "barycentric feature without the extension must not enable the capability");
	Check(!c.color_write_enable && !c.depth_clip_enable && !c.image_int64_atomics &&
	          !c.mesh_shader && !c.push_descriptors && !c.compute_derivatives,
	      "no extension, no capability");

	facts                    = FullFacts();
	facts.image_view_min_lod = false; // extension listed, feature not reported
	decision                 = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(!decision.capabilities.image_view_min_lod,
	      "the extension without the feature must not enable minLod");
}

void TestSharedInt64AtomicsNeedExplicitWorkgroupLayout() {
	auto facts                             = FullFacts();
	facts.workgroup_memory_explicit_layout = false;
	auto decision                          = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(!decision.capabilities.shared_int64_atomics,
	      "native 64-bit LDS atomics need the explicit workgroup layout too");

	facts                             = FullFacts();
	facts.shader_shared_int64_atomics = false;
	decision                          = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(!decision.capabilities.shared_int64_atomics,
	      "native 64-bit LDS atomics need shaderSharedInt64Atomics");
}

void TestFloat64NeedsSpecialValues() {
	auto facts                             = FullFacts();
	facts.float64_preserves_special_values = false;
	const auto decision                    = EvaluateDeviceSuitability(facts, kRequiredExtensions);
	Check(decision.Accepted(), "Float64 is optional");
	Check(!decision.capabilities.float64,
	      "Float64 that loses signed zero, infinity or NaN must not count");
}

void TestEveryMandatoryFeatureRejectsByName() {
	struct Case {
		bool DeviceFacts::* field;
		const char*         name;
	};
	const Case cases[] = {
	    {&DeviceFacts::shader_int64, "shaderInt64"},
	    {&DeviceFacts::vertex_pipeline_stores_and_atomics, "vertexPipelineStoresAndAtomics"},
	    {&DeviceFacts::dual_src_blend, "dualSrcBlend"},
	    {&DeviceFacts::storage_buffer_16bit_access, "storageBuffer16BitAccess"},
	    {&DeviceFacts::storage_buffer_8bit_access, "storageBuffer8BitAccess"},
	    {&DeviceFacts::shader_sampled_image_array_non_uniform_indexing,
	     "shaderSampledImageArrayNonUniformIndexing"},
	    {&DeviceFacts::depth_clip_control, "depthClipControl"},
	    {&DeviceFacts::buffer_device_address, "bufferDeviceAddress"},
	    {&DeviceFacts::tessellation_shader, "tessellationShader"},
	};
	for (const auto& test: cases) {
		auto facts          = FullFacts();
		facts.*(test.field) = false;
		const auto decision = EvaluateDeviceSuitability(facts, kRequiredExtensions);
		Check(!decision.Accepted(), "a missing mandatory feature must reject the device");
		Check(Contains(decision.rejections, test.name), "the rejection must name the feature");
	}
}

} // namespace

int main() {
	TestMoltenVkLikeDeviceIsAccepted();
	TestDriverDefectsAreNotCapabilities();
	TestMissingRequiredFeatureIsRejectedByName();
	TestEveryRejectionIsReported();
	TestFullDeviceKeepsEveryCapability();
	TestOptionalFeaturesDoNotRejectADevice();
	TestCapabilityNeedsBothFeatureAndExtension();
	TestSharedInt64AtomicsNeedExplicitWorkgroupLayout();
	TestFloat64NeedsSpecialValues();
	TestEveryMandatoryFeatureRejectsByName();
	if (g_failures != 0) {
		std::cerr << "DeviceSuitabilityTests: " << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "DeviceSuitabilityTests: all cases passed\n";
	return 0;
}
