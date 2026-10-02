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
    "VK_KHR_swapchain", "VK_EXT_depth_clip_control", "VK_KHR_push_descriptor",
    "VK_KHR_maintenance1"};

// A device that reports every feature and extension the emulator can use.
DeviceFacts FullFacts() {
	DeviceFacts f;
	f.extensions = {"VK_KHR_swapchain",
	                "VK_EXT_depth_clip_control",
	                "VK_KHR_push_descriptor",
	                "VK_KHR_maintenance1",
	                "VK_EXT_image_view_min_lod",
	                "VK_KHR_fragment_shader_barycentric",
	                "VK_KHR_workgroup_memory_explicit_layout"};
	f.depth_clamp                         = true;
	f.depth_bias_clamp                    = true;
	f.sample_rate_shading                 = true;
	f.shader_clip_distance                = true;
	f.shader_cull_distance                = true;
	f.large_points                        = true;
	f.multi_viewport                      = true;
	f.fill_mode_non_solid                 = true;
	f.fragment_stores_and_atomics         = true;
	f.vertex_pipeline_stores_and_atomics  = true;
	f.sampler_anisotropy                  = true;
	f.robust_buffer_access                = true;
	f.depth_bounds                        = true;
	f.shader_storage_image_write_without_format = true;
	f.shader_image_gather_extended        = true;
	f.independent_blend                   = true;
	f.dual_src_blend                      = true;
	f.tessellation_shader                 = true;
	f.shader_int64                        = true;
	f.sampler_mirror_clamp_to_edge        = true;
	f.timeline_semaphore                  = true;
	f.shader_output_layer                 = true;
	f.shader_output_viewport_index        = true;
	f.buffer_device_address               = true;
	f.shader_buffer_int64_atomics         = true;
	f.shader_shared_int64_atomics         = true;
	f.robust_image_access                 = true;
	f.dynamic_rendering                   = true;
	f.synchronization2                    = true;
	f.color_write_enable                  = true;
	f.depth_clip_control                  = true;
	f.depth_clip_enable                   = true;
	f.image_view_min_lod                  = true;
	f.fragment_shader_barycentric         = true;
	f.workgroup_memory_explicit_layout    = true;
	return f;
}

// What MoltenVK 1.4.2 reports on an Apple M4 Max (measured with a probe).
DeviceFacts MoltenVkFacts() {
	auto f = FullFacts();
	f.extensions                       = {"VK_KHR_swapchain", "VK_EXT_depth_clip_control",
	                                      "VK_KHR_push_descriptor", "VK_KHR_maintenance1",
	                                      "VK_KHR_portability_subset",
	                                      "VK_KHR_fragment_shader_barycentric"};
	f.shader_cull_distance             = false;
	f.depth_bounds                     = false;
	f.shader_buffer_int64_atomics      = false;
	f.shader_shared_int64_atomics      = false;
	f.color_write_enable               = false;
	f.depth_clip_enable                = false;
	f.image_view_min_lod               = false;
	f.workgroup_memory_explicit_layout = false;
	return f;
}

DeviceRequirements AppleRequirements() {
	DeviceRequirements r;
	r.color_write_enable = false;
	r.depth_clip_enable  = false;
	r.depth_clamp        = false;
	return r;
}

const std::vector<std::string> kAppleRequiredExtensions = {
    "VK_KHR_swapchain", "VK_EXT_depth_clip_control", "VK_KHR_push_descriptor",
    "VK_KHR_maintenance1", "VK_KHR_portability_subset"};

void TestMoltenVkLikeDeviceIsAccepted() {
	const auto decision =
	    EvaluateDeviceSuitability(MoltenVkFacts(), AppleRequirements(), kAppleRequiredExtensions);
	Check(decision.Accepted(), "MoltenVK-like device must be accepted");
	Check(!decision.capabilities.image_view_min_lod, "minLod must be off without the extension");
	Check(!decision.capabilities.shader_cull_distance, "cull distance must be off");
	Check(!decision.capabilities.shader_buffer_int64_atomics, "buffer int64 atomics must be off");
	Check(!decision.capabilities.shader_shared_int64_atomics, "shared int64 atomics must be off");
	Check(!decision.capabilities.depth_bounds, "depth bounds must be off");
	Check(decision.capabilities.fragment_shader_barycentric,
	      "barycentrics must be on: MoltenVK offers the extension and the feature");
	Check(Contains(decision.unavailable, "image view minLod"),
	      "an unavailable minLod must be reported");
	Check(Contains(decision.unavailable, "shaderCullDistance"),
	      "an unavailable cull distance must be reported");
	Check(Contains(decision.unavailable, "shaderBufferInt64Atomics"),
	      "unavailable 64-bit buffer atomics must be reported");
}

void TestMissingRequiredFeatureIsRejectedByName() {
	auto facts              = MoltenVkFacts();
	facts.dynamic_rendering = false;
	const auto decision =
	    EvaluateDeviceSuitability(facts, AppleRequirements(), kAppleRequiredExtensions);
	Check(!decision.Accepted(), "a device without dynamicRendering must be rejected");
	Check(Contains(decision.rejections, "dynamicRendering is not supported"),
	      "the rejection must name dynamicRendering");
}

void TestEveryRejectionIsReported() {
	auto facts                 = FullFacts();
	facts.dynamic_rendering    = false;
	facts.timeline_semaphore   = false;
	facts.extensions           = {"VK_KHR_swapchain"};
	const auto decision =
	    EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(Contains(decision.rejections, "dynamicRendering is not supported"),
	      "dynamicRendering must be listed");
	Check(Contains(decision.rejections, "timelineSemaphore is not supported"),
	      "timelineSemaphore must be listed");
	Check(Contains(decision.rejections, "VK_KHR_push_descriptor is not supported"),
	      "a missing required extension must be listed");
}

void TestFullDeviceKeepsEveryCapability() {
	const auto decision =
	    EvaluateDeviceSuitability(FullFacts(), DeviceRequirements {}, kRequiredExtensions);
	Check(decision.Accepted(), "a full-feature device must be accepted");
	Check(decision.unavailable.empty(), "a full-feature device has nothing unavailable");
	const auto& c = decision.capabilities;
	Check(c.image_view_min_lod && c.shader_cull_distance && c.shader_buffer_int64_atomics &&
	          c.shader_shared_int64_atomics && c.fragment_shader_barycentric && c.depth_bounds,
	      "every optional capability must stay enabled when the device has it");
}

void TestIntelLikeDeviceIsAccepted() {
	// Intel HD 630 report: no fragment barycentrics, no depth bounds.
	auto facts                        = FullFacts();
	facts.fragment_shader_barycentric = false;
	facts.depth_bounds                = false;
	const auto decision =
	    EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(decision.Accepted(), "a device lacking only barycentrics and depth bounds is accepted");
	Check(!decision.capabilities.fragment_shader_barycentric, "barycentrics must be off");
	Check(!decision.capabilities.depth_bounds, "depth bounds must be off");
}

void TestPascalLikeDeviceIsAccepted() {
	// GTX 1060 report: no fragment barycentrics.
	auto facts                        = FullFacts();
	facts.fragment_shader_barycentric = false;
	const auto decision =
	    EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(decision.Accepted(), "a device lacking only barycentrics is accepted");
	Check(!decision.capabilities.fragment_shader_barycentric, "barycentrics must be off");
	Check(decision.capabilities.depth_bounds, "depth bounds stay on when the device has them");
}

void TestCapabilityNeedsBothFeatureAndExtension() {
	auto facts                = FullFacts();
	facts.extensions          = kRequiredExtensions; // drops the optional extensions
	auto decision = EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(!decision.capabilities.image_view_min_lod,
	      "minLod feature without the extension must not enable the capability");
	Check(!decision.capabilities.fragment_shader_barycentric,
	      "barycentric feature without the extension must not enable the capability");

	facts                = FullFacts();
	facts.image_view_min_lod = false; // extension listed, feature not reported
	decision = EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(!decision.capabilities.image_view_min_lod,
	      "the extension without the feature must not enable minLod");
}

void TestSharedInt64AtomicsNeedExplicitWorkgroupLayout() {
	auto facts                               = FullFacts();
	facts.workgroup_memory_explicit_layout   = false;
	auto decision = EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(!decision.capabilities.shader_shared_int64_atomics,
	      "native 64-bit LDS atomics need the explicit workgroup layout too");

	facts                                = FullFacts();
	facts.shader_shared_int64_atomics    = false;
	decision = EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(!decision.capabilities.shader_shared_int64_atomics,
	      "native 64-bit LDS atomics need shaderSharedInt64Atomics");
}

void TestPlatformRequirementsAreHonoured() {
	auto facts               = FullFacts();
	facts.color_write_enable = false;
	facts.depth_clip_enable  = false;
	facts.depth_clamp        = false;

	auto decision = EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
	Check(!decision.Accepted(), "the default requirements reject a device without these features");
	Check(Contains(decision.rejections, "colorWriteEnable"), "colorWriteEnable must be named");
	Check(Contains(decision.rejections, "depthClipEnable"), "depthClipEnable must be named");
	Check(Contains(decision.rejections, "depthClamp"), "depthClamp must be named");

	decision = EvaluateDeviceSuitability(facts, AppleRequirements(), kRequiredExtensions);
	Check(decision.Accepted(), "relaxed platform requirements must accept the same device");
}

void TestCreationTimeRequirementsAreSelectionRequirements() {
	// These used to stop the emulator at device creation, after a device was already chosen.
	for (const char* name : {"shaderInt64", "vertexPipelineStoresAndAtomics", "dualSrcBlend"}) {
		auto facts = FullFacts();
		if (std::string(name) == "shaderInt64") {
			facts.shader_int64 = false;
		} else if (std::string(name) == "vertexPipelineStoresAndAtomics") {
			facts.vertex_pipeline_stores_and_atomics = false;
		} else {
			facts.dual_src_blend = false;
		}
		const auto decision =
		    EvaluateDeviceSuitability(facts, DeviceRequirements {}, kRequiredExtensions);
		Check(!decision.Accepted(), "a missing creation-time feature must reject the device");
		Check(Contains(decision.rejections, name), "the rejection must name the feature");
	}
}

} // namespace

int main() {
	TestMoltenVkLikeDeviceIsAccepted();
	TestMissingRequiredFeatureIsRejectedByName();
	TestEveryRejectionIsReported();
	TestFullDeviceKeepsEveryCapability();
	TestIntelLikeDeviceIsAccepted();
	TestPascalLikeDeviceIsAccepted();
	TestCapabilityNeedsBothFeatureAndExtension();
	TestSharedInt64AtomicsNeedExplicitWorkgroupLayout();
	TestPlatformRequirementsAreHonoured();
	TestCreationTimeRequirementsAreSelectionRequirements();
	if (g_failures != 0) {
		std::cerr << "DeviceSuitabilityTests: " << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "DeviceSuitabilityTests: all cases passed\n";
	return 0;
}
