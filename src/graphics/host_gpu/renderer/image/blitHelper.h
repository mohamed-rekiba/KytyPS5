#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <compare>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
struct GraphicContext;

class BlitHelper final {
public:
	inline static constexpr auto ColorToMsDepthLayout =
	    vk::ImageLayout::eDepthStencilAttachmentOptimal;

	BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~BlitHelper();
	KYTY_CLASS_NO_COPY(BlitHelper);

	void ReinterpretColorAsMsDepth(Image& source, Image& destination);
	// Copies sample `sample` of layer `source_layer` of a multisampled depth image into `rect` of
	// a single-sample R32 float color image, at `destination_level` and `destination_layer`.
	void CopyDepthSampleToColor(Image& source, uint32_t source_layer, Image& destination,
	                            uint32_t destination_level, uint32_t destination_layer,
	                            uint32_t sample, const vk::Rect2D& rect);

private:
	struct PipelineKey {
		uint32_t   samples = 1;
		vk::Format format  = vk::Format::eUndefined;

		auto operator<=>(const PipelineKey&) const = default;
	};

	struct Pipeline {
		PipelineKey  key;
		vk::Pipeline handle = nullptr;
	};

	[[nodiscard]] vk::Pipeline GetPipeline(PipelineKey key);
	[[nodiscard]] vk::Pipeline GetDepthSamplePipeline();

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::ShaderModule        m_vertex_shader     = nullptr;
	vk::ShaderModule        m_fragment_shader   = nullptr;
	std::vector<Pipeline>   m_pipelines;
	vk::PipelineLayout      m_depth_sample_layout   = nullptr;
	vk::ShaderModule        m_depth_sample_shader   = nullptr;
	vk::Pipeline            m_depth_sample_pipeline = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
