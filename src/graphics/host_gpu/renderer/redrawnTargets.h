#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REDRAWNTARGETS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REDRAWNTARGETS_H_

#include <cstdint>
#include <unordered_map>

namespace Libs::Graphics {

// Which render targets the game draws again in every frame.
//
// A draw whose pipeline is still being built may be left out, so that the frame is not held up.
// That is harmless only when the next frame draws the target again. A target the game draws
// once (a lighting table, a baked shadow map) would keep the hole for as long as the game uses
// it. So a draw may be left out only when its target was also drawn in the frame before.
class RedrawnTargets final {
public:
	// The game showed a frame.
	void NextFrame() {
		++m_frame;
		// Forget targets that have not been drawn for a long time, so that the table stays small.
		if (m_frame % ForgetPeriod == 0) {
			std::erase_if(m_targets, [this](const auto& entry) {
				return m_frame - entry.second.frame > ForgetPeriod;
			});
		}
	}

	// A draw goes to the target at this guest address. True when the frame before this one drew
	// the target too.
	[[nodiscard]] bool Draw(uint64_t target) {
		auto& entry = m_targets[target];
		if (entry.frame != m_frame) {
			entry.redrawn = entry.frame + 1 == m_frame;
			entry.frame   = m_frame;
		}
		return entry.redrawn;
	}

private:
	static constexpr uint64_t ForgetPeriod = 600;

	struct Target {
		uint64_t frame   = UINT64_MAX - 1; // of the last draw; a new target was never drawn
		bool     redrawn = false;
	};

	uint64_t                             m_frame = 0;
	std::unordered_map<uint64_t, Target> m_targets;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REDRAWNTARGETS_H_
