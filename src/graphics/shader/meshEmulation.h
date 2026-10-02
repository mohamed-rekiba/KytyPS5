#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_MESHEMULATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_MESHEMULATION_H_

#include <cstdint>

// A host without mesh shaders (MoltenVK) runs the guest mesh shader as a compute shader. Each
// workgroup writes one record ("slot") to a buffer, and a generated vertex shader reads the
// records back and feeds the unchanged pixel shader. The compute shader and the vertex shader
// agree on the record layout through this type.
//
// Record layout, in 32-bit words:
//   [0]  vertex count          [1]  primitive count       [2..3]  unused
//   then `vertices` vertices, each `vertex_vec4s` vec4 values (position first, then parameters)
//   then `primitives` primitives, each two words: the three 10-bit vertex indices packed with the
//   cull flag in bit 31, then the layer.

namespace Libs::Graphics {

struct MeshEmulationLayout {
	static constexpr uint32_t kHeaderWords    = 4;
	static constexpr uint32_t kPrimitiveWords = 2;

	uint32_t vertices     = 0; // vertices a workgroup can output
	uint32_t primitives   = 0; // primitives a workgroup can output
	uint32_t vertex_vec4s = 0; // vec4 values per vertex

	[[nodiscard]] constexpr uint32_t VertexWords() const { return vertex_vec4s * 4u; }
	[[nodiscard]] constexpr uint32_t PrimitiveOffsetWords() const {
		return kHeaderWords + vertices * VertexWords();
	}
	[[nodiscard]] constexpr uint32_t SlotWords() const {
		return PrimitiveOffsetWords() + primitives * kPrimitiveWords;
	}
	[[nodiscard]] constexpr uint64_t SlotBytes() const {
		return static_cast<uint64_t>(SlotWords()) * sizeof(uint32_t);
	}
	// The generated vertex shader draws one triangle per primitive slot of every workgroup.
	[[nodiscard]] constexpr uint32_t VerticesPerSlot() const { return primitives * 3u; }
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_MESHEMULATION_H_ */
