#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitProgram(const IR::Program& program, ShaderStageInputInfo input_info,
                                  ShaderHostFeatures host_features = {});

// A host without mesh shaders runs a mesh program as a compute shader (EmitProgram with an emulated
// mesh program). This is the vertex shader that draws what the compute shader wrote: one triangle
// per primitive slot of every workgroup, with the position and parameter outputs read back from
// the records. The pixel shader is unchanged.
struct MeshEmulationVertexProgram {
	std::vector<uint32_t> spirv;
	uint32_t              slot_words = 0;
};
MeshEmulationVertexProgram EmitMeshEmulationVertexProgram(const IR::Program&   program,
                                                          ShaderStageInputInfo input_info);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
