#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_

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
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERHOSTFEATURES_H_ */
