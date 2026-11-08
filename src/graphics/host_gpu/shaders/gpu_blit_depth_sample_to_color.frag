#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

// Copies one sample of a multisampled depth image into a single-sample color target.
layout(binding = 0, set = 0) uniform texture2DMS depth;
layout(push_constant) uniform Push {
	uint sample_index;
}
push;
layout(location = 0) out float color;

void main() {
	color = texelFetch(depth, ivec2(gl_FragCoord.xy), int(push.sample_index)).x;
}
