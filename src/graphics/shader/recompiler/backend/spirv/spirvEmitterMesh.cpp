#include "common/assert.h"
#include "graphics/shader/meshEmulation.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t MeshArray(EmitterState& state, spv::StorageClass storage, uint32_t type, uint32_t count) {
	const auto array = state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	return state.builder.DefineGlobalVariable(TypePointer(state, storage, array), storage);
}

uint32_t MeshElement(EmitterState& state, uint32_t variable, spv::StorageClass storage,
                     uint32_t type, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, type), pointer,
	                          variable, index);
	return pointer;
}

uint32_t MeshLoad(EmitterState& state, uint32_t variable, spv::StorageClass storage, uint32_t type,
                  uint32_t index) {
	const auto pointer = MeshElement(state, variable, storage, type, index);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, value, pointer);
	return value;
}

uint32_t MeshOutputType(EmitterState& state, IR::StageOutputKind kind) {
	return kind == IR::StageOutputKind::Layer ? TypeU32(state) : TypeF32Vector(state, 4);
}

} // namespace

void DefineMeshOutputs(EmitterState& state) {
	const auto& mesh     = state.input_info.vertex->mesh;
	const bool  emulated = MeshEmulated(state);
	for (auto& output: state.outputs) {
		if (output.kind != IR::StageOutputKind::Position &&
		    output.kind != IR::StageOutputKind::Parameter &&
		    output.kind != IR::StageOutputKind::Layer) {
			EXIT("unsupported mesh output kind=%u\n", static_cast<uint32_t>(output.kind));
		}
		const auto type = MeshOutputType(state, output.kind);
		// Only Layer is read by another invocation, through the primitive's provoking vertex.
		const bool shared = output.kind == IR::StageOutputKind::Layer;
		output.mesh_data_variable =
		    MeshArray(state, shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, type,
		              shared ? mesh.max_vertices : state.lane_count);
		if (emulated) {
			// The outputs go to the record buffer instead of host mesh output arrays.
			continue;
		}
		output.variable_id = MeshArray(
		    state, spv::StorageClassOutput, type,
		    output.kind == IR::StageOutputKind::Layer ? mesh.max_primitives : mesh.max_vertices);
		state.interface_variables.push_back(output.variable_id);
		state.builder.AddName(output.variable_id, output.debug_name.c_str());
		if (output.kind == IR::StageOutputKind::Parameter) {
			state.builder.AddAnnotation(spv::OpDecorate, output.variable_id,
			                            spv::DecorationLocation, output.location);
		} else {
			state.builder.AddAnnotation(spv::OpDecorate, output.variable_id, spv::DecorationBuiltIn,
			                            output.kind == IR::StageOutputKind::Layer
			                                ? spv::BuiltInLayer
			                                : spv::BuiltInPosition);
		}
		if (output.kind == IR::StageOutputKind::Layer) {
			state.builder.AddAnnotation(spv::OpDecorate, output.variable_id,
			                            spv::DecorationPerPrimitiveEXT); // PerPrimitiveEXT
		}
	}
	state.mesh_allocation = MeshArray(state, spv::StorageClassWorkgroup, TypeU32(state), 2);
	state.mesh_primitive_data =
	    MeshArray(state, spv::StorageClassPrivate, TypeU32(state), state.lane_count);
	if (emulated) {
		return;
	}
	state.mesh_primitives =
	    MeshArray(state, spv::StorageClassOutput, TypeU32Vector(state, 3), mesh.max_primitives);
	state.mesh_cull =
	    MeshArray(state, spv::StorageClassOutput, TypeBool(state), mesh.max_primitives);
	state.interface_variables.push_back(state.mesh_primitives);
	state.interface_variables.push_back(state.mesh_cull);
	state.builder.AddAnnotation(
	    spv::OpDecorate, state.mesh_primitives, spv::DecorationBuiltIn,
	    spv::BuiltInPrimitiveTriangleIndicesEXT); // PrimitiveTriangleIndicesEXT
	state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull, spv::DecorationBuiltIn,
	                            spv::BuiltInCullPrimitiveEXT); // CullPrimitiveEXT
	state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull, spv::DecorationPerPrimitiveEXT);
}

uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	const auto output = std::ranges::find_if(state.outputs, [=](const OutputBinding& binding) {
		return binding.kind == kind && binding.index == index;
	});
	if (output == state.outputs.end()) {
		EXIT("mesh export has no output binding: kind=%u index=%u\n", static_cast<uint32_t>(kind),
		     index);
	}
	const bool shared = kind == IR::StageOutputKind::Layer;
	return MeshElement(
	    state, output->mesh_data_variable,
	    shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, MeshOutputType(state, kind),
	    shared ? EmitLocalInvocationIndex(state) : ConstantU32(state, state.lane_half));
}

uint32_t MeshPrimitivePointer(EmitterState& state) {
	return MeshElement(state, state.mesh_primitive_data, spv::StorageClassPrivate, TypeU32(state),
	                   ConstantU32(state, state.lane_half));
}

void EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto first = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first,
	                          EmitLocalInvocationIndex(state), ConstantU32(state, 0));
	EmitIfCondition(state, first, [&] {
		const auto allocation = ctx.Arg(inst, 0);
		for (uint32_t field = 0; field < 2; field++) {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), value, allocation,
			                          ConstantU32(state, field * 12u),
			                          ConstantU32(state, field == 0 ? 10u : 11u));
			const auto pointer =
			    MeshElement(state, state.mesh_allocation, spv::StorageClassWorkgroup,
			                TypeU32(state), ConstantU32(state, field));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		}
	});
}

void EmitMeshEntryPoint(EmitterState& state) {
	if (MeshEmulated(state)) {
		EmitMeshComputeEntryPoint(state);
		return;
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.mesh_guest_func);
	// All guest waves finish before the uniform Vulkan allocation and output stores.
	EmitBarrier(state);
	const auto vertices   = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 0));
	const auto primitives = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpSetMeshOutputsEXT, vertices,
	                          primitives); // OpSetMeshOutputsEXT
	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.lane_half      = half;
		const auto index     = EmitLocalInvocationIndex(state);
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			for (const auto& output: state.outputs) {
				if (output.kind == IR::StageOutputKind::Layer) {
					continue;
				}
				const auto type  = MeshOutputType(state, output.kind);
				const auto value =
				    MeshLoad(state, output.mesh_data_variable, spv::StorageClassPrivate, type,
				             ConstantU32(state, half));
				const auto pointer =
				    MeshElement(state, output.variable_id, spv::StorageClassOutput, type, index);
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
		                          primitives);
		EmitIfCondition(state, is_primitive, [&] {
			const auto packed = MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			                             TypeU32(state), ConstantU32(state, half));
			uint32_t   vertex[3] {};
			for (uint32_t component = 0; component < 3; component++) {
				vertex[component] = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpBitFieldUExtract, TypeU32(state), vertex[component], packed,
				    ConstantU32(state, component * 10u), ConstantU32(state, 10));
			}
			const auto triangle = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 3), triangle,
			                          vertex[0], vertex[1], vertex[2]);
			const auto triangle_pointer =
			    MeshElement(state, state.mesh_primitives, spv::StorageClassOutput,
			                TypeU32Vector(state, 3), index);
			state.builder.AddFunction(spv::OpStore, triangle_pointer, triangle);
			const auto null_bit =
			    EmitBinaryU32(state, spv::OpBitwiseAnd, packed, ConstantU32(state, 0x80000000u));
			const auto culled = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), culled, null_bit,
			                          ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore,
			                          MeshElement(state, state.mesh_cull, spv::StorageClassOutput,
			                                      TypeBool(state), index),
			                          culled);
			for (const auto& output: state.outputs) {
				if (output.kind != IR::StageOutputKind::Layer) {
					continue;
				}
				const auto layer = MeshLoad(state, output.mesh_data_variable,
				                            spv::StorageClassWorkgroup, TypeU32(state),
				                            vertex[state.input_info.vertex->mesh.provoking_vertex]);
				const auto pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput,
				                                 TypeU32(state), index);
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
		});
	}
	state.lane_half = 0;
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

namespace {

MeshEmulationLayout MeshLayout(const EmitterState& state) {
	const auto& mesh  = state.input_info.vertex->mesh;
	uint32_t    vec4s = 0;
	for (const auto& output: state.outputs) {
		if (output.kind != IR::StageOutputKind::Layer) {
			vec4s++;
		}
	}
	return {mesh.max_vertices, mesh.max_primitives, vec4s};
}

uint32_t MeshPushDword(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, index));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

// Stores one 32-bit word of this workgroup's record. `word` is the word index within the record.
void StoreRecordWord(EmitterState& state, uint32_t record, uint32_t word, uint32_t value) {
	const auto offset  = Binary(state, spv::OpIMul, TypeScalarU64(state),
	                            Unary(state, spv::OpUConvert, TypeScalarU64(state), word),
	                            ConstantDeviceAddress(state, sizeof(uint32_t)));
	const auto address = Binary(state, spv::OpIAdd, TypeScalarU64(state), record, offset);
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
	                          address);
	constexpr uint32_t alignment = sizeof(uint32_t);
	state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask,
	                          alignment);
}

uint32_t WordIndex(EmitterState& state, uint32_t base, uint32_t index, uint32_t stride) {
	return EmitAddU32(state, ConstantU32(state, base),
	                  EmitBinaryU32(state, spv::OpIMul, index, ConstantU32(state, stride)));
}

} // namespace

// The emulated mesh shader: the guest code runs as usual, then each invocation writes its vertex
// and primitive to this workgroup's record in the buffer the draw provides (meshEmulation.h).
void EmitMeshComputeEntryPoint(EmitterState& state) {
	const auto layout = MeshLayout(state);
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.mesh_guest_func);
	// All guest waves finish before the uniform allocation and output stores.
	EmitBarrier(state);
	const auto vertices   = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 0));
	const auto primitives = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 1));

	const auto group    = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
	const auto instance = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
	const auto groups   = MeshPushDword(state, IR::PushData::MeshEmulationGroupsDword);
	const auto slot = EmitAddU32(state, group, EmitBinaryU32(state, spv::OpIMul, instance, groups));
	const auto buffer =
	    DeviceAddressFromWords(state, MeshPushDword(state, IR::PushData::MeshEmulationAddressDword),
	                           MeshPushDword(state, IR::PushData::MeshEmulationAddressDword + 1));
	const auto record = Binary(state, spv::OpIAdd, TypeScalarU64(state), buffer,
	                           Binary(state, spv::OpIMul, TypeScalarU64(state),
	                                  Unary(state, spv::OpUConvert, TypeScalarU64(state), slot),
	                                  ConstantDeviceAddress(state, layout.SlotBytes())));

	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.lane_half  = half;
		const auto index = EmitLocalInvocationIndex(state);
		if (half == 0) {
			const auto first = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first, index,
			                          ConstantU32(state, 0));
			EmitIfCondition(state, first, [&] {
				StoreRecordWord(state, record, ConstantU32(state, 0), vertices);
				StoreRecordWord(state, record, ConstantU32(state, 1), primitives);
			});
		}
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			uint32_t vec4 = 0;
			for (const auto& output: state.outputs) {
				if (output.kind == IR::StageOutputKind::Layer) {
					continue;
				}
				const auto value =
				    MeshLoad(state, output.mesh_data_variable, spv::StorageClassPrivate,
				             MeshOutputType(state, output.kind), ConstantU32(state, half));
				for (uint32_t component = 0; component < 4; component++) {
					const auto lane = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), lane, value,
					                          component);
					StoreRecordWord(
					    state, record,
					    WordIndex(state, MeshEmulationLayout::kHeaderWords + vec4 * 4u + component,
					              index, layout.VertexWords()),
					    Unary(state, spv::OpBitcast, TypeU32(state), lane));
				}
				vec4++;
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
		                          primitives);
		EmitIfCondition(state, is_primitive, [&] {
			const auto packed = MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			                             TypeU32(state), ConstantU32(state, half));
			const auto word   = WordIndex(state, layout.PrimitiveOffsetWords(), index,
			                              MeshEmulationLayout::kPrimitiveWords);
			StoreRecordWord(state, record, word, packed);
			// The layer of a primitive is the one at its provoking vertex.
			uint32_t layer = ConstantU32(state, 0);
			for (const auto& output: state.outputs) {
				if (output.kind != IR::StageOutputKind::Layer) {
					continue;
				}
				const auto provoking = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpBitFieldUExtract, TypeU32(state), provoking, packed,
				    ConstantU32(state, state.input_info.vertex->mesh.provoking_vertex * 10u),
				    ConstantU32(state, 10));
				layer = MeshLoad(state, output.mesh_data_variable, spv::StorageClassWorkgroup,
				                 TypeU32(state), provoking);
			}
			StoreRecordWord(state, record, EmitAddU32(state, word, ConstantU32(state, 1)), layer);
		});
	}
	state.lane_half = 0;
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler::Spirv {

MeshEmulationVertexProgram EmitMeshEmulationVertexProgram(const IR::Program&   program,
                                                          ShaderStageInputInfo input_info) {
	const auto& mesh = input_info.vertex->mesh;
	struct Vec4Output {
		IR::StageOutputKind kind;
		uint32_t            location;
	};
	std::vector<Vec4Output> vec4s;
	bool                    has_layer    = false;
	bool                    has_position = false;
	for (const auto& output: program.info.outputs) {
		if (output.kind == IR::StageOutputKind::Layer) {
			has_layer = true;
			continue;
		}
		if (output.kind != IR::StageOutputKind::Position &&
		    output.kind != IR::StageOutputKind::Parameter) {
			EXIT("unsupported mesh output kind=%u\n", static_cast<uint32_t>(output.kind));
		}
		has_position |= output.kind == IR::StageOutputKind::Position;
		vec4s.push_back({output.kind, output.location});
	}
	EXIT_IF(!has_position);
	const MeshEmulationLayout layout {mesh.max_vertices, mesh.max_primitives,
	                                  static_cast<uint32_t>(vec4s.size())};

	Builder b(0x00010300u);
	if (has_layer) {
		b.RequireVersion(0x00010500u);
		b.RequireCapability(spv::CapabilityShaderLayer);
	}
	b.RequireCapability(spv::CapabilityShader);
	b.RequireCapability(spv::CapabilityInt64);
	b.RequireCapability(spv::CapabilityPhysicalStorageBufferAddresses);
	b.RequireExtension("SPV_KHR_physical_storage_buffer");
	b.AddMemoryModel(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);

	const auto t_void  = b.Type(spv::OpTypeVoid);
	const auto t_fn    = b.Type(spv::OpTypeFunction, t_void);
	const auto t_bool  = b.Type(spv::OpTypeBool);
	const auto t_u32   = b.Type(spv::OpTypeInt, 32, 0);
	const auto t_i32   = b.Type(spv::OpTypeInt, 32, 1);
	const auto t_u64   = b.Type(spv::OpTypeInt, 64, 0);
	const auto t_f32   = b.Type(spv::OpTypeFloat, 32);
	const auto t_v4    = b.Type(spv::OpTypeVector, t_f32, 4);
	const auto t_v2u   = b.Type(spv::OpTypeVector, t_u32, 2);
	const auto pointer = [&](spv::StorageClass storage, uint32_t type) {
		return b.Type(spv::OpTypePointer, storage, type);
	};
	const auto u32c = [&](uint32_t value) { return b.Constant(spv::OpConstant, t_u32, value); };
	const auto u64c = [&](uint64_t value) {
		return b.Constant(spv::OpConstant, t_u64, static_cast<uint32_t>(value),
		                  static_cast<uint32_t>(value >> 32u));
	};
	const auto f32c = [&](float value) {
		return b.Constant(spv::OpConstant, t_f32, std::bit_cast<uint32_t>(value));
	};
	const auto binary = [&](spv::Op op, uint32_t type, uint32_t lhs, uint32_t rhs) {
		const auto result = b.AllocateId();
		b.AddFunction(op, type, result, lhs, rhs);
		return result;
	};
	const auto unary = [&](spv::Op op, uint32_t type, uint32_t value) {
		const auto result = b.AllocateId();
		b.AddFunction(op, type, result, value);
		return result;
	};
	const auto select = [&](uint32_t type, uint32_t condition, uint32_t yes, uint32_t no) {
		const auto result = b.AllocateId();
		b.AddFunction(spv::OpSelect, type, result, condition, yes, no);
		return result;
	};

	// Parameters: the record buffer address (two dwords) is all this shader needs.
	const auto push_array = b.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, sizeof(uint32_t)}}},
	    t_u32, u32c(IR::PushData::MeshEmulatedDrawDwordCount));
	const auto push_block = b.DecoratedType(spv::OpTypeStruct,
	                                        {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                         {spv::OpDecorate, {spv::DecorationBlock}}},
	                                        push_array);
	const auto push = b.DefineGlobalVariable(pointer(spv::StorageClassPushConstant, push_block),
	                                         spv::StorageClassPushConstant);
	std::vector<uint32_t> interface;
	const auto            vertex_index =
	    b.DefineGlobalVariable(pointer(spv::StorageClassInput, t_i32), spv::StorageClassInput);
	b.AddAnnotation(spv::OpDecorate, vertex_index, spv::DecorationBuiltIn, spv::BuiltInVertexIndex);
	interface.push_back(vertex_index);
	const auto per_vertex_type =
	    b.DecoratedType(spv::OpTypeStruct,
	                    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInPosition}},
	                     {spv::OpDecorate, {spv::DecorationBlock}}},
	                    t_v4);
	const auto per_vertex = b.DefineGlobalVariable(
	    pointer(spv::StorageClassOutput, per_vertex_type), spv::StorageClassOutput);
	interface.push_back(per_vertex);
	std::vector<uint32_t> parameters(vec4s.size(), 0);
	for (size_t i = 0; i < vec4s.size(); i++) {
		if (vec4s[i].kind != IR::StageOutputKind::Parameter) {
			continue;
		}
		parameters[i] =
		    b.DefineGlobalVariable(pointer(spv::StorageClassOutput, t_v4), spv::StorageClassOutput);
		b.AddAnnotation(spv::OpDecorate, parameters[i], spv::DecorationLocation, vec4s[i].location);
		interface.push_back(parameters[i]);
	}
	uint32_t layer_variable = 0;
	if (has_layer) {
		layer_variable = b.DefineGlobalVariable(pointer(spv::StorageClassOutput, t_u32),
		                                        spv::StorageClassOutput);
		b.AddAnnotation(spv::OpDecorate, layer_variable, spv::DecorationBuiltIn, spv::BuiltInLayer);
		interface.push_back(layer_variable);
	}

	const auto main = b.AllocateId();
	b.AddFunction(spv::OpFunction, t_void, main, spv::FunctionControlMaskNone, t_fn);
	b.AddFunction(spv::OpLabel, b.AllocateId());
	const auto push_dword = [&](uint32_t index) {
		const auto access = b.AllocateId();
		const auto value  = b.AllocateId();
		b.AddFunction(spv::OpAccessChain, pointer(spv::StorageClassPushConstant, t_u32), access,
		              push, u32c(0), u32c(index));
		b.AddFunction(spv::OpLoad, t_u32, value, access);
		return value;
	};
	const auto load_word = [&](uint32_t record, uint32_t word) {
		const auto address = binary(spv::OpIAdd, t_u64, record,
		                            binary(spv::OpIMul, t_u64, unary(spv::OpUConvert, t_u64, word),
		                                   u64c(sizeof(uint32_t))));
		const auto access  = b.AllocateId();
		const auto value   = b.AllocateId();
		b.AddFunction(spv::OpConvertUToPtr, pointer(spv::StorageClassPhysicalStorageBuffer, t_u32),
		              access, address);
		constexpr uint32_t alignment = sizeof(uint32_t);
		b.AddFunction(spv::OpLoad, t_u32, value, access, spv::MemoryAccessAlignedMask, alignment);
		return value;
	};

	// Which primitive slot and corner this invocation draws.
	const auto raw_index = b.AllocateId();
	b.AddFunction(spv::OpLoad, t_i32, raw_index, vertex_index);
	const auto index        = unary(spv::OpBitcast, t_u32, raw_index);
	const auto slots        = u32c(layout.VerticesPerSlot());
	const auto slot         = binary(spv::OpUDiv, t_u32, index, slots);
	const auto rest         = binary(spv::OpUMod, t_u32, index, slots);
	const auto prim         = binary(spv::OpUDiv, t_u32, rest, u32c(3));
	const auto corner       = binary(spv::OpUMod, t_u32, rest, u32c(3));
	const auto address_pair = b.AllocateId();
	b.AddFunction(spv::OpCompositeConstruct, t_v2u, address_pair,
	              push_dword(IR::PushData::MeshEmulationAddressDword),
	              push_dword(IR::PushData::MeshEmulationAddressDword + 1));
	const auto record = binary(
	    spv::OpIAdd, t_u64, unary(spv::OpBitcast, t_u64, address_pair),
	    binary(spv::OpIMul, t_u64, unary(spv::OpUConvert, t_u64, slot), u64c(layout.SlotBytes())));

	// A slot beyond the workgroup's primitive count, a culled primitive, and a corrupt vertex
	// index all draw nothing: every corner takes the same position, so the triangle has no area.
	const auto primitive_count = load_word(record, u32c(1));
	const auto word =
	    binary(spv::OpIAdd, t_u32, u32c(layout.PrimitiveOffsetWords()),
	           binary(spv::OpIMul, t_u32, prim, u32c(MeshEmulationLayout::kPrimitiveWords)));
	const auto packed = load_word(record, word);
	const auto culled =
	    binary(spv::OpINotEqual, t_bool,
	           binary(spv::OpBitwiseAnd, t_u32, packed, u32c(0x80000000u)), u32c(0));
	const auto vertex = binary(spv::OpBitwiseAnd, t_u32,
	                           binary(spv::OpShiftRightLogical, t_u32, packed,
	                                  binary(spv::OpIMul, t_u32, corner, u32c(10))),
	                           u32c(0x3ffu));
	auto       visible =
	    binary(spv::OpLogicalAnd, t_bool, binary(spv::OpULessThan, t_bool, prim, primitive_count),
	           unary(spv::OpLogicalNot, t_bool, culled));
	visible           = binary(spv::OpLogicalAnd, t_bool, visible,
	                           binary(spv::OpULessThan, t_bool, vertex, u32c(layout.vertices)));
	const auto source = select(t_u32, visible, vertex, u32c(0));

	for (size_t i = 0; i < vec4s.size(); i++) {
		const auto first =
		    binary(spv::OpIAdd, t_u32,
		           u32c(MeshEmulationLayout::kHeaderWords + static_cast<uint32_t>(i) * 4u),
		           binary(spv::OpIMul, t_u32, source, u32c(layout.VertexWords())));
		uint32_t lanes[4] {};
		for (uint32_t c = 0; c < 4; c++) {
			lanes[c] = unary(spv::OpBitcast, t_f32,
			                 load_word(record, binary(spv::OpIAdd, t_u32, first, u32c(c))));
			if (vec4s[i].kind == IR::StageOutputKind::Position) {
				lanes[c] = select(t_f32, visible, lanes[c], f32c(c == 3 ? 1.0f : 0.0f));
			}
		}
		const auto value = b.AllocateId();
		b.AddFunction(spv::OpCompositeConstruct, t_v4, value, lanes[0], lanes[1], lanes[2],
		              lanes[3]);
		if (vec4s[i].kind == IR::StageOutputKind::Position) {
			const auto access = b.AllocateId();
			b.AddFunction(spv::OpAccessChain, pointer(spv::StorageClassOutput, t_v4), access,
			              per_vertex, u32c(0));
			b.AddFunction(spv::OpStore, access, value);
		} else {
			b.AddFunction(spv::OpStore, parameters[i], value);
		}
	}
	if (has_layer) {
		const auto layer = load_word(record, binary(spv::OpIAdd, t_u32, word, u32c(1)));
		b.AddFunction(spv::OpStore, layer_variable, select(t_u32, visible, layer, u32c(0)));
	}
	b.AddFunction(spv::OpReturn);
	b.AddFunction(spv::OpFunctionEnd);
	b.AddEntryPoint(spv::ExecutionModelVertex, main, "main", interface);
	return {b.Build(), layout.SlotWords()};
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
