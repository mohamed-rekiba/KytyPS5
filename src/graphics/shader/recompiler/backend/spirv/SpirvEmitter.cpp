#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

// The host stage that runs a program, as a VkShaderStageFlagBits value.
uint32_t HostStageBit(ShaderType stage, bool mesh_emulated = false) {
	if (stage == ShaderType::Mesh && mesh_emulated) {
		return 0x20u; // a compute shader
	}
	switch (stage) {
		case ShaderType::Vertex:
		case ShaderType::Local: return 0x1u;
		case ShaderType::TessellationControl: return 0x2u;
		case ShaderType::TessellationEvaluation: return 0x4u;
		case ShaderType::Pixel: return 0x10u;
		case ShaderType::Compute: return 0x20u;
		case ShaderType::Mesh: return 0x80u;
		default: return 0u;
	}
}

} // namespace

bool UsesSingleLaneModel(ShaderType stage, const HostGpu& host) {
	const bool one_lane_per_invocation = stage == ShaderType::Vertex ||
	                                     stage == ShaderType::Local ||
	                                     stage == ShaderType::TessellationEvaluation;
	return one_lane_per_invocation &&
	       (host.capabilities.subgroup_supported_stages & HostStageBit(stage)) == 0u;
}

std::optional<MissingCapability> FindMissingCapability(const IR::Program& program,
                                                       const HostGpu&     host) {
	const auto  requirements = Emitter::AnalyzeProgramRequirements(program);
	const auto& caps         = host.capabilities;
	// Compute shared memory that is kept in a device buffer is a buffer for this purpose.
	const bool shared_in_buffer =
	    IR::FindBinding(program.bindings, IR::DescriptorBindingKind::SharedMemory) != nullptr;
	if (!caps.buffer_int64_atomics && (requirements.buffer_int64_atomics ||
	                                   (requirements.shared_int64_atomics && shared_in_buffer))) {
		return MissingCapability {"64-bit buffer atomics", "shaderBufferInt64Atomics"};
	}
	if (!caps.shared_int64_atomics && requirements.shared_int64_atomics && !shared_in_buffer) {
		return MissingCapability {"64-bit atomics on compute shared memory",
		                          "shaderSharedInt64Atomics with workgroupMemoryExplicitLayout"};
	}
	if (!caps.image_int64_atomics &&
	    std::ranges::any_of(program.info.images,
	                        [](const IR::ImageResource& image) { return image.atomic64; })) {
		return MissingCapability {"64-bit image atomics", "shaderImageInt64Atomics"};
	}
	if (!caps.float64 && requirements.float64) {
		return MissingCapability {"64-bit floating point", "shaderFloat64"};
	}
	if (!caps.compute_derivatives && requirements.compute_derivatives &&
	    program.stage == ShaderType::Compute) {
		return MissingCapability {"derivatives in a compute shader", "computeDerivativeGroupQuads"};
	}
	const bool subgroups = requirements.subgroup_ballot || requirements.subgroup_barrier ||
	                       requirements.subgroup_shuffle ||
	                       requirements.subgroup_local_invocation_id;
	// A wave barrier has no single-lane form.
	const bool single_lane =
	    UsesSingleLaneModel(program.stage, host) && !requirements.subgroup_barrier;
	if (subgroups && !single_lane &&
	    (caps.subgroup_supported_stages & HostStageBit(program.stage, program.mesh_emulated)) ==
	        0u) {
		return MissingCapability {"subgroup operations in this stage", "subgroupSupportedStages"};
	}
	if (subgroups && !single_lane) {
		// VkSubgroupFeatureFlagBits: basic 0x1, ballot 0x8, shuffle 0x10.
		const uint32_t operations =
		    (requirements.subgroup_local_invocation_id || requirements.subgroup_barrier ? 0x1u
		                                                                                : 0u) |
		    (requirements.subgroup_ballot ? 0x8u : 0u) |
		    (requirements.subgroup_shuffle ? 0x10u : 0u);
		if ((caps.subgroup_supported_operations & operations) != operations) {
			return MissingCapability {"a subgroup operation (lane id, ballot or shuffle)",
			                          "subgroupSupportedOperations"};
		}
	}
	if (!caps.cull_distance &&
	    std::ranges::any_of(program.info.outputs, [](const IR::StageOutput& output) {
		    return output.kind == IR::StageOutputKind::CullDistance;
	    })) {
		return MissingCapability {"cull distance outputs", "shaderCullDistance"};
	}
	const auto reads = [&](auto predicate) {
		return std::ranges::any_of(program.info.inputs, predicate);
	};
	const bool per_vertex  = reads([](const IR::StageInput& input) { return input.per_vertex; });
	const bool centroid    = reads([](const IR::StageInput& input) {
		return input.kind == IR::StageInputKind::BaryCoordSmoothCentroid;
	});
	const bool barycentric = centroid || reads([](const IR::StageInput& input) {
		                         return input.kind == IR::StageInputKind::BaryCoordSmooth ||
		                                input.kind == IR::StageInputKind::BaryCoordNoPerspective;
	                         });
	if (!caps.fragment_barycentric && (per_vertex || barycentric)) {
		return MissingCapability {"barycentrics or raw vertex values", "fragmentShaderBarycentric"};
	}
	if (host.faults.no_per_vertex_inputs && per_vertex) {
		return MissingCapability {"raw vertex values in a pixel shader", "PerVertexKHR inputs"};
	}
	if (host.faults.no_centroid_barycentric && centroid) {
		return MissingCapability {"barycentrics at the centroid", "centroid BaryCoordKHR"};
	}
	return std::nullopt;
}

namespace {

// A capability the host does not offer must not reach SPIR-V: the driver would reject the module
// later with a message that does not say which guest shader needed it.
void ValidateHost(const IR::Program& program, const HostGpu& host) {
	if (const auto missing = FindMissingCapability(program, host)) {
		EXIT("shader needs %s (%s), which the host GPU does not offer: hash=0x%016" PRIx64
		     " stage=%u\n",
		     missing->need, missing->name, program.shader_hash,
		     static_cast<unsigned>(program.stage));
	}
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	using Kind                                             = IR::DescriptorBindingKind;
	constexpr auto                               KindCount = static_cast<size_t>(Kind::Count);
	std::array<std::vector<uint32_t>, KindCount> expected;
	std::array<bool, KindCount>                  present {};
	const auto                                   Dense = [](size_t size) {
		std::vector<uint32_t> values(size);
		for (uint32_t i = 0; i < values.size(); i++) {
			values[i] = i;
		}
		return values;
	};
	auto Expect = [&](Kind kind, std::vector<uint32_t> resources = {}) {
		const auto index = static_cast<size_t>(kind);
		present[index]   = true;
		expected[index]  = std::move(resources);
	};
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = IR::DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		present[static_cast<size_t>(*kind)] = true;
		const auto dynamic = program.info.images[i].mip_mode == IR::ImageMipMode::Dynamic;
		const auto count   = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		expected[static_cast<size_t>(*kind)].insert(expected[static_cast<size_t>(*kind)].end(),
		                                            count, i);
	}
	if (!program.info.samplers.empty()) {
		Expect(Kind::Samplers, Dense(program.info.samplers.size()));
	}
	auto& buffers = expected[static_cast<size_t>(Kind::Buffers)];
	const auto shared = IR::CollectMemoryResources(program, buffers);
	present[static_cast<size_t>(Kind::Buffers)] = !buffers.empty();
	if (shared.gds) {
		Expect(Kind::Gds);
	}
	if (shared.lds && lds_storage) {
		Expect(Kind::SharedMemory);
	}
	if (program.info.uses_dma) {
		Expect(Kind::BdaPagetable);
		Expect(Kind::FaultBuffer);
	}
	if (IR::UsesFlattenedSrt(program)) {
		Expect(Kind::FlattenedSrt);
	}
	if (program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData()) {
		Expect(Kind::ShaderData);
	}

	std::array<bool, KindCount> seen {};
	for (const auto& binding: program.bindings.descriptors) {
		const auto kind = static_cast<size_t>(binding.kind);
		if (kind >= KindCount || seen[kind] || !present[kind] ||
		    binding.resources != expected[kind]) {
			Fail(program, "native descriptor groups do not match shader topology");
		}
		seen[kind] = true;
	}
	for (size_t i = 0; i < KindCount; i++) {
		if (present[i] != seen[i]) {
			Fail(program, "native shader plan is missing a required descriptor group");
		}
	}
	const auto has_shader_data_storage = present[static_cast<size_t>(Kind::ShaderData)];
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto user_data_dwords = program.bindings.user_data_registers.size();
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != user_data_dwords)) ||
	    program.bindings.memory_offset_dword != user_data_dwords + (has_dispatch_threads ? 3u : 0u) ||
	    program.bindings.memory_offset_count != buffers.size() ||
	    has_shader_data_storage != (shader_data_dwords != 0 && !program.bindings.UsesPushData()) ||
	    !std::is_sorted(program.bindings.user_data_registers.begin(),
	                    program.bindings.user_data_registers.end()) ||
	    std::adjacent_find(program.bindings.user_data_registers.begin(),
	                       program.bindings.user_data_registers.end()) !=
	        program.bindings.user_data_registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}

	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::BufferAccessOf(use.user->GetOpcode()) != IR::BufferAccess::Read) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	const auto local_flat_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::AddressOpcodeInfoOf(use.user->GetOpcode()).access == IR::AddressAccess::None)
				       return false;
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::FlatLocal;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || (!program.info.uses_dma && !local_flat_handle(inst))) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
}

} // namespace

Emitter::SpirvRequirements Emitter::AnalyzeProgramRequirements(const IR::Program& program) {
	SpirvRequirements requirements {};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			requirements.float64 |= inst.GetType() == IR::Type::F64;
			if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::Atomic &&
			    inst.GetType() == IR::Type::U64) {
				requirements.buffer_int64_atomics = true;
			}
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "address operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				const auto kind = memory.kind;
				if (kind == IR::ResourceKind::Scratch || kind == IR::ResourceKind::FlatLocal) {
					if (program.scratch_dwords == 0) {
						Fail(program, "scratch operation has no per-thread storage");
					}
					requirements.function_scratch = true;
					if (kind == IR::ResourceKind::FlatLocal && program.stage != ShaderType::Compute &&
					    program.stage != ShaderType::Mesh) {
						requirements.function_lds = true;
					}
				} else if (address_access == IR::AddressAccess::Write) {
					Fail(program, "writable FLAT/GLOBAL addresses require GPU ownership tracking");
				} else {
					// Descriptor stores can alias coherent physical-address loads.
					requirements.coherent_buffers |= memory.coherent;
				}
			}
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "buffer operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::IndirectBuffer &&
				    inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer) {
					requirements.subgroup_local_invocation_id = true;
				}
				if (memory.kind == IR::ResourceKind::Buffer) {
					requirements.coherent_buffers |= memory.coherent;
					if (memory.resource >= program.info.buffers.size()) {
						Fail(program, "buffer operation has invalid resource metadata");
					}
					const auto bits = StorageBufferElementBits(program, memory);
					requirements.buffer_u8 |= bits == 8u;
					requirements.buffer_u16 |= bits == 16u;
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage != ShaderType::Compute) {
							Fail(program, "buffer ADD_TID is only valid for compute shaders");
						}
						requirements.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "shared operation has invalid memory metadata");
				}
				const auto kind = program.memory_info[index].kind;
				if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
					Fail(program, "shared operation has invalid resource kind");
				}
				if (shared_access == IR::SharedAccess::Atomic &&
				    IR::SharedComponentCount(inst.GetOpcode()) == 2u) {
					if (kind != IR::ResourceKind::Lds || program.stage != ShaderType::Compute) {
						Fail(program, "64-bit shared atomics require compute LDS");
					}
					requirements.shared_int64_atomics = true;
				}
				if (program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh &&
				    kind == IR::ResourceKind::Lds) {
					requirements.function_lds = true;
				}
				if (shared_access == IR::SharedAccess::Append ||
				    shared_access == IR::SharedAccess::Consume) {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::StoreCompletion: requirements.subgroup_barrier = true; break;
				case IR::ValueOpcode::BvhIntersect: requirements.bvh = true; break;
				case IR::ValueOpcode::Ballot: requirements.subgroup_ballot = true; break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::ReadFirstLane:
				case IR::ValueOpcode::ReadLane: {
					requirements.subgroup_ballot  = true;
					requirements.subgroup_shuffle = true;
					if (inst.GetOpcode() == IR::ValueOpcode::DppMoveU32) {
						requirements.subgroup_local_invocation_id = true;
					}
					break;
				}
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::WriteLane: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::Permlane16U32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::SwizzleU32:
				case IR::ValueOpcode::PermuteU32:
				case IR::ValueOpcode::BpermuteU32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::LaneId:
					requirements.subgroup_local_invocation_id |=
					    program.stage != ShaderType::TessellationControl;
					break;
				case IR::ValueOpcode::ImageQueryLod: requirements.compute_derivatives = true; break;
				case IR::ValueOpcode::ImageGatherRaw:
					requirements.image_gather_extended = true;
					break;
				case IR::ValueOpcode::SetAttribute: {
					const auto index = inst.Flags<IR::ExportFlags>().index;
					if (index >= program.export_info.size()) {
						Fail(program, "attribute export has invalid metadata");
					}
					if (program.stage == ShaderType::Pixel &&
					    program.export_info[index].vm) {
						requirements.pixel_valid_mask = true;
					}
					break;
				}
				default: break;
			}
		}
	}
	return requirements;
}

std::vector<uint32_t> EmitProgram(const IR::Program& program, ShaderStageInputInfo input_info,
                                  const HostGpu& host) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete || !program.binding_layout_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	IR::ValidateProgram(program, true);
	EmitterState state(program, input_info, host);
	state.single_lane = UsesSingleLaneModel(program.stage, host);
	ValidateHost(program, host);
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(MeshEmulated(state) ? spv::ExecutionModelGLCompute
	                                                : ExecutionModelForStage(state.program.stage),
	                            state.main_func, "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
