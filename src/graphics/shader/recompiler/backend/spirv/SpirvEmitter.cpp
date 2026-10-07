#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

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
	const auto& requirements = program.info;
	const auto& caps         = host.capabilities;
	// Compute shared memory that is kept in a device buffer is a buffer for this purpose. The
	// emitter sets the descriptor counts before it asks.
	const bool shared_in_buffer =
	    program.bindings.descriptor_counts[static_cast<size_t>(IR::DescriptorBindingKind::SharedMemory)] != 0;
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
	uint64_t live_buffers = 0;
	bool uses_lds = false;
	bool uses_gds = false;
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_root != UINT32_MAX;
	};
	bool uses_flattened_srt = std::ranges::any_of(program.info.buffers, uses_mapping) ||
	                          std::ranges::any_of(program.info.images, uses_mapping);
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
			const auto op = inst.GetOpcode();
			uses_flattened_srt |= op == IR::ValueOpcode::ReadConst;
			if (IR::BufferAccessOf(op) != IR::BufferAccess::None ||
			    IR::SharedAccessOf(op) != IR::SharedAccess::None ||
			    IR::AddressOpcodeInfoOf(op).access != IR::AddressAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "typed shader contains invalid memory metadata");
				}
				const auto& memory = program.memory_info[index];
				if (!memory.planning_only) {
					uses_lds |= memory.kind == IR::ResourceKind::FlatLocal;
					if (IR::SharedAccessOf(op) != IR::SharedAccess::None) {
						if (memory.kind != IR::ResourceKind::Lds && memory.kind != IR::ResourceKind::Gds) {
							Fail(program, "typed shader contains invalid shared-memory metadata");
						}
						uses_gds |= memory.kind == IR::ResourceKind::Gds;
						uses_lds |= memory.kind == IR::ResourceKind::Lds;
					} else if (memory.kind == IR::ResourceKind::Buffer ||
					           memory.kind == IR::ResourceKind::ScalarBuffer) {
						if (memory.resource >= program.info.buffers.size()) {
							Fail(program, "typed shader contains an invalid buffer resource");
						}
						live_buffers |= uint64_t{1} << memory.resource;
						for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
							if (child >= program.info.buffers.size()) {
								Fail(program, "typed shader contains an invalid indirect buffer resource");
							}
							live_buffers |= uint64_t{1} << child;
						}
					}
				}
			}
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
	using Kind = IR::DescriptorBindingKind;
	std::array<uint32_t, static_cast<size_t>(Kind::Count)> expected {};
	auto& buffer_count = expected[static_cast<size_t>(Kind::Buffers)];
	for (uint32_t index = 0; index < program.info.buffers.size(); ++index) {
		const auto slot = (live_buffers & (uint64_t{1} << index)) != 0 ? buffer_count++ : UINT32_MAX;
		if (program.info.buffers[index].descriptor_index != slot) {
			Fail(program, "native buffer descriptor index does not match shader topology");
		}
	}
	for (const auto& image: program.info.images) {
		const auto kind = IR::DescriptorBindingForImage(image);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		if (image.mip_count == 0u ||
		    (image.mip_mode != IR::ImageMipMode::Dynamic && image.mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		auto& count = expected[static_cast<size_t>(*kind)];
		if (image.descriptor_index != count) {
			Fail(program, "native image descriptor index does not match shader topology");
		}
		count += image.mip_count;
	}
	expected[static_cast<size_t>(Kind::Samplers)] = static_cast<uint32_t>(program.info.samplers.size());
	expected[static_cast<size_t>(Kind::Gds)] = uses_gds;
	expected[static_cast<size_t>(Kind::SharedMemory)] = uses_lds && lds_storage;
	expected[static_cast<size_t>(Kind::BdaPagetable)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FaultBuffer)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FlattenedSrt)] = uses_flattened_srt;
	expected[static_cast<size_t>(Kind::ShaderData)] =
	    program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData();
	uint64_t descriptor_mask = 0;
	for (uint32_t index = 0; index < expected.size(); ++index) {
		if (expected[index] != 0) descriptor_mask |= uint64_t{1} << index;
	}
	if (program.bindings.descriptor_counts != expected ||
	    program.bindings.descriptor_mask != descriptor_mask) {
		Fail(program, "native descriptor counts do not match shader topology");
	}
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto& registers = program.info.user_data_registers;
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != registers.size())) ||
	    program.bindings.memory_offset_dword != registers.size() + (has_dispatch_threads ? 3u : 0u) ||
	    !std::is_sorted(registers.begin(), registers.end()) ||
	    std::adjacent_find(registers.begin(), registers.end()) != registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const HostGpu& host) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	if (program.info.buffers.size() > IR::ShaderInfo::MaxBuffers) {
		Fail(program, "native shader exceeds the buffer resource limit");
	}
	IR::ValidateProgram(program, true);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, host);
	state.single_lane = UsesSingleLaneModel(program.stage, host);
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	DefineModule(state);
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	ValidateHost(program, host);
	EmitProgram(state);
	state.builder.AddEntryPoint(MeshEmulated(state) ? spv::ExecutionModelGLCompute
	                                                : ExecutionModelForStage(state.program.stage),
	                            state.main_func, "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
