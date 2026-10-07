#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/programList.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// The file is tied to the device and its driver, not to the emulator build. The driver finds
	// an entry by the shader code and state a pipeline is created from, so an entry that a newer
	// emulator build no longer asks for is only unused. With the build in the signature, every
	// update threw the file away, and the next session compiled each pipeline again at its first
	// use: in a 3D scene 500 pipelines took 66 s that way, against 1 to 2 s with the file kept.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	// Translated IR to SPIR-V, checked. Has no state of its own: a worker thread can run it.
	static ShaderRecompiler::CompileResult
	CompileChecked(const char* stage_name, const ShaderRecompiler::CompileOptions& options,
	               ShaderRecompiler::TranslateResult                   translated,
	               const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	               uint32_t                                            push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		if (!result.mesh_vertex_spirv.empty()) {
			if (!ValidateShaderSpirv(options.dump_label, options.shader_hash,
			                         result.mesh_vertex_spirv)) {
				DumpShaderSpirv("mesh_vs", options.shader_hash, result.mesh_vertex_spirv);
				EXIT("%s failed hash=0x%016" PRIx64 ": emulated mesh vertex shader is invalid\n",
				     options.dump_label, options.shader_hash);
			}
			DumpShaderSpirv("mesh_vs", options.shader_hash, result.mesh_vertex_spirv);
		}
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return result;
	}

	// Renderer thread: the host shader modules and the handle of a compiled program.
	Permutation MakePermutation(ShaderRecompiler::CompileResult              result,
	                            ShaderRecompiler::IR::ResourceSpecialization specialization) {
		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		vk::ShaderModule mesh_vertex_module = nullptr;
		if (!result.mesh_vertex_spirv.empty()) {
			mesh_vertex_module = CompileSPV(result.mesh_vertex_spirv, device);
			EXIT_IF(mesh_vertex_module == nullptr);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id                 = ++next_shader_id,
		                       .module             = module,
		                       .mesh_vertex_module = mesh_vertex_module,
		                       .mesh_slot_words    = result.mesh_slot_words},
		};
	}

	struct StageNames {
		const char* label      = nullptr;
		const char* stage_name = nullptr;
	};

	static StageNames NamesOf(ShaderType stage) {
		switch (stage) {
			case ShaderType::Vertex: return {"ShaderRecompiler VS", "vs"};
			case ShaderType::Mesh: return {"ShaderRecompiler MS", "ms"};
			case ShaderType::Local: return {"ShaderRecompiler LS", "ls"};
			case ShaderType::TessellationControl: return {"ShaderRecompiler HS", "hs"};
			case ShaderType::TessellationEvaluation: return {"ShaderRecompiler DS", "ds"};
			case ShaderType::Pixel: return {"ShaderRecompiler PS", "ps"};
			case ShaderType::Compute: return {"ShaderRecompiler CS", "cs"};
			default: EXIT("invalid pipeline shader stage\n");
		}
	}

	// The translator's options for one stage. `user_data`: only its size matters here; the
	// values select the specialization, which the caller passes on its own.
	template <typename InputInfo>
	ShaderRecompiler::CompileOptions
	MakeOptions(ShaderType stage, uint64_t hash, std::span<const uint32_t> user_data,
	            std::span<const uint32_t> back_code, InputInfo& input_info) const {
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = hash;
		options.user_data   = user_data;
		options.back_code   = back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = NamesOf(stage).label;
		options.input_info  = stage_input;
		options.host        = host;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		if (entry != programs.end()) {
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == entry->second.specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		const auto options =
		    MakeOptions(stage, params.hash, user_data, params.back_code, input_info);
		const char* stage_name = NamesOf(stage).stage_name;
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
		}
		Remember(stage, params, input_info, entry->second.specialization, push_data_cursor);
		entry->second.permutations.push_back(
		    MakePermutation(CompileChecked(stage_name, options, std::move(translated),
		                                   entry->second.specialization, push_data_cursor),
		                    entry->second.specialization));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// ---- The program list: what each program was translated from, kept on disk for a game.
	// At the next start worker threads translate the listed programs again, so a draw finds its
	// program ready and the renderer does not stop to translate it.

	// A program a worker built from a record. It has no host objects yet.
	struct Prebuilt {
		ProgramKey                                   key;
		ShaderRecompiler::IR::ResourcePlan           plan;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::CompileResult              result;
	};

	template <typename InputInfo>
	static bool ReadInputInfo(const ProgramRecord& record, InputInfo& info) {
		static_assert(std::is_trivially_copyable_v<InputInfo>);
		if (record.input_info.size() != sizeof(InputInfo)) {
			return false;
		}
		std::memcpy(static_cast<void*>(&info), record.input_info.data(), sizeof(InputInfo));
		info.stage = {};
		return true;
	}

	// Names a record for "is it on the list already". The input info is left out for its key
	// words: its raw bytes have padding, and equal inputs must give an equal name.
	static uint64_t Identity(const ProgramRecord& record, std::span<const uint32_t> static_state) {
		ProgramRecord named = record;
		named.code          = {static_cast<uint32_t>(record.code.size())};
		named.input_info.assign(reinterpret_cast<const uint8_t*>(static_state.data()),
		                        reinterpret_cast<const uint8_t*>(static_state.data()) +
		                            static_state.size_bytes());
		std::vector<uint8_t> bytes;
		AppendProgramRecord(bytes, named);
		return XXH3_64bits(bytes.data(), bytes.size());
	}

	bool StaticStateOf(const ProgramRecord& record, std::vector<uint32_t>& static_state) const {
		const auto build = [&]<typename InputInfo>(InputInfo info) {
			if (!ReadInputInfo(record, info)) {
				return false;
			}
			BuildStageStaticKey(info, static_state);
			return true;
		};
		switch (record.stage) {
			case ShaderType::Vertex:
			case ShaderType::Mesh:
			case ShaderType::Local:
			case ShaderType::TessellationControl:
			case ShaderType::TessellationEvaluation: return build(ShaderVertexInputInfo {});
			case ShaderType::Pixel: return build(ShaderPixelInputInfo {});
			case ShaderType::Compute: return build(ShaderComputeInputInfo {});
			default: return false;
		}
	}

	// The host as the list's signature sees it: every field by value, never the object's bytes,
	// which have padding. A new field must be added here, or programs built for another host
	// would be taken from the list.
	uint64_t HostWords() const {
		const auto& c = host.capabilities;
		const auto& f = host.faults;
		static_assert(sizeof(HostCapabilities) == 24 && sizeof(DriverFaults) == 5,
		              "a host field was added: add it to the list signature");
		const uint32_t words[] = {
		    c.image_view_min_lod,
		    c.cull_distance,
		    c.buffer_int64_atomics,
		    c.shared_int64_atomics,
		    c.image_int64_atomics,
		    c.float64,
		    c.fragment_barycentric,
		    c.depth_bounds,
		    c.depth_clamp,
		    c.depth_clip_enable,
		    c.color_write_enable,
		    c.mesh_shader,
		    c.compute_derivatives,
		    c.push_descriptors,
		    c.subgroup_supported_stages,
		    c.subgroup_supported_operations,
		    f.pushed_buffers_have_no_size,
		    f.no_per_vertex_inputs,
		    f.no_centroid_barycentric,
		    f.volatile_loads_are_reused,
		    f.pipeline_cache_load_is_slow,
		};
		return XXH3_64bits(words, sizeof(words));
	}

	// With the build: a record holds what the translator was given, and what those inputs mean
	// can change with the translator. A list from another build is started again.
	std::string ListSignature() const {
		return fmt::format("KytyPL2:{}:{}:{}:{}:{:016x}\n", KYTY_GIT_HASH,
		                   sizeof(ShaderVertexInputInfo), sizeof(ShaderPixelInputInfo),
		                   sizeof(ShaderComputeInputInfo), HostWords());
	}

	// Worker thread. Builds what the live path builds for the same inputs.
	template <typename InputInfo>
	bool Prebuild(const ProgramRecord& record, Prebuilt& out) const {
		InputInfo info;
		if (!ReadInputInfo(record, info)) {
			return false;
		}
		out.key.stage           = record.stage;
		out.key.hash            = record.hash;
		out.key.user_data_count = record.user_data_count;
		out.key.code_size       = static_cast<uint32_t>(record.code.size());
		BuildStageStaticKey(info, out.key.static_state);
		const std::vector<uint32_t> user_data(record.user_data_count, 0u);
		const auto                  options =
		    MakeOptions(record.stage, record.hash, user_data, record.back_code, info);
		auto translated    = ShaderRecompiler::TranslateProgram(record.code, options);
		out.plan           = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		out.specialization = record.specialization;
		out.result =
		    CompileChecked(NamesOf(record.stage).stage_name, options, std::move(translated),
		                   record.specialization, record.push_data_start);
		return true;
	}

	bool Prebuild(const ProgramRecord& record, Prebuilt& out) const {
		switch (record.stage) {
			case ShaderType::Vertex:
			case ShaderType::Mesh:
			case ShaderType::Local:
			case ShaderType::TessellationControl:
			case ShaderType::TessellationEvaluation:
				return Prebuild<ShaderVertexInputInfo>(record, out);
			case ShaderType::Pixel: return Prebuild<ShaderPixelInputInfo>(record, out);
			case ShaderType::Compute: return Prebuild<ShaderComputeInputInfo>(record, out);
			default: return false;
		}
	}

	// Main thread, once. Reads the list of the game and starts the workers on it.
	void OpenProgramList(const std::filesystem::path& path) {
		const auto           signature = ListSignature();
		std::vector<uint8_t> bytes;
		if (std::FILE* file = std::fopen(Common::PathToString(path).c_str(), "rb")) {
			std::fseek(file, 0, SEEK_END);
			const auto size = std::ftell(file);
			std::fseek(file, 0, SEEK_SET);
			bytes.resize(size > 0 ? static_cast<size_t>(size) : 0);
			if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
				bytes.clear();
			}
			std::fclose(file);
		}
		const bool current = bytes.size() >= signature.size() &&
		                     std::memcmp(bytes.data(), signature.data(), signature.size()) == 0;
		if (current) {
			std::vector<uint32_t> static_state;
			for (auto& record: ReadProgramRecords(std::span(bytes).subspan(signature.size()))) {
				if (StaticStateOf(record, static_state) &&
				    m_listed.insert(Identity(record, static_state)).second) {
					m_records.push_back(std::move(record));
				}
			}
		}
		std::error_code error;
		std::filesystem::create_directories(path.parent_path(), error);
		// A list from another layout or device is started again; a damaged tail is cut off.
		m_list_file = std::fopen(Common::PathToString(path).c_str(), "wb");
		if (m_list_file == nullptr) {
			PipelineCacheLog("Program list: cannot write {}", Common::PathToString(path));
			m_records.clear();
			return;
		}
		std::fwrite(signature.data(), 1, signature.size(), m_list_file);
		std::vector<uint8_t> kept;
		for (const auto& record: m_records) {
			AppendProgramRecord(kept, record);
		}
		std::fwrite(kept.data(), 1, kept.size(), m_list_file);
		std::fflush(m_list_file);
		PipelineCacheLog("Program list: {} program(s) to build ahead from {}", m_records.size(),
		                 Common::PathToString(path));
		if (m_records.empty()) {
			return;
		}
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 4u);
		m_workers_running  = threads;
		for (uint32_t i = 0; i < threads; i++) {
			m_workers.emplace_back([this, started = std::chrono::steady_clock::now()](
			                           std::stop_token stop) {
				for (;;) {
					const auto index = m_next_record.fetch_add(1);
					if (stop.stop_requested() || index >= m_records.size()) {
						if (m_workers_running.fetch_sub(1) == 1 && !stop.stop_requested()) {
							PipelineCacheLog("Program list: {} program(s) built ahead in {} ms",
							                 m_records.size(),
							                 std::chrono::duration_cast<std::chrono::milliseconds>(
							                     std::chrono::steady_clock::now() - started)
							                     .count());
						}
						return;
					}
					Prebuilt built;
					if (Prebuild(m_records[index], built)) {
						std::lock_guard lock(m_built_mutex);
						m_built.push_back(std::move(built));
						m_built_waiting.store(true, std::memory_order_release);
					}
				}
			});
		}
	}

	// Renderer thread, before the programs of a draw or dispatch are looked up. Takes over what
	// the workers have finished. Not later in a draw: adding a permutation can move the ones a
	// stage of the same draw already points to.
	void Adopt() {
		if (!m_built_waiting.load(std::memory_order_acquire)) {
			return;
		}
		std::vector<Prebuilt> built;
		{
			std::lock_guard lock(m_built_mutex);
			built.swap(m_built);
			m_built_waiting.store(false, std::memory_order_release);
		}
		for (auto& item: built) {
			auto entry = programs.try_emplace(std::move(item.key), std::move(item.plan)).first;
			const auto start = item.result.program.bindings.push_data_start_dword;
			if (std::ranges::any_of(entry->second.permutations, [&](const Permutation& known) {
				    return known.program.bindings.push_data_start_dword == start &&
				           known.specialization == item.specialization;
			    })) {
				continue; // a draw needed it before the worker was done
			}
			entry->second.permutations.push_back(
			    MakePermutation(std::move(item.result), std::move(item.specialization)));
			m_adopted++;
		}
	}

	// Renderer thread: a program was translated for a draw. Puts it on the list, once.
	template <typename InputInfo>
	void Remember(ShaderType stage, const ShaderParams& params, const InputInfo& input_info,
	              const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	              uint32_t                                            push_data_cursor) {
		static_assert(std::is_trivially_copyable_v<InputInfo>);
		if (m_list_file == nullptr) {
			return;
		}
		ProgramRecord record;
		record.stage           = stage;
		record.hash            = params.hash;
		record.user_data_count = params.user_data_count;
		record.push_data_start = push_data_cursor;
		record.code.assign(params.code.begin(), params.code.end());
		record.back_code.assign(params.back_code.begin(), params.back_code.end());
		record.specialization = specialization;
		InputInfo plain       = input_info;
		plain.stage           = {};
		record.input_info.resize(sizeof(InputInfo));
		std::memcpy(record.input_info.data(), static_cast<const void*>(&plain), sizeof(InputInfo));
		if (!m_listed.insert(Identity(record, lookup_key.static_state)).second) {
			return; // on the list, and built here before its worker was done
		}
		std::vector<uint8_t> bytes;
		AppendProgramRecord(bytes, record);
		std::fwrite(bytes.data(), 1, bytes.size(), m_list_file);
		std::fflush(m_list_file);
	}

	ProgramCache(vk::Device device, const HostGpu& host): device(device), host(host) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (auto& worker: m_workers) {
			worker.request_stop();
		}
		m_workers.clear();
		if (m_list_file != nullptr) {
			std::fclose(m_list_file);
		}
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
				if (permutation.handle.mesh_vertex_module != nullptr) {
					device.destroyShaderModule(permutation.handle.mesh_vertex_module, nullptr);
				}
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	std::vector<ProgramRecord>                                  m_records;
	std::atomic<size_t>                                         m_next_record {0};
	std::vector<std::jthread>                                   m_workers;
	std::mutex                                                  m_built_mutex;
	std::vector<Prebuilt>                                       m_built;
	std::atomic<bool>                                           m_built_waiting {false};
	std::atomic<uint32_t>                                       m_workers_running {0};
	std::unordered_set<uint64_t>                                m_listed;
	std::FILE*                                                  m_list_file = nullptr;
	uint64_t                                                    m_adopted   = 0;
	vk::Device                                                  device;
	HostGpu                                                     host;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(graphics.device, graphics.host)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	// The list is signed with the build. A build that cannot name its source has no signature
	// to trust, like the driver cache.
	const std::string_view git_hash = KYTY_GIT_HASH;
	if (git_hash == "unknown" || git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Program list: off (the build cannot name its source)");
	} else if (const auto title_id = PipelineCacheTitleId(); !title_id.empty()) {
		m_program_cache->OpenProgramList(std::filesystem::path("_PipelineCache") /
		                                 (title_id + ".programs"));
	}
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			if (pipeline->mesh_compute != nullptr) {
				m_graphics.device.destroyPipeline(pipeline->mesh_compute, nullptr);
				m_graphics.device.destroyPipelineLayout(pipeline->mesh_compute_layout, nullptr);
			}
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}
	if (m_graphics.host.faults.pipeline_cache_load_is_slow) {
		// Measured: 76 s to load 75 MB. The in-memory cache still serves this session.
		PipelineCacheLog("Vulkan pipeline cache: not kept on disk (loading it would hold the "
		                 "start of the game; the game's pipeline list is built ahead instead)");
		vk::PipelineCacheCreateInfo create {};
		EXIT_NOT_IMPLEMENTED(m_graphics.device.createPipelineCache(&create, nullptr,
		                                                           &m_driver_cache) !=
		                     vk::Result::eSuccess);
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	m_program_cache->Adopt();
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		// Without mesh shaders the guest shader runs as a compute shader (see
		// capturedVertexLayout.h).
		mesh.emulated = !m_graphics.host.capabilities.mesh_shader;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		bool within_limits = false;
		if (mesh.emulated) {
			const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
			// Besides the guest LDS, the shader keeps the layers and the allocation in shared
			// memory.
			within_limits = host_threads <= limits.maxComputeWorkGroupInvocations &&
			                host_threads <= limits.maxComputeWorkGroupSize[0] &&
			                (mesh.lds_size_dwords + mesh.max_vertices + 2u) * sizeof(uint32_t) <=
			                    limits.maxComputeSharedMemorySize;
		} else {
			const auto& limits = m_graphics.mesh_shader_properties;
			within_limits =
			    host_threads <= limits.maxMeshWorkGroupInvocations &&
			    host_threads <= limits.maxMeshWorkGroupSize[0] &&
			    mesh.max_vertices <= limits.maxMeshOutputVertices &&
			    mesh.max_primitives <= limits.maxMeshOutputPrimitives &&
			    mesh.lds_size_dwords * sizeof(uint32_t) <= limits.maxMeshSharedMemorySize;
		}
		if (!within_limits) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t push_data_cursor = !mesh_active ? 0
	                            : vertex_info[0].mesh.emulated
	                                ? ShaderRecompiler::IR::PushData::MeshEmulatedDrawDwordCount
	                                : ShaderRecompiler::IR::PushData::MeshDrawDwordCount;
	GraphicsPrograms  result;
	if (pixel_active && !m_graphics.host.capabilities.depth_bounds &&
	    context.GetDepthControl().depth_bounds_enable) {
		// The host has no depth bounds test: the pixel shader applies it to a copy of the depth
		// buffer (see EmitDepthBoundsTest).
		switch (context.GetDepthRenderTarget().z_info.format) {
			case Prospero::DepthFormat::kZ32F: pixel_info.ps_depth_bounds_format = 1; break;
			case Prospero::DepthFormat::kZ16: pixel_info.ps_depth_bounds_format = 2; break;
			default: EXIT("depth bounds test on an unsupported depth format\n");
		}
		pixel_info.ps_depth_bounds_dword = push_data_cursor;
		push_data_cursor += ShaderRecompiler::IR::PushData::DepthBoundsDwordCount;
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	m_program_cache->Adopt();
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
