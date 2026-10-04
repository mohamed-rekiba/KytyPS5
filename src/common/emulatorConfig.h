#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };
// What VideoOutGetResolutionStatus reports to the game. Uhd: 4K unless the game's param.json
// enables resolution detection (the behaviour so far). Window: the window's size, so a game running
// in a 1920x1080 window renders 1080p.
enum class VideoOutResolution { Uhd, Window };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	std::optional<ControllerColor> controller_color;
	PresentMode            present_mode                = PresentMode::Mailbox;
	VideoOutResolution             video_out_resolution            = VideoOutResolution::Uhd;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   vr_enabled                  = false;
	bool                   amd_cpu_enabled             = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	// Name Vulkan objects and label passes, draws, dispatches and copies, so a GPU capture reads
	// as the emulator's work and not as anonymous commands.
	bool                   gpu_debug_labels_enabled        = false;
	// Record the guest GPU stream into this file: `gpu_capture_frames` frames, starting at the
	// first frame boundary at or after frame `gpu_capture_first_frame`.
	std::filesystem::path gpu_capture_file;
	uint32_t              gpu_capture_first_frame = 0;
	uint32_t              gpu_capture_frames      = 1;
	// When set, recording starts at the first frame boundary after this file appears, and
	// `gpu_capture_first_frame` is not used. Lets a script start the capture in a chosen scene.
	std::filesystem::path gpu_capture_trigger_file;
	// Store a hash of every image the host GPU wrote, each time this many draws and dispatches
	// have passed. A replay then names the first image that differs. 0: no such checks.
	uint32_t gpu_capture_checks = 0;
	// A draw whose pipeline is not built yet waits for it, whatever that takes. Off: the draw
	// waits only a moment and is skipped until the pipeline is ready, so the game does not stop.
	bool pipeline_wait_enabled = false;
	// Replay a recorded guest GPU stream instead of running a game, this many times.
	std::filesystem::path  gpu_replay_file;
	uint32_t               gpu_replay_loops                = 1;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   tessellation_enabled        = false;
	bool                   playgo_hack_enabled         = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
const std::optional<ControllerColor>& GetControllerColor();
PresentMode GetPresentMode();
VideoOutResolution                    GetVideoOutResolution();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     VrEnabled();
bool     AmdCpuEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();
bool GpuDebugLabelsEnabled();
std::filesystem::path GetGpuCaptureFile();
uint32_t              GetGpuCaptureFirstFrame();
uint32_t              GetGpuCaptureFrames();
std::filesystem::path GetGpuCaptureTriggerFile();
uint32_t              GetGpuCaptureChecks();
bool                  PipelineWaitEnabled();
std::filesystem::path GetGpuReplayFile();
uint32_t              GetGpuReplayLoops();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool TessellationEnabled();
bool PlayGoHackEnabled();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
