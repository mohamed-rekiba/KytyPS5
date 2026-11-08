#include "emulator.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/dataFolder.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "common/trace.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/window.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kytyGitVersion.h"
#include "libs/agc.h"
#include "libs/audio.h"
#include "libs/controller.h"
#include "libs/libs.h"
#include "libs/network.h"
#include "loader/additionalContent.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <cstdlib>
#include <filesystem>
#include <thread>

namespace Emulator {

static void PrintSystemInfo() {
	const Common::SystemInfo info = Common::GetSystemInfo();

#if defined(__APPLE__)
	static constexpr auto platform_name = "macOS";
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr auto platform_name = "Windows";
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static constexpr auto platform_name = "Linux";
#else
	static constexpr auto platform_name = "Unknown";
#endif

	LOGF("Build\n"
	     "  version: %s\n\n"
	     "Host\n"
	     "  os:      %s\n"
	     "  cpu:     %s\n"
	     "  threads: %u\n\n",
	     KYTY_BUILD_LABEL, platform_name, info.ProcessorName.c_str(),
	     std::thread::hardware_concurrency());
}

static void KytyClose() {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	rt->Clear();

	LOGF("done!\n");

	Common::Subsystems::EmergencyShutdownActive();
}

static void MountOrCreateDir(const std::filesystem::path& dir, const std::string& point) {
	if (!Common::File::IsDirectoryExisting(dir)) {
		Common::File::CreateDirectories(dir);
	}

	EXIT_NOT_IMPLEMENTED(!Common::File::IsDirectoryExisting(dir));

	Libs::LibKernel::FileSystem::Mount(dir, point);
	auto dir_text = Common::PathToString(dir);
	LOGF("Mounted %s -> %s\n", point.c_str(), dir_text.c_str());
}

static void MoveToDataFolder(const std::filesystem::path& data_folder,
                             const std::filesystem::path& entry) {
	std::string error;
	const auto  result     = Common::MoveToDataFolder(data_folder, entry, &error);
	const auto  entry_text = Common::PathToString(entry);
	const auto  data_text  = Common::PathToString(data_folder);
	switch (result) {
		case Common::DataMove::NotNeeded: break;
		case Common::DataMove::Moved:
			LOGF("Moved %s from the working directory to %s\n", entry_text.c_str(),
			     data_text.c_str());
			break;
		case Common::DataMove::Kept:
			LOGF("Kept %s in the working directory: %s has its own\n", entry_text.c_str(),
			     data_text.c_str());
			break;
		case Common::DataMove::Failed:
			LOGF("Could not move %s to %s: %s\n", entry_text.c_str(), data_text.c_str(),
			     error.c_str());
			break;
	}
}

// Moves the saves, downloads and temporary files of the game that older builds kept in the
// working directory.
static void MoveGameToDataFolder(const std::filesystem::path& data_folder,
                                 const std::string&           title_id) {
	if (data_folder.empty()) {
		return;
	}
	const std::filesystem::path saves = "_SaveData";
	MoveToDataFolder(data_folder, saves / title_id);
	std::error_code ec;
	for (std::filesystem::directory_iterator user(saves / "[users]", ec), end; !ec && user != end;
	     user.increment(ec)) {
		MoveToDataFolder(data_folder, saves / "[users]" / user->path().filename() / title_id);
	}
	MoveToDataFolder(data_folder, std::filesystem::path("_DownloadData") / title_id);
	MoveToDataFolder(data_folder, std::filesystem::path("_TempData") / title_id);
}

static void MountSandboxDirs() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	const auto data_folder = Config::GetDataFolder();
	MoveGameToDataFolder(data_folder, title_id);

	MountOrCreateDir(data_folder / "_DownloadData" / title_id, "/download0");
	MountOrCreateDir(data_folder / "_TempData" / title_id, "/temp0");
	MountOrCreateDir(data_folder / "_TempData" / title_id, "/temp");
}

static void Init(const Config::ConfigOptions& cfg, const std::filesystem::path& param_json,
                 Common::Subsystems& subsystems) {
	EXIT_IF(!Common::Thread::IsMainThread());

	subsystems.Initialize<Config::Lifecycle>();
	Config::Load(cfg);
	subsystems.Initialize<Log::Lifecycle>();
	subsystems.Initialize<Trace::Lifecycle>();

	if (Common::File::IsFileExisting(param_json)) {
		Loader::SystemContentLoadParamSfo(param_json);
	}
	if (cfg.splash_enabled) {
		Libs::Graphics::InitializeSplashScreen(param_json.parent_path());
	}

	// Initialization order is explicit; destruction is automatic and reversed.
	subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
	subsystems.Initialize<Profiler::Lifecycle>();
	subsystems.Initialize<Libs::Network::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::FileSystem::Lifecycle>();
	subsystems.Initialize<Libs::Controller::Lifecycle>();
	subsystems.Initialize<Libs::Audio::Lifecycle>();
	subsystems.Initialize<Libs::Graphics::Lifecycle>();
}

static void LoadElf(const std::filesystem::path& elf, bool dbg_print_reloc = false,
                    const std::filesystem::path& save_name = {}) {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	auto* program = rt->LoadProgram(
	    Libs::LibKernel::FileSystem::GetRealFilename(Common::PathToGenericString(elf)));

	if (dbg_print_reloc) {
		program->dbg_print_reloc = true;
	}

	if (!save_name.empty()) {
		rt->SaveProgram(program, Libs::LibKernel::FileSystem::GetRealFilename(
		                             Common::PathToGenericString(save_name)));
	}
}

static void Execute(const std::filesystem::path& game_patch) {
	std::thread([game_patch] {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->Execute(game_patch);
	}).detach();
	Libs::Graphics::WindowRun();
	std::quick_exit(0);
}

void Run(const RunOptions& options) {
	if (options.app0_dir.empty()) {
		EXIT("app0 directory is required\n");
	}

	if (options.elf.empty()) {
		EXIT("ELF is required\n");
	}

	const auto         param_json = options.app0_dir / "sce_sys" / "param.json";
	Common::Subsystems subsystems(true);
	Init(options.config, param_json, subsystems);

	PrintSystemInfo();
	std::string title_id;
	Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
	if (title_id.empty()) {
		std::string content_id;
		if (Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) &&
		    content_id.size() == 36 && content_id[6] == '-' && content_id[16] == '_') {
			title_id = content_id.substr(7, 9);
		}
	}
	if (!title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {}\n", title_id));
	}
	int ok = atexit(KytyClose);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	// Guest threads are still running, so skip KytyClose() and only flush emergency state.
	ok = at_quick_exit(Common::Subsystems::EmergencyShutdownActive);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/app0");
	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/hostapp");

	MountSandboxDirs();

	Loader::AddContentLoad(options.addcont_dir, options.addcont, title_id);
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	Libs::InitAll(rt->Symbols());

	LoadElf(options.elf);

	Execute(options.game_patch);
}

} // namespace Emulator
