#ifndef KYTY_COMMON_DATAFOLDER_H_
#define KYTY_COMMON_DATAFOLDER_H_

#include "common/cacheFolder.h"

#include <filesystem>
#include <string>

namespace Common {

struct DataEnvironment {
	std::string home;          // HOME
	std::string xdg_data_home; // XDG_DATA_HOME
	std::string app_data;      // APPDATA
};

// Where the emulator keeps what a game writes and cannot build again: saves, downloads and
// temporary files. Not in the working directory: an emulator started from another folder, or a
// new copy of it, would not find the saves of the last one. The host's per-user data folder is
// shared by every copy, and the host does not clear it.
//
// `configured`: the folder given on the command line, if any. When the environment names no
// per-user folder, an empty path: the working directory as before.
[[nodiscard]] inline std::filesystem::path DataFolder(const std::filesystem::path& configured,
                                                      CacheHost host, const DataEnvironment& env) {
	if (!configured.empty()) {
		return configured;
	}
	switch (host) {
		case CacheHost::Windows:
			if (!env.app_data.empty()) {
				return std::filesystem::path(env.app_data) / "KytyPS5";
			}
			break;
		case CacheHost::MacOs:
			if (!env.home.empty()) {
				return std::filesystem::path(env.home) / "Library" / "Application Support" /
				       "KytyPS5";
			}
			break;
		case CacheHost::Linux:
			if (!env.xdg_data_home.empty()) {
				return std::filesystem::path(env.xdg_data_home) / "KytyPS5";
			}
			if (!env.home.empty()) {
				return std::filesystem::path(env.home) / ".local" / "share" / "KytyPS5";
			}
			break;
	}
	return {};
}

enum class DataMove {
	NotNeeded, // nothing in the working directory, or the data folder is the working directory
	Moved,     // the entry is now in the data folder and gone from the working directory
	Kept,      // both have the entry: neither is changed
	Failed,    // the entry stays in the working directory (and may be in the data folder too)
};

// Builds before the data folder kept saves, downloads and temporary files in the working
// directory. Moves `entry` (a relative path such as "_SaveData/<title>") from the working
// directory into `data_folder` when the data folder does not have it yet. Never overwrites: an
// entry the data folder already has stays as it is, and so does the one in the working directory.
// A move across file systems copies, then removes the old entry only when the copy is complete.
DataMove MoveToDataFolder(const std::filesystem::path& data_folder,
                          const std::filesystem::path& entry, std::string* error = nullptr);

} // namespace Common

#endif /* KYTY_COMMON_DATAFOLDER_H_ */
