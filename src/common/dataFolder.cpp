#include "common/dataFolder.h"

#include <system_error>

namespace Common {

namespace {

bool IsPlainRelativePath(const std::filesystem::path& entry) {
	if (entry.empty() || entry.has_root_name() || entry.has_root_directory()) {
		return false;
	}
	for (const auto& part: entry) {
		if (part == "." || part == "..") {
			return false;
		}
	}
	return true;
}

bool Exists(const std::filesystem::path& path) {
	std::error_code ec;
	return std::filesystem::symlink_status(path, ec).type() !=
	       std::filesystem::file_type::not_found;
}

DataMove Fail(std::string* error, const std::string& text) {
	if (error != nullptr) {
		*error = text;
	}
	return DataMove::Failed;
}

} // namespace

DataMove MoveToDataFolder(const std::filesystem::path& data_folder,
                          const std::filesystem::path& entry, std::string* error) {
	namespace fs = std::filesystem;

	if (data_folder.empty() || !Exists(entry)) {
		return DataMove::NotNeeded;
	}
	if (!IsPlainRelativePath(entry)) {
		return Fail(error, "not a plain relative path: " + entry.string());
	}
	std::error_code ec;
	if (fs::exists(data_folder, ec) && fs::equivalent(data_folder, fs::current_path(ec), ec)) {
		return DataMove::NotNeeded;
	}

	const auto target = data_folder / entry;
	if (Exists(target)) {
		return DataMove::Kept;
	}
	fs::create_directories(target.parent_path(), ec);
	if (ec) {
		return Fail(error, target.parent_path().string() + ": " + ec.message());
	}

	fs::rename(entry, target, ec);
	if (!ec) {
		return DataMove::Moved;
	}

	// The rename fails across file systems. Copy under a name of its own first, so an
	// interrupted copy never looks like a complete entry.
	auto staging = target;
	staging += ".moving";
	fs::remove_all(staging, ec);
	fs::copy(entry, staging, fs::copy_options::recursive | fs::copy_options::copy_symlinks, ec);
	if (!ec) {
		fs::rename(staging, target, ec);
	}
	if (ec) {
		const auto text = ec.message();
		fs::remove_all(staging, ec);
		return Fail(error, entry.string() + " -> " + target.string() + ": " + text);
	}
	fs::remove_all(entry, ec);
	if (ec) {
		return Fail(error, "copied, but could not remove " + entry.string() + ": " + ec.message());
	}
	return DataMove::Moved;
}

} // namespace Common
