// Tests for the lock that keeps two emulators from writing the same pipeline lists.

#include "common/fileLock.h"

#include <filesystem>
#include <iostream>

namespace {

using Common::FileLock;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAILED: " << message << '\n';
		g_failures++;
	}
}

std::filesystem::path LockPath() {
	return std::filesystem::temp_directory_path() / "kyty-file-lock-test" / "nested" / "T.lock";
}

void TestOneHolderAtATime() {
	std::error_code error;
	std::filesystem::remove_all(LockPath().parent_path().parent_path(), error);
	{
		FileLock first;
		Check(first.TryAcquire(LockPath()) && first.Held(),
		      "the first taker gets the lock, and its folder is created");
		FileLock second;
		Check(!second.TryAcquire(LockPath()) && !second.Held(),
		      "a second taker does not get it while the first holds it");
		Check(first.TryAcquire(LockPath()), "the holder taking it again keeps it");
	}
	FileLock after;
	Check(after.TryAcquire(LockPath()), "a lock is free again once its holder is gone");
}

} // namespace

int main() {
	TestOneHolderAtATime();
	std::error_code error;
	std::filesystem::remove_all(LockPath().parent_path().parent_path(), error);
	if (g_failures != 0) {
		std::cerr << g_failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "file lock tests passed\n";
	return 0;
}
