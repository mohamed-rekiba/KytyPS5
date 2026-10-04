// Tests for the capture file of the guest GPU stream: what the writer stores, the reader returns,
// in the same order, and a damaged file is reported instead of replayed. No GPU is needed.

#include "graphics/guest_gpu/capture/captureFile.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace {

using namespace Libs::Graphics::Capture;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		g_failures++;
	}
}

std::filesystem::path TempFile(const char* name) {
	return std::filesystem::temp_directory_path() / name;
}

std::vector<uint8_t> PageFilledWith(uint8_t value) {
	return std::vector<uint8_t>(PageSize, value);
}

void TestRecordsComeBackInOrder() {
	const auto path = TempFile("kyty_gpu_capture_order.bin");
	{
		Writer writer;
		Check(writer.Open(path), "the writer opens a new file");
		writer.WriteStruct(RecordType::MapRange, Range {0x200000000, 0x8000});

		std::vector<uint64_t> addresses = {0x200000000, 0x200003000};
		auto                  data      = PageFilledWith(0x11);
		const auto            second    = PageFilledWith(0x22);
		data.insert(data.end(), second.begin(), second.end());
		writer.WritePages(addresses, data);

		writer.Write(RecordType::FrameBegin);
		Submission submission;
		submission.kind             = static_cast<uint32_t>(SubmissionKind::Compute);
		submission.queue            = 0x21;
		submission.commands_address = 0x200001000;
		submission.commands_dwords  = 77;
		writer.WriteStruct(RecordType::Submission, submission);

		const std::vector<uint8_t> label = {1, 2, 3, 4};
		writer.WriteBytes(0x200002ff0, label, true);
		writer.Write(RecordType::FrameEnd);
		Check(writer.Close(), "the writer closes without a failed write");
	}

	Reader reader;
	Check(reader.Open(path) == Reader::OpenResult::Ok, "the reader accepts the file");
	Reader::Record record;

	Check(reader.Next(record) && record.type == RecordType::MapRange, "first record: map");
	Range range;
	Check(record.As(range) && range.address == 0x200000000 && range.size == 0x8000,
	      "the mapped range is unchanged");

	Check(reader.Next(record) && record.type == RecordType::Pages, "second record: pages");
	std::vector<uint64_t> seen;
	bool                  bytes_ok = true;
	Check(Reader::ForEachPage(record,
	                          [&](uint64_t address, std::span<const uint8_t> bytes) {
		                          const uint8_t expected = seen.empty() ? 0x11 : 0x22;
		                          bytes_ok &= bytes.size() == PageSize &&
		                                      bytes.front() == expected && bytes.back() == expected;
		                          seen.push_back(address);
	                          }),
	      "the pages record splits into pages");
	Check(seen == std::vector<uint64_t> {0x200000000, 0x200003000}, "page addresses are unchanged");
	Check(bytes_ok, "page bytes are unchanged");

	Check(reader.Next(record) && record.type == RecordType::FrameBegin && record.payload.empty(),
	      "third record: frame begin, no payload");

	Check(reader.Next(record) && record.type == RecordType::Submission,
	      "fourth record: submission");
	Submission submission;
	Check(record.As(submission) && submission.queue == 0x21 &&
	          submission.commands_address == 0x200001000 && submission.commands_dwords == 77,
	      "the submission is unchanged");

	Check(reader.Next(record) && record.type == RecordType::WaitBytes, "fifth record: wait bytes");
	uint64_t                 address = 0;
	std::span<const uint8_t> bytes;
	bool                     had_blocked = false;
	Check(Reader::Bytes(record, address, bytes, had_blocked) && had_blocked &&
	          address == 0x200002ff0 && bytes.size() == 4 && bytes[0] == 1 && bytes[3] == 4,
	      "the wait bytes are unchanged");

	Check(reader.Next(record) && record.type == RecordType::FrameEnd, "sixth record: frame end");
	Check(!reader.Next(record), "no record after the last one");
	Check(!reader.Truncated(), "a complete file is not reported as truncated");
	reader.Close();
	std::filesystem::remove(path);
}

void TestForeignFileIsRejected() {
	const auto path = TempFile("kyty_gpu_capture_foreign.bin");
	{
		std::FILE* file = std::fopen(path.string().c_str(), "wb");
		Check(file != nullptr, "the test can create a file");
		if (file != nullptr) {
			const char text[] = "this is not a capture, it is some other file";
			(void)std::fwrite(text, 1, sizeof(text), file);
			(void)std::fclose(file);
		}
	}
	Reader reader;
	Check(reader.Open(path) == Reader::OpenResult::NotACapture,
	      "a file with another magic is rejected");
	Check(reader.Open(TempFile("kyty_gpu_capture_missing.bin")) == Reader::OpenResult::Missing,
	      "a missing file is rejected");
	std::filesystem::remove(path);
}

void TestCutFileIsReportedAsTruncated() {
	const auto path = TempFile("kyty_gpu_capture_cut.bin");
	{
		Writer writer;
		Check(writer.Open(path), "the writer opens a new file");
		writer.Write(RecordType::FrameBegin);
		const std::vector<uint64_t> addresses = {0x200000000};
		writer.WritePages(addresses, PageFilledWith(0x33));
		Check(writer.Close(), "the writer closes");
	}
	// Cut the file in the middle of the page record.
	std::filesystem::resize_file(path, std::filesystem::file_size(path) - 8);

	Reader reader;
	Check(reader.Open(path) == Reader::OpenResult::Ok, "the header of a cut file is still valid");
	Reader::Record record;
	Check(reader.Next(record) && record.type == RecordType::FrameBegin, "the whole record is read");
	Check(!reader.Next(record), "the cut record is not returned");
	Check(reader.Truncated(), "the cut is reported");
	reader.Close();
	std::filesystem::remove(path);
}

void TestPagesWithWrongSizeFailTheCapture() {
	const auto path = TempFile("kyty_gpu_capture_badpages.bin");
	Writer     writer;
	Check(writer.Open(path), "the writer opens a new file");
	const std::vector<uint64_t> addresses = {0x200000000, 0x200001000};
	writer.WritePages(addresses, PageFilledWith(0x44)); // one page of data for two addresses
	Check(writer.Failed(), "page data that does not match the addresses fails the capture");
	Check(!writer.Close(), "a failed capture reports the failure on close");
	std::filesystem::remove(path);

	Reader::Record record;
	record.type = RecordType::Pages;
	record.payload.resize(sizeof(PageHeader) + PageSize - 1);
	Check(!Reader::ForEachPage(record, [](uint64_t, std::span<const uint8_t>) {}),
	      "a pages record that is not a whole number of pages is rejected");
}

void TestCaptureOfAnotherBuildIsRefused() {
	const auto        path = TempFile("kyty_gpu_capture_layout.bin");
	const StateLayout written {
	    .version = 3, .processor_state_size = 1000, .started_submission_size = 200};
	{
		Writer writer;
		Check(writer.Open(path, written), "the writer opens a new file");
		writer.Write(RecordType::End);
		Check(writer.Close(), "the writer closes");
	}
	Reader reader;
	Check(reader.Open(path, written) == Reader::OpenResult::Ok,
	      "a build with the same layout reads the capture");
	auto other                 = written;
	other.processor_state_size = 1008;
	Check(reader.Open(path, other) == Reader::OpenResult::OtherLayout,
	      "a build whose processor state has another size refuses it");
	other = written;
	other.started_submission_size++;
	Check(reader.Open(path, other) == Reader::OpenResult::OtherLayout,
	      "a build whose suspended submission has another size refuses it");
	other = written;
	other.version++;
	Check(reader.Open(path, other) == Reader::OpenResult::OtherLayout,
	      "a build with another layout version refuses it");
	Reader::Record record;
	Check(!reader.Next(record), "a refused capture gives no record");

	// A capture of an older file format version is told apart from a foreign file.
	{
		std::FILE* file = std::fopen(path.string().c_str(), "r+b");
		Check(file != nullptr, "the test can reopen the file");
		if (file != nullptr) {
			const uint32_t old_version = FileVersion - 1;
			(void)std::fseek(file, sizeof(FileMagic), SEEK_SET);
			(void)std::fwrite(&old_version, 1, sizeof(old_version), file);
			(void)std::fclose(file);
		}
	}
	Check(reader.Open(path, written) == Reader::OpenResult::OtherVersion,
	      "a capture of another file version is reported as such");
	std::filesystem::remove(path);
}

void TestPagesAreStoredSmallerAndComeBackWhole() {
	const auto path = TempFile("kyty_gpu_capture_packed.bin");
	// Pages as guest memory has them: long stretches of one value, and one page of noise.
	std::vector<uint64_t> addresses;
	std::vector<uint8_t>  data;
	for (uint64_t i = 0; i < 64; i++) {
		addresses.push_back(0x200000000 + i * PageSize);
		auto page = PageFilledWith(static_cast<uint8_t>(i));
		if (i == 5) {
			uint32_t state = 12345;
			for (auto& byte: page) {
				state = state * 1664525u + 1013904223u;
				byte  = static_cast<uint8_t>(state >> 24u);
			}
		}
		data.insert(data.end(), page.begin(), page.end());
	}
	{
		Writer writer;
		Check(writer.Open(path), "the writer opens a new file");
		// Packed away from the writer, as the scanning threads do.
		writer.WritePacked(Writer::PackPages(addresses, data));
		Check(writer.Close(), "the writer closes");
	}
	Check(std::filesystem::file_size(path) < data.size() / 4,
	      "the file is much smaller than the pages");

	Reader reader;
	Check(reader.Open(path) == Reader::OpenResult::Ok, "the reader accepts the file");
	Reader::Record record;
	Check(reader.Next(record) && record.type == RecordType::Pages,
	      "the reader hands out a plain pages record");
	std::vector<uint64_t> seen;
	std::vector<uint8_t>  bytes_back;
	Check(Reader::ForEachPage(record,
	                          [&](uint64_t address, std::span<const uint8_t> bytes) {
		                          seen.push_back(address);
		                          bytes_back.insert(bytes_back.end(), bytes.begin(), bytes.end());
	                          }),
	      "the record splits into pages");
	Check(seen == addresses && bytes_back == data, "addresses and bytes are unchanged");
	reader.Close();

	// A flipped byte inside the compressed data must not be replayed.
	{
		std::FILE* file = std::fopen(path.string().c_str(), "r+b");
		Check(file != nullptr, "the test can reopen the file");
		if (file != nullptr) {
			(void)std::fseek(file, -20, SEEK_END);
			const uint8_t garbage[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
			(void)std::fwrite(garbage, 1, sizeof(garbage), file);
			(void)std::fclose(file);
		}
	}
	Check(reader.Open(path) == Reader::OpenResult::Ok, "the header is still valid");
	Check(!reader.Next(record) && reader.Damaged(), "a damaged page record is reported");
	reader.Close();
	std::filesystem::remove(path);
}

void TestGuestWritesComeBack() {
	const auto                    path   = TempFile("kyty_gpu_capture_writes.bin");
	const std::vector<WriteRange> writes = {
	    {0x200010000, 0x10000, static_cast<uint32_t>(WriteTarget::Buffers), 0},
	    {0x200012345, 1, static_cast<uint32_t>(WriteTarget::Images), 0},
	};
	{
		Writer writer;
		Check(writer.Open(path), "the writer opens a new file");
		writer.Write(RecordType::GuestWrites, {reinterpret_cast<const uint8_t*>(writes.data()),
		                                       writes.size() * sizeof(WriteRange)});
		Check(writer.Close(), "the writer closes");
	}
	Reader reader;
	Check(reader.Open(path) == Reader::OpenResult::Ok, "the reader accepts the file");
	Reader::Record record;
	Check(reader.Next(record) && record.type == RecordType::GuestWrites, "the record comes back");
	std::vector<WriteRange> back;
	Check(Reader::ForEach<WriteRange>(record,
	                                  [&](const WriteRange& write) { back.push_back(write); }),
	      "it splits into ranges");
	Check(back.size() == 2 && back[0].address == 0x200010000 && back[0].size == 0x10000 &&
	          back[1].address == 0x200012345 &&
	          back[1].target == static_cast<uint32_t>(WriteTarget::Images),
	      "the ranges are unchanged");
	record.payload.pop_back();
	Check(!Reader::ForEach<WriteRange>(record, [](const WriteRange&) {}),
	      "a record that is not a whole number of ranges is rejected");
	reader.Close();
	std::filesystem::remove(path);
}

} // namespace

int main() {
	TestCaptureOfAnotherBuildIsRefused();
	TestPagesAreStoredSmallerAndComeBackWhole();
	TestGuestWritesComeBack();
	TestRecordsComeBackInOrder();
	TestForeignFileIsRejected();
	TestCutFileIsReportedAsTruncated();
	TestPagesWithWrongSizeFailTheCapture();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("gpu capture file tests passed\n");
	return 0;
}
