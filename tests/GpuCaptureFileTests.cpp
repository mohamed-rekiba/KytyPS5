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
	Check(reader.Open(path), "the reader accepts the file");
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
	Check(!reader.Open(path), "a file with another magic is rejected");
	Check(!reader.Open(TempFile("kyty_gpu_capture_missing.bin")), "a missing file is rejected");
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
	// Cut the file in the middle of the page.
	std::filesystem::resize_file(path, std::filesystem::file_size(path) - PageSize / 2);

	Reader reader;
	Check(reader.Open(path), "the header of a cut file is still valid");
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

} // namespace

int main() {
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
