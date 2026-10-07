// A program list on disk holds what each host shader program was translated from. These tests
// cover the file format: what was written is read back, and a damaged tail loses only itself.

#include "graphics/host_gpu/renderer/pipeline/programList.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::AppendProgramRecord;
using Libs::Graphics::ProgramRecord;
using Libs::Graphics::ReadProgramRecords;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ProgramListTests: failed: %s\n", message);
		std::abort();
	}
}

ProgramRecord Vertex() {
	ProgramRecord record;
	record.stage           = Libs::Graphics::ShaderType::Vertex;
	record.hash            = 0x1122334455667788ull;
	record.user_data_count = 24;
	record.push_data_start = 3;
	record.code            = {0xbf8c007fu, 0x7e000280u, 0xbf810000u};
	record.input_info      = {1, 2, 3, 4, 5, 6, 7};
	record.specialization.buffers.push_back(
	    {.packed_stride = 16, .descriptor_swizzle = 0x688, .zero_stride_oob = true});
	record.specialization.images.push_back({.mip_count = 5, .indirect_root = 2, .cube = true});
	record.specialization.images.push_back({.shader_swizzle = 0x123, .fmask = true});
	return record;
}

ProgramRecord Geometry() {
	ProgramRecord record;
	record.stage     = Libs::Graphics::ShaderType::Mesh;
	record.hash      = 7;
	record.code      = {1, 2};
	record.back_code = {3, 4, 5};
	return record;
}

void TestRoundTrip() {
	std::vector<uint8_t> bytes;
	AppendProgramRecord(bytes, Vertex());
	AppendProgramRecord(bytes, Geometry());
	const auto records = ReadProgramRecords(bytes);
	Check(records.size() == 2 && records[0] == Vertex() && records[1] == Geometry(),
	      "records were not read back as written");
	Check(ReadProgramRecords({}).empty(), "an empty file gave a record");
}

void TestEqualRecordsGiveEqualBytes() {
	std::vector<uint8_t> first;
	std::vector<uint8_t> second;
	AppendProgramRecord(first, Vertex());
	AppendProgramRecord(second, Vertex());
	Check(first == second, "the same record was written as different bytes");
}

void TestCutOffTail() {
	std::vector<uint8_t> bytes;
	AppendProgramRecord(bytes, Vertex());
	const auto whole = bytes.size();
	AppendProgramRecord(bytes, Geometry());
	for (auto size = whole; size < bytes.size(); size++) {
		const auto records = ReadProgramRecords(std::span(bytes).first(size));
		Check(records.size() == 1 && records[0] == Vertex(),
		      "a cut-off record took the record before it with it, or was read");
	}
}

void TestDamagedRecord() {
	std::vector<uint8_t> bytes;
	AppendProgramRecord(bytes, Vertex());
	const auto whole = bytes.size();
	AppendProgramRecord(bytes, Geometry());
	bytes[whole + 9] ^= 0x40;
	const auto records = ReadProgramRecords(bytes);
	Check(records.size() == 1 && records[0] == Vertex(), "a damaged record was read");
	bytes[5] ^= 0x01;
	Check(ReadProgramRecords(bytes).empty(), "records after a damaged one were read");
}

} // namespace

int main() {
	TestRoundTrip();
	TestEqualRecordsGiveEqualBytes();
	TestCutOffTail();
	TestDamagedRecord();
	std::puts("ProgramListTests: all cases passed");
	return 0;
}
