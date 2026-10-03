#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_

// Container for a capture of the guest GPU stream: a header, then records in the order the GPU
// thread saw them. Each record is a type, a payload size and the payload. The reader hands out
// one record at a time, so a capture larger than host memory can be replayed.
//
// This file knows nothing about the renderer. The recorder and the player give the records their
// meaning.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace Libs::Graphics::Capture {

inline constexpr std::array<char, 8> FileMagic   = {'K', 'G', 'P', 'U', 'C', 'A', 'P', '\0'};
inline constexpr uint32_t            FileVersion = 1;
inline constexpr uint64_t            PageSize    = 4096;

enum class RecordType : uint32_t {
	// Guest ranges visible to the GPU at capture start.
	MapRange = 1,
	// Whole guest pages: `PageHeader`, then `PageSize` bytes, repeated to the end of the record.
	// Before `FrameBegin`: the memory at capture start. After it: pages that changed before the
	// submission, the wait or the frame end that follows.
	Pages = 3,
	// A wait on guest memory that passed: `BytesHeader`, then the bytes it read.
	WaitBytes = 4,
	// A video-out call that rebuilds the configuration at capture start: `VideoOutCall`.
	VideoOut = 5,
	// A submission that the GPU thread started: `Submission`.
	Submission = 6,
	// Start of the captured frames: everything before it is state.
	FrameBegin = 7,
	// The guest's suspend point that ends a frame.
	FrameEnd = 8,
	// Register state of one command processor: `ProcessorHeader`, then the state bytes. The
	// bytes are the emulator's own structure, so they only load into the same build.
	ProcessorState = 9,
	// The GDS bytes.
	Gds = 10,
	// Shader registrations: `ShaderEntry`, repeated to the end of the record.
	Shaders = 11,
	// The picture shown by the frame that ends next: `Picture`.
	Picture = 13,
	// A submission that was suspended in its command stream at capture start: the emulator's
	// own structure, so it only loads into the same build.
	StartedSubmission = 14,
	// Last record of a complete capture. A file without it was cut short or abandoned.
	End = 12,
};

struct FileHeader {
	std::array<char, 8> magic     = FileMagic;
	uint32_t            version   = FileVersion;
	uint32_t            page_size = static_cast<uint32_t>(PageSize);
};

struct RecordHeader {
	uint32_t type     = 0;
	uint32_t reserved = 0;
	uint64_t size     = 0;
};

struct Range {
	uint64_t address = 0;
	uint64_t size    = 0;
};

struct PageHeader {
	uint64_t address = 0;
};

struct BytesHeader {
	uint64_t address = 0;
	uint32_t size    = 0;
	// 1 when the wait had failed its test before it passed: something released it.
	uint32_t had_blocked = 0;
};

enum class SubmissionKind : uint32_t { Graphics = 0, Compute = 1 };

struct Submission {
	uint32_t kind                      = 0;
	uint32_t queue                     = 0; // guest queue number for a compute submission
	uint64_t commands_address          = 0;
	uint64_t commands_dwords           = 0;
	uint64_t constant_commands_address = 0;
	uint64_t constant_commands_dwords  = 0;
};

// What the frame flipped to the display. `hash` covers the guest bytes of the display buffer
// after the GPU's picture was written back; it is only set when `available` is 1.
struct Picture {
	int32_t  handle    = 0;
	int32_t  index     = 0;
	uint32_t available = 0;
	uint32_t reserved  = 0;
	uint64_t hash      = 0;
};

struct ProcessorHeader {
	uint32_t queue_index = 0;
	uint32_t state_size  = 0;
};

struct ShaderEntry {
	uint64_t address             = 0;
	uint64_t hash                = 0;
	uint64_t user_data           = 0;
	uint64_t input_semantics     = 0;
	uint32_t type                = 0;
	uint32_t num_input_semantics = 0;
	uint32_t code_size_bytes     = 0;
	uint32_t scratch_size_dwords = 0;
};

enum class VideoOutCallKind : uint32_t {
	Open            = 0,
	RegisterBuffers = 1,
	SetFlipRate     = 2,
	SetFlipMaster   = 3,
};

inline constexpr uint32_t MaxVideoOutBuffers = 16;

struct VideoOutCall {
	uint32_t kind   = 0;
	int32_t  handle = 0; // handle the live run used (`Open`: the handle it returned)
	// Open
	int32_t bus_type   = 0;
	int32_t open_index = 0;
	// RegisterBuffers
	int32_t set_index          = 0;
	int32_t buffer_index_start = 0;
	int32_t buffer_num         = 0;
	int32_t category           = 0;
	// SetFlipRate
	int32_t rate = 0;
	// SetFlipMaster: `handle` is the slave
	int32_t master_handle = 0;
	// Raw bytes of the guest's attribute structure.
	std::array<uint8_t, 96> attribute {};
	// Guest address of each buffer's pixel data and metadata.
	std::array<uint64_t, MaxVideoOutBuffers> buffer_data {};
	std::array<uint64_t, MaxVideoOutBuffers> buffer_metadata {};
};

class Writer final {
public:
	Writer() = default;
	~Writer() { (void)Close(); }
	Writer(const Writer&)            = delete;
	Writer& operator=(const Writer&) = delete;

	[[nodiscard]] bool Open(const std::filesystem::path& path) {
		(void)Close();
		m_file = std::fopen(path.string().c_str(), "wb");
		if (m_file == nullptr) {
			return false;
		}
		const FileHeader header;
		return WriteRaw(&header, sizeof(header));
	}

	// Returns false when a write failed; the capture is then unusable.
	[[nodiscard]] bool Close() {
		if (m_file == nullptr) {
			return !m_failed;
		}
		const bool closed = std::fclose(m_file) == 0;
		m_file            = nullptr;
		return closed && !m_failed;
	}

	[[nodiscard]] bool IsOpen() const { return m_file != nullptr; }
	[[nodiscard]] bool Failed() const { return m_failed; }

	void Write(RecordType type, std::span<const uint8_t> payload = {}) {
		const RecordHeader header {static_cast<uint32_t>(type), 0, payload.size()};
		(void)WriteRaw(&header, sizeof(header));
		(void)WriteRaw(payload.data(), payload.size());
	}

	template <typename T>
	void WriteStruct(RecordType type, const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Write(type, {reinterpret_cast<const uint8_t*>(&value), sizeof(T)});
	}

	void WriteBytes(uint64_t address, std::span<const uint8_t> bytes, bool had_blocked) {
		const RecordHeader header {static_cast<uint32_t>(RecordType::WaitBytes), 0,
		                           sizeof(BytesHeader) + bytes.size()};
		const BytesHeader  bytes_header {address, static_cast<uint32_t>(bytes.size()),
		                                 had_blocked ? 1u : 0u};
		(void)WriteRaw(&header, sizeof(header));
		(void)WriteRaw(&bytes_header, sizeof(bytes_header));
		(void)WriteRaw(bytes.data(), bytes.size());
	}

	// One `Pages` record. `addresses[i]` is the guest address of the page whose bytes are
	// `data[i * PageSize ...]`.
	void WritePages(std::span<const uint64_t> addresses, std::span<const uint8_t> data) {
		if (addresses.empty() || data.size() != addresses.size() * PageSize) {
			m_failed |= !addresses.empty();
			return;
		}
		const RecordHeader header {static_cast<uint32_t>(RecordType::Pages), 0,
		                           addresses.size() * (sizeof(PageHeader) + PageSize)};
		(void)WriteRaw(&header, sizeof(header));
		for (size_t i = 0; i < addresses.size(); i++) {
			const PageHeader page {addresses[i]};
			(void)WriteRaw(&page, sizeof(page));
			(void)WriteRaw(data.data() + i * PageSize, PageSize);
		}
	}

private:
	bool WriteRaw(const void* data, size_t size) {
		if (m_file == nullptr || m_failed) {
			return false;
		}
		if (size != 0 && std::fwrite(data, 1, size, m_file) != size) {
			m_failed = true;
			return false;
		}
		return true;
	}

	std::FILE* m_file   = nullptr;
	bool       m_failed = false;
};

class Reader final {
public:
	struct Record {
		RecordType           type = RecordType::FrameEnd;
		std::vector<uint8_t> payload;

		// The payload as `T`. False when the record is shorter than `T`.
		template <typename T>
		[[nodiscard]] bool As(T& out) const {
			static_assert(std::is_trivially_copyable_v<T>);
			if (payload.size() < sizeof(T)) {
				return false;
			}
			std::memcpy(&out, payload.data(), sizeof(T));
			return true;
		}
	};

	Reader() = default;
	~Reader() { Close(); }
	Reader(const Reader&)            = delete;
	Reader& operator=(const Reader&) = delete;

	// False when the file is missing or is not a capture of this version.
	[[nodiscard]] bool Open(const std::filesystem::path& path) {
		Close();
		m_file = std::fopen(path.string().c_str(), "rb");
		if (m_file == nullptr) {
			return false;
		}
		FileHeader header;
		if (std::fread(&header, 1, sizeof(header), m_file) != sizeof(header) ||
		    header.magic != FileMagic || header.version != FileVersion ||
		    header.page_size != PageSize) {
			Close();
			return false;
		}
		return true;
	}

	void Close() {
		if (m_file != nullptr) {
			(void)std::fclose(m_file);
			m_file = nullptr;
		}
	}

	// Reads the next record. False at the end of the file. `Truncated()` tells a clean end from a
	// record that was cut short.
	[[nodiscard]] bool Next(Record& record) {
		if (m_file == nullptr || m_truncated) {
			return false;
		}
		RecordHeader header;
		const size_t got = std::fread(&header, 1, sizeof(header), m_file);
		if (got == 0) {
			return false;
		}
		if (got != sizeof(header)) {
			m_truncated = true;
			return false;
		}
		record.type = static_cast<RecordType>(header.type);
		record.payload.resize(header.size);
		if (header.size != 0 &&
		    std::fread(record.payload.data(), 1, header.size, m_file) != header.size) {
			m_truncated = true;
			return false;
		}
		return true;
	}

	[[nodiscard]] bool Truncated() const { return m_truncated; }

	// Calls `func(address, bytes)` for each page of a `Pages` record. False when the payload is
	// not a whole number of pages.
	template <typename Func>
	[[nodiscard]] static bool ForEachPage(const Record& record, Func&& func) {
		constexpr size_t entry = sizeof(PageHeader) + PageSize;
		if (record.type != RecordType::Pages || record.payload.size() % entry != 0) {
			return false;
		}
		for (size_t offset = 0; offset < record.payload.size(); offset += entry) {
			PageHeader page;
			std::memcpy(&page, record.payload.data() + offset, sizeof(page));
			func(page.address, std::span<const uint8_t> {
			                       record.payload.data() + offset + sizeof(PageHeader), PageSize});
		}
		return true;
	}

	// Splits a `WaitBytes` record. False when the payload does not match its header.
	[[nodiscard]] static bool Bytes(const Record& record, uint64_t& address,
	                                std::span<const uint8_t>& bytes, bool& had_blocked) {
		BytesHeader header;
		if (record.type != RecordType::WaitBytes || !record.As(header) ||
		    record.payload.size() != sizeof(header) + header.size) {
			return false;
		}
		address     = header.address;
		had_blocked = header.had_blocked != 0;
		bytes       = {record.payload.data() + sizeof(header), static_cast<size_t>(header.size)};
		return true;
	}

private:
	std::FILE* m_file      = nullptr;
	bool       m_truncated = false;
};

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_
