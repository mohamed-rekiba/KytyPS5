#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_

// Container for a capture of the guest GPU stream: a header, then records in the order the GPU
// thread saw them. Each record is a type, a payload size and the payload. The reader hands out
// one record at a time, so a capture larger than host memory can be replayed.
//
// This file knows nothing about the renderer. The recorder and the player give the records their
// meaning.
//
// Some records hold the emulator's own structures as raw bytes. The header carries the layout of
// those structures (`StateLayout`), so a build with another layout refuses the file instead of
// misreading it.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <zstd.h>

namespace Libs::Graphics::Capture {

inline constexpr std::array<char, 8> FileMagic   = {'K', 'G', 'P', 'U', 'C', 'A', 'P', '\0'};
inline constexpr uint32_t            FileVersion = 2;
inline constexpr uint64_t            PageSize    = 4096;

enum class RecordType : uint32_t {
	// Guest ranges visible to the GPU at capture start.
	MapRange = 1,
	// Whole guest pages: `PageHeader`, then `PageSize` bytes, repeated to the end of the record.
	// Before `FrameBegin`: the memory at capture start. After it: pages that changed before the
	// submission, the wait or the frame end that follows. The file stores them compressed
	// (`Compressed`); the reader hands them out as `Pages`.
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
	// Guest ranges that were written before the submission, the wait or the frame end that
	// follows, by the guest (and a cache of the renderer was told) or by the renderer itself:
	// `WriteRange`, repeated to the end of the record.
	GuestWrites = 15,
	// Clear state of depth and colour metadata at capture start, which no guest memory holds:
	// `SurfaceMeta`, repeated to the end of the record.
	SurfaceMetas = 16,
	// Guest ranges that had a buffer in the renderer's cache at capture start: `Range`, repeated
	// to the end of the record.
	Buffers = 19,
	// The guest mapped or unmapped GPU-visible memory inside the captured frames: every range
	// that is mapped from here on, as `Range`, repeated to the end of the record.
	Mapping = 21,
	// The renderer copied guest memory to the host, and pages of it had changed since the last
	// stored state: `ReadPoint`. The `Pages` in front of it are those pages as it read them.
	ReadPoint = 20,
	// The images the host GPU had written, after a number of draws and dispatches of the capture:
	// `ImageChecksHeader`, then `ImageCheck`, repeated to the end of the record. A debugging aid:
	// the player takes the same check and names the images that differ.
	ImageChecks = 23,
	// What the renderer handed to the host GPU for the frame that ends next: `FrameWork`.
	Work = 18,
	// The payload of another record, compressed: `CompressedHeader`, then one zstd frame. Only
	// in the file: the reader hands out the record it holds.
	Compressed = 17,
	// The bytes of the display buffer the frame flipped, after the GPU's picture was written
	// back. Stored compressed. With them the player can tell how far a replayed picture is from
	// the live one, not only that it differs.
	PictureBytes = 22,
	// Last record of a complete capture. A file without it was cut short or abandoned.
	End = 12,
};

// Layout of the emulator structures a capture stores as raw bytes. The recorder and the player
// fill it in from the structures they are built with.
struct StateLayout {
	// Raised by hand when a stored structure changes meaning without changing size.
	uint32_t version                 = 0;
	uint32_t processor_state_size    = 0;
	uint32_t started_submission_size = 0;
	uint32_t reserved                = 0;

	[[nodiscard]] bool operator==(const StateLayout&) const = default;
};

struct FileHeader {
	std::array<char, 8> magic     = FileMagic;
	uint32_t            version   = FileVersion;
	uint32_t            page_size = static_cast<uint32_t>(PageSize);
	StateLayout         layout;
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

struct CompressedHeader {
	uint64_t raw_size = 0; // size of the payload of the record it holds
	uint32_t type     = 0; // type of the record it holds
	uint32_t reserved = 0;
};

// Which cache was told about a guest write. A page fault tells the buffer cache about a whole
// window and the texture cache about the faulting byte, so the two ranges differ.
// `Host`: no cache was told, because the renderer itself wrote the bytes: it copied what the host
// GPU held to guest memory. The pages that changed that way are not a write of the guest.
enum class WriteTarget : uint32_t { Buffers = 1, Images = 2, Both = 3, Host = 4 };

struct WriteRange {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint32_t target  = 0;
	// When the write came: the number of draws and dispatches the renderer had handed to the
	// host GPU since the scan of guest memory before this record.
	uint32_t position = 0;
};

struct ReadPoint {
	// The number of draws and dispatches the renderer had handed to the host GPU since the
	// stored state before this one.
	uint32_t position = 0;
	uint32_t reserved = 0;
};

// `position` of a check that was taken at a frame end, the only point at which the queues of the
// live run and of a replay are known to have done the same work.
inline constexpr uint64_t FrameEndCheck = ~0ull;

struct ImageChecksHeader {
	uint64_t position = 0; // draws and dispatches since the capture start, or `FrameEndCheck`
};

struct ImageCheck {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0; // of the guest bytes, after the host picture was written back
	uint32_t width   = 0;
	uint32_t height  = 0;
	uint32_t depth   = 0;
	uint32_t format  = 0; // host format
};

struct SurfaceMeta {
	uint64_t address    = 0;
	uint32_t type       = 0;
	uint32_t clear_mask = 0;
};

struct FrameWork {
	uint64_t draws               = 0;
	uint64_t dispatches          = 0;
	uint64_t render_passes       = 0;
	uint64_t buffer_uploads      = 0;
	uint64_t buffer_upload_bytes = 0;
	uint64_t image_uploads       = 0;
	uint64_t image_upload_bytes  = 0;
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

	[[nodiscard]] bool Open(const std::filesystem::path& path, const StateLayout& layout = {}) {
		(void)Close();
		m_failed = false;
		m_file   = std::fopen(path.string().c_str(), "wb");
		if (m_file == nullptr) {
			return false;
		}
		FileHeader header;
		header.layout = layout;
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

	// The payload of a `Compressed` record that holds a record of `type` with payload `raw`.
	// Empty when the compression failed. Needs no writer, so several threads can pack side by
	// side.
	[[nodiscard]] static std::vector<uint8_t> Pack(RecordType type, std::span<const uint8_t> raw) {
		const CompressedHeader header {raw.size(), static_cast<uint32_t>(type), 0};
		std::vector<uint8_t>   packed(sizeof(header) + ZSTD_compressBound(raw.size()));
		// Level 1: the recorder stops the game while it stores, so speed comes first.
		const size_t size =
		    ZSTD_compress(packed.data() + sizeof(header), packed.size() - sizeof(header),
		                  raw.data(), raw.size(), 1);
		if (ZSTD_isError(size) != 0) {
			return {};
		}
		std::memcpy(packed.data(), &header, sizeof(header));
		packed.resize(sizeof(header) + size);
		return packed;
	}

	// The payload of one page record, ready to be written. `addresses[i]` is the guest address
	// of the page whose bytes are `data[i * PageSize ...]`. Empty when there is no page, the
	// sizes do not match, or the compression failed.
	[[nodiscard]] static std::vector<uint8_t> PackPages(std::span<const uint64_t> addresses,
	                                                    std::span<const uint8_t>  data) {
		if (addresses.empty() || data.size() != addresses.size() * PageSize) {
			return {};
		}
		std::vector<uint8_t> raw(addresses.size() * (sizeof(PageHeader) + PageSize));
		for (size_t i = 0; i < addresses.size(); i++) {
			const PageHeader page {addresses[i]};
			uint8_t*         out = raw.data() + i * (sizeof(PageHeader) + PageSize);
			std::memcpy(out, &page, sizeof(page));
			std::memcpy(out + sizeof(page), data.data() + i * PageSize, PageSize);
		}
		return Pack(RecordType::Pages, raw);
	}

	// Writes a payload made by `Pack` or `PackPages`. An empty one fails the capture.
	void WritePacked(std::span<const uint8_t> packed) {
		if (packed.empty()) {
			m_failed = true;
			return;
		}
		Write(RecordType::Compressed, packed);
	}

	// One record of `type`, stored compressed.
	void WriteCompressed(RecordType type, std::span<const uint8_t> raw) {
		WritePacked(Pack(type, raw));
	}

	// One page record; see `PackPages`. No pages: nothing is written.
	void WritePages(std::span<const uint64_t> addresses, std::span<const uint8_t> data) {
		if (!addresses.empty()) {
			WritePacked(PackPages(addresses, data));
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

	enum class OpenResult {
		Ok,
		Missing,      // the file cannot be opened
		NotACapture,  // no capture header
		OtherVersion, // a capture in another file format version
		OtherLayout,  // written by a build whose stored structures differ from `layout`
	};

	// Opens a capture written with the same file version and the same `layout`.
	[[nodiscard]] OpenResult Open(const std::filesystem::path& path,
	                              const StateLayout&           layout = {}) {
		Close();
		m_truncated = false;
		m_file      = std::fopen(path.string().c_str(), "rb");
		if (m_file == nullptr) {
			return OpenResult::Missing;
		}
		// The start of the header is the same in every version.
		FileHeader   header;
		const size_t start  = sizeof(header.magic) + sizeof(header.version);
		auto         result = OpenResult::Ok;
		if (std::fread(&header, 1, start, m_file) != start || header.magic != FileMagic) {
			result = OpenResult::NotACapture;
		} else if (header.version != FileVersion) {
			result = OpenResult::OtherVersion;
		} else if (std::fread(reinterpret_cast<uint8_t*>(&header) + start, 1,
		                      sizeof(header) - start, m_file) != sizeof(header) - start) {
			result = OpenResult::NotACapture;
		} else if (header.page_size != PageSize || !(header.layout == layout)) {
			result = OpenResult::OtherLayout;
		}
		if (result != OpenResult::Ok) {
			Close();
		}
		return result;
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
		if (record.type == RecordType::Compressed && !Unpack(record)) {
			m_damaged = true;
			return false;
		}
		return true;
	}

	[[nodiscard]] bool Truncated() const { return m_truncated; }
	// A record could not be unpacked: the file is not what the writer stored.
	[[nodiscard]] bool Damaged() const { return m_damaged; }

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

	// Calls `func(item)` for each `T` of a record that is a list of `T`. False when the payload
	// is not a whole number of them.
	template <typename T, typename Func>
	[[nodiscard]] static bool ForEach(const Record& record, Func&& func) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (record.payload.size() % sizeof(T) != 0) {
			return false;
		}
		for (size_t offset = 0; offset < record.payload.size(); offset += sizeof(T)) {
			T item;
			std::memcpy(&item, record.payload.data() + offset, sizeof(T));
			func(item);
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
	// Turns a `Compressed` record into the record it holds.
	[[nodiscard]] static bool Unpack(Record& record) {
		constexpr uint64_t MaxRawSize = 1ull << 32u;
		CompressedHeader   header;
		if (!record.As(header) || header.raw_size > MaxRawSize ||
		    header.type == static_cast<uint32_t>(RecordType::Compressed)) {
			return false;
		}
		std::vector<uint8_t> raw(header.raw_size);
		const size_t         size =
		    ZSTD_decompress(raw.data(), raw.size(), record.payload.data() + sizeof(header),
		                    record.payload.size() - sizeof(header));
		if (ZSTD_isError(size) != 0 || size != raw.size()) {
			return false;
		}
		record.type    = static_cast<RecordType>(header.type);
		record.payload = std::move(raw);
		return true;
	}

	std::FILE* m_file      = nullptr;
	bool       m_truncated = false;
	bool       m_damaged   = false;
};

} // namespace Libs::Graphics::Capture

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CAPTURE_CAPTUREFILE_H_
