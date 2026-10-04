#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/exposedGpuBytes.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/writeWatchSet.h"

#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// A CPU write faulted at `fault_vaddr`. Unprotects the whole aligned window around it when no
	// page in the window holds GPU-modified data, so a run of writes costs one fault, not one per
	// page. Pages the CPU did not write count as written, and the next GPU use uploads them.
	// Returns the range it invalidated: a window around the fault, or the faulting byte.
	GuestRange InvalidateWrittenMemory(uint64_t fault_vaddr, bool window_is_mapped);
	// Forgets every byte only the host GPU holds, without reading it back: guest memory is the
	// current content from now on. For the capture player, which puts guest memory back to an
	// earlier state. GPU thread, with no host work in flight.
	void DiscardGpuWrites();
	// The renderer copied bytes the host GPU held to guest memory: tells whoever observes the
	// guest GPU. Any thread.
	void NoteHostWrite(uint64_t vaddr, uint64_t size);
	// The guest ranges that have a cached buffer, and the way to have one for each of them
	// again. Which buffers exist decides what is brought up to date before a shader that reads
	// through addresses, so a replay needs the buffers the live run had. GPU thread.
	[[nodiscard]] std::vector<GuestRange> SaveBufferRanges() const;
	void                                  EnsureBuffers(std::span<const GuestRange> ranges);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	// The GDS bytes, for the capture of the guest GPU stream. GPU thread, with no GPU work
	// pending. `RestoreGds` takes exactly `SaveGds().size()` bytes.
	[[nodiscard]] std::span<const uint8_t> SaveGds() const noexcept {
		return m_gds_buffer.Mapped();
	}
	void                  RestoreGds(std::span<const uint8_t> bytes);
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Moves into `ranges` the guest ranges that gained bytes the GPU has not seen since the last
	// call: CPU write faults and new buffers. False when the log overflowed or a full pass was
	// requested; the caller must then synchronize every buffer.
	[[nodiscard]] bool TakeCpuWrites(std::vector<GuestRange>& ranges);
	void               RequestFullSynchronization();
	// Every CPU or GPU write that touches a watched range gives it a new generation. Writers of
	// guest memory that do not go through the buffer cache, such as image downloads, notify it
	// themselves.
	[[nodiscard]] WriteWatchSet& WriteWatches() noexcept { return m_write_watches; }
	void               ProcessFaultBuffer();
	// Between Begin and End, uploads share one barrier before the first copy and one after the
	// last. A barrier between two copies makes the Metal backend start a new blit encoder for
	// each copy; a run of copies with nothing in between shares one. No draw or dispatch may be
	// recorded inside a batch.
	void               BeginUploadBatch();
	void               EndUploadBatch();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	static constexpr size_t  MaxCpuWriteLog = 4096;
	// Records a range for TakeCpuWrites.
	void RecordCpuWrite(uint64_t vaddr, uint64_t size);
	// Guest memory is about to be copied to the host: tells whoever observes the guest GPU.
	void NoteGuestRead(uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Records the copy of `copies` from `buffer` to guest memory. The bytes arrive when the host
	// GPU has finished the commands recorded so far; nobody waits here. `exposed`: the ranges
	// are exposed GPU bytes, which need to know the value that arrived.
	void RecordDownload(Buffer& buffer, std::vector<vk::BufferCopy> copies, uint64_t total_size,
	                    bool exposed);
	// The guest touched `size` bytes at `vaddr`, in a page the GPU owns. When the GPU's bytes in
	// that page are few and the guest touched none of them, gives the page back to the guest
	// without waiting for the GPU: see `ExposedGpuBytes`. False when the page has to be read
	// back the usual way.
	[[nodiscard]] bool TryExposePage(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	ExposedGpuBytes                                    m_exposed;
	MemoryTracker                                     m_memory_tracker;
	WriteWatchSet                                      m_write_watches;
	// See TakeCpuWrites. Written by the fault thread and the GPU thread.
	bool                                              m_upload_batch_open    = false;
	bool                                              m_upload_batch_started = false;
	std::mutex                                        m_cpu_write_log_mutex;
	std::vector<GuestRange>                           m_cpu_write_log;
	bool                                              m_cpu_writes_need_full_pass = true;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
