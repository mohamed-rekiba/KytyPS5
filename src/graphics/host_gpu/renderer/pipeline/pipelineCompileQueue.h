#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_

#include "common/uniqueFunction.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Runs pipeline compiles on a fixed number of worker threads, in the order they are posted. A
// driver compile of a large shader can take minutes and several gigabytes: the bound keeps that
// many of them from running at once.
class PipelineCompileQueue {
public:
	// A posted compile: `done` is ready once it ran or was dropped, and `ran` tells which.
	struct Ticket {
		std::future<void>                  done;
		std::shared_ptr<std::atomic<bool>> ran;
	};

	explicit PipelineCompileQueue(uint32_t threads) {
		m_workers.reserve(threads);
		for (uint32_t i = 0; i < threads; i++) {
			m_workers.emplace_back([this](const std::stop_token& stop) { Run(stop); });
		}
	}
	// Drops the compiles that have not started and waits for the running ones.
	~PipelineCompileQueue() {
		CancelQueued();
		for (auto& worker: m_workers) {
			worker.request_stop();
		}
		m_ready.notify_all();
	}
	PipelineCompileQueue(const PipelineCompileQueue&)            = delete;
	PipelineCompileQueue& operator=(const PipelineCompileQueue&) = delete;

	Ticket Post(Common::UniqueFunction<void>&& task) {
		Job    job {.task = std::move(task), .ran = std::make_shared<std::atomic<bool>>(false)};
		Ticket ticket {.done = job.done.get_future(), .ran = job.ran};
		{
			std::scoped_lock lock(m_mutex);
			m_jobs.push_back(std::move(job));
		}
		m_ready.notify_one();
		return ticket;
	}

	// Completes the compiles that have not started without running them.
	void CancelQueued() {
		std::deque<Job> dropped;
		{
			std::scoped_lock lock(m_mutex);
			dropped.swap(m_jobs);
		}
		for (auto& job: dropped) {
			job.done.set_value();
		}
	}

private:
	struct Job {
		Common::UniqueFunction<void>       task;
		std::shared_ptr<std::atomic<bool>> ran;
		std::promise<void>                 done;
	};

	void Run(const std::stop_token& stop) {
		for (;;) {
			Job job;
			{
				std::unique_lock lock(m_mutex);
				if (!m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
					return;
				}
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			job.task();
			job.ran->store(true, std::memory_order_release);
			job.done.set_value();
		}
	}

	std::mutex                  m_mutex;
	std::condition_variable_any m_ready;
	std::deque<Job>             m_jobs;
	// Declared last, so the workers stop and join before the queue they read is destroyed.
	std::vector<std::jthread> m_workers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
