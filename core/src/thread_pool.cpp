#include "detail/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace snake::detail {
namespace {

inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __yield();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
ThreadPool::ThreadPool(int total_threads, int total_items)
    : total_items_(std::max(1, total_items)) {
  total_threads_ = std::clamp(total_threads, 1, total_items_);
  chunk_begin_.resize(static_cast<std::size_t>(total_threads_));
  chunk_end_.resize(static_cast<std::size_t>(total_threads_));
  for (int thread = 0; thread < total_threads_; ++thread) {
    chunk_begin_[static_cast<std::size_t>(thread)] =
        static_cast<int>((static_cast<std::int64_t>(total_items_) * thread) / total_threads_);
    chunk_end_[static_cast<std::size_t>(thread)] =
        static_cast<int>((static_cast<std::int64_t>(total_items_) * (thread + 1)) / total_threads_);
  }

  if (total_threads_ > 1) {
    start_workers();
  }
}

ThreadPool::~ThreadPool() noexcept { stop_workers(); }

void ThreadPool::start_workers() {
  const int count = total_threads_ - 1;
  workers_.reserve(static_cast<std::size_t>(count));
  try {
    for (int worker = 0; worker < count; ++worker) {
      workers_.emplace_back([this, worker] { worker_main(worker); });
    }
  } catch (...) {
    stop_workers();
    throw;
  }
  wait_for_workers_ready();
}

void ThreadPool::stop_workers() noexcept {
  if (workers_.empty()) {
    return;
  }
  stop_.store(true, std::memory_order_relaxed);
  task_generation_.fetch_add(1, std::memory_order_release);
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      // join() is only called for joinable threads. A failure here would mean
      // resource exhaustion in the OS, which noexcept leaves as terminate().
      worker.join();
    }
  }
  workers_.clear();
}

void ThreadPool::wait_for_workers_ready() noexcept {
  while (workers_ready_.load(std::memory_order_acquire) != worker_count()) {
    cpu_relax();
  }
}

void ThreadPool::wait_for_completion() noexcept {
  while (tasks_done_.load(std::memory_order_acquire) != worker_count()) {
    cpu_relax();
  }
}

void ThreadPool::run_chunk(int chunk) noexcept {
  task_fn_(task_context_, chunk_begin_[static_cast<std::size_t>(chunk)],
           chunk_end_[static_cast<std::size_t>(chunk)]);
}

void ThreadPool::worker_main(int worker_index) noexcept {
  const int chunk = worker_index + 1;
  // Establish the baseline BEFORE publishing readiness. If the owner were able
  // to dispatch the first task between these two statements, this worker would
  // adopt the new generation as "already seen" and never run it, leaving the
  // owner spinning forever in wait_for_completion().
  std::uint64_t seen = task_generation_.load(std::memory_order_acquire);
  workers_ready_.fetch_add(1, std::memory_order_release);

  while (true) {
    std::uint64_t generation = task_generation_.load(std::memory_order_acquire);
    int idle = 0;
    constexpr int kSpinIterations = 8192;
    constexpr int kYieldIterations = 2048;

    while (generation == seen) {
      if (stop_.load(std::memory_order_relaxed)) {
        return;
      }
      if (idle < kSpinIterations) {
        cpu_relax();
      } else if (idle < kSpinIterations + kYieldIterations) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      ++idle;
      generation = task_generation_.load(std::memory_order_acquire);
    }

    if (stop_.load(std::memory_order_relaxed)) {
      return;
    }

    seen = generation;
    run_chunk(chunk);
    tasks_done_.fetch_add(1, std::memory_order_release);
  }
}

}  // namespace snake::detail
