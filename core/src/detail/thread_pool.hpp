#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace snake::detail {

// Persistent thread pool for one batch owner. The calling thread participates
// as chunk 0; workers process chunks 1..T-1. A dispatch publishes a function
// pointer and context once and then waits for all workers.
//
// Task functions must be noexcept. A single ThreadPool instance is not safe
// for concurrent parallel_for() calls.
class ThreadPool {
 public:
  // Parameters are (thread count, item count); both are meaningful ints.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  ThreadPool(int total_threads, int total_items);
  ~ThreadPool() noexcept;

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  int total_threads() const noexcept { return total_threads_; }
  bool parallel() const noexcept { return total_threads_ > 1; }

  template <typename Fn>
  void parallel_for(Fn&& fn) noexcept {
    using FnType = std::remove_reference_t<Fn>;
    static_assert(std::is_nothrow_invocable_r_v<void, FnType&, int, int>,
                  "ThreadPool tasks must be noexcept and callable as fn(begin, end)");
    assert(!dispatching_ && "parallel_for() is not reentrant");
    if (!parallel()) {
      fn(0, total_items_);
      return;
    }

    dispatching_ = true;
    task_context_ = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
    task_fn_ = [](void* context, int begin, int end) noexcept {
      (*static_cast<FnType*>(context))(begin, end);
    };

    tasks_done_.store(0, std::memory_order_relaxed);
    task_generation_.fetch_add(1, std::memory_order_release);

    // The caller processes the first chunk and then waits for the rest.
    task_fn_(task_context_, chunk_begin_[0], chunk_end_[0]);
    wait_for_completion();
    dispatching_ = false;
  }

 private:
  using TaskFn = void (*)(void*, int, int) noexcept;

  void start_workers();
  void stop_workers() noexcept;
  void worker_main(int worker_index) noexcept;
  void wait_for_workers_ready() noexcept;
  void wait_for_completion() noexcept;
  void run_chunk(int chunk) noexcept;
  int worker_count() const noexcept { return static_cast<int>(workers_.size()); }

  int total_threads_ = 1;
  int total_items_ = 1;

  std::vector<int> chunk_begin_;
  std::vector<int> chunk_end_;
  std::vector<std::thread> workers_;

  TaskFn task_fn_ = nullptr;
  void* task_context_ = nullptr;
  bool dispatching_ = false;

  alignas(64) std::atomic<std::uint64_t> task_generation_{0};
  alignas(64) std::atomic<int> tasks_done_{0};
  alignas(64) std::atomic<int> workers_ready_{0};
  alignas(64) std::atomic<bool> stop_{false};
};

}  // namespace snake::detail
