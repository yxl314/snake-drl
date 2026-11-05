#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "detail/thread_pool.hpp"
#include "test_framework.hpp"

void run_thread_pool_tests() {
  constexpr int items = 1000;

  snake::detail::ThreadPool serial(1, items);
  CHECK(!serial.parallel());
  CHECK(serial.total_threads() == 1);

  std::vector<int> counts(static_cast<std::size_t>(items), 0);
  auto count_task = [&counts](int begin, int end) noexcept {
    for (int index = begin; index < end; ++index) {
      counts[static_cast<std::size_t>(index)] += 1;
    }
  };
  serial.parallel_for(count_task);
  for (int value : counts) {
    CHECK(value == 1);
  }

  snake::detail::ThreadPool parallel(4, items);
  CHECK(parallel.parallel());
  CHECK(parallel.total_threads() == 4);

  for (int repetition = 0; repetition < 100; ++repetition) {
    std::fill(counts.begin(), counts.end(), 0);
    parallel.parallel_for(count_task);
    for (int value : counts) {
      CHECK(value == 1);
    }
  }

  snake::detail::ThreadPool clamped(8, 3);
  CHECK(clamped.total_threads() == 3);
  std::vector<int> small(3, 0);
  auto small_task = [&small](int begin, int end) noexcept {
    for (int index = begin; index < end; ++index) {
      small[static_cast<std::size_t>(index)] += 1;
    }
  };
  clamped.parallel_for(small_task);
  CHECK(small[0] == 1);
  CHECK(small[1] == 1);
  CHECK(small[2] == 1);

  // Regression guard for the worker startup handshake: construct a pool and
  // dispatch immediately, which is the window where the "seen" baseline used
  // to be established after publishing readiness.
  for (int repetition = 0; repetition < 500; ++repetition) {
    snake::detail::ThreadPool fresh(4, 64);
    std::atomic<int> hits{0};
    fresh.parallel_for([&hits](int begin, int end) noexcept {
      for (int index = begin; index < end; ++index) {
        hits.fetch_add(1, std::memory_order_relaxed);
      }
    });
    CHECK(hits.load() == 64);
  }

  // Exercise the idle backoff: workers reach the sleep branch, then must wake
  // for a new dispatch. Also destroys a pool whose workers are sleeping.
  for (int repetition = 0; repetition < 5; ++repetition) {
    snake::detail::ThreadPool idle(4, items);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    std::fill(counts.begin(), counts.end(), 0);
    idle.parallel_for(count_task);
    for (int value : counts) {
      CHECK(value == 1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
}
