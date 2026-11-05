#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "detail/free_cells.hpp"
#include "detail/grid_observation.hpp"
#include "detail/rng.hpp"
#include "detail/thread_pool.hpp"
#include "snake/config.hpp"
#include "snake/view.hpp"

namespace snake::detail {

// Test-only access hook. The implementation lives in tests/support and is only
// linked into snake_core_tests. The friend declaration is unconditional on
// purpose: conditionally compiling it would make the class definition differ
// between translation units and violate the ODR.
struct VecSnakeTestAccess;

// Explicit index conversion for the many int-indexed SoA arrays. Keeping the
// cast in one place makes -Wsign-conversion meaningful without littering the
// hot path with static_cast<std::size_t>.
constexpr std::size_t idx(int value) noexcept {
  return static_cast<std::size_t>(value);
}

class VecSnakeImpl {
 public:
  explicit VecSnakeImpl(const VecConfig& config);

  ~VecSnakeImpl() = default;

  VecSnakeImpl(const VecSnakeImpl&) = delete;
  VecSnakeImpl& operator=(const VecSnakeImpl&) = delete;
  VecSnakeImpl(VecSnakeImpl&&) = delete;
  VecSnakeImpl& operator=(VecSnakeImpl&&) = delete;

  void step(const std::int8_t* actions) noexcept;
  void reset_all() noexcept;
  void reset_mask(const std::uint8_t* mask) noexcept;
  void reseed_all(std::uint64_t seed) noexcept;
  void reseed_env(int env, std::uint64_t seed);

  const VecConfig& config() const noexcept { return config_; }
  int num_envs() const noexcept { return n_; }
  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }
  int cells() const noexcept { return cells_; }
  int active_threads() const noexcept { return pool_->total_threads(); }

  [[nodiscard]] GridView observations() const noexcept {
    return GridView(obs_.empty() ? nullptr : obs_.data(), n_, height_, width_);
  }
  [[nodiscard]] GridView final_observations() const noexcept {
    return GridView(final_obs_.empty() ? nullptr : final_obs_.data(), n_, height_, width_);
  }
  [[nodiscard]] BoardView bitboards() const noexcept {
    return BoardView(rows_.empty() ? nullptr : rows_.data(), n_, height_);
  }
  [[nodiscard]] BoardView final_bitboards() const noexcept {
    return BoardView(final_rows_.empty() ? nullptr : final_rows_.data(), n_, height_);
  }

  [[nodiscard]] CellView head_cells() const noexcept { return CellView(head_.data(), n_); }
  [[nodiscard]] CellView food_cells() const noexcept { return CellView(food_.data(), n_); }
  [[nodiscard]] CellView final_head_cells() const noexcept {
    return CellView(final_head_.empty() ? nullptr : final_head_.data(), n_);
  }
  [[nodiscard]] CellView final_food_cells() const noexcept {
    return CellView(final_food_.empty() ? nullptr : final_food_.data(), n_);
  }

  [[nodiscard]] FloatView rewards() const noexcept { return FloatView(rewards_.data(), n_); }
  [[nodiscard]] FloatView returns() const noexcept { return FloatView(returns_.data(), n_); }
  [[nodiscard]] FloatView final_returns() const noexcept {
    return FloatView(final_returns_.data(), n_);
  }
  [[nodiscard]] Int32View scores() const noexcept { return Int32View(score_.data(), n_); }
  [[nodiscard]] Int32View lengths() const noexcept { return Int32View(len_.data(), n_); }
  [[nodiscard]] Int32View steps() const noexcept { return Int32View(steps_.data(), n_); }
  [[nodiscard]] Int32View final_scores() const noexcept {
    return Int32View(final_score_.data(), n_);
  }
  [[nodiscard]] Int32View final_lengths() const noexcept {
    return Int32View(final_len_.data(), n_);
  }

  [[nodiscard]] ByteView dones() const noexcept { return ByteView(done_.data(), n_); }
  [[nodiscard]] ByteView terminated() const noexcept { return ByteView(terminated_.data(), n_); }
  [[nodiscard]] ByteView truncated() const noexcept { return ByteView(truncated_.data(), n_); }
  [[nodiscard]] ByteView wins() const noexcept { return ByteView(won_.data(), n_); }
  [[nodiscard]] ByteView needs_reset() const noexcept { return ByteView(needs_reset_.data(), n_); }
  [[nodiscard]] ByteView directions() const noexcept { return ByteView(dir_.data(), n_); }

  [[nodiscard]] bool check_invariants(int env) const;
  [[nodiscard]] bool check_all_invariants() const;

  friend struct VecSnakeTestAccess;

 private:
  void init_env(int env) noexcept;
  void reset_env(int env) noexcept;
  void reset_counters(int env) noexcept;
  void place_initial_snake(int env) noexcept;
  void spawn_food(int env) noexcept;
  void capture_final(int env) noexcept;
  void finish_step(int env, float reward, bool terminated, bool truncated, bool won) noexcept;
  void step_one(int env, int action) noexcept;

  int resolve_direction(int env, int action) const noexcept;
  void clear_rows(int env) noexcept;

  std::uint16_t* body_slice(int env) noexcept { return body_.data() + idx(env) * idx(cells_); }
  const std::uint16_t* body_slice(int env) const noexcept {
    return body_.data() + idx(env) * idx(cells_);
  }
  std::uint64_t* row_slice(int env) noexcept { return rows_.data() + idx(env) * idx(height_); }
  const std::uint64_t* row_slice(int env) const noexcept {
    return rows_.data() + idx(env) * idx(height_);
  }
  std::uint8_t* obs_slice(int env) noexcept { return obs_.data() + idx(env) * 3 * idx(cells_); }
  const std::uint8_t* obs_slice(int env) const noexcept {
    return obs_.data() + idx(env) * 3 * idx(cells_);
  }

  FreeCellSet free_cells(int env) noexcept {
    const std::size_t offset = idx(env) * idx(cells_);
    return FreeCellSet(free_.data() + offset, free_pos_.data() + offset, &free_count_[idx(env)],
                       cells_, width_);
  }

  GridObservation grid_observation(int env) noexcept {
    return GridObservation(obs_.empty() ? nullptr : obs_slice(env), width_, height_);
  }

  VecConfig config_;
  int n_ = 0;
  int width_ = 0;
  int height_ = 0;
  int cells_ = 0;

  // Hot per-environment state, split into contiguous slices.
  std::vector<std::uint64_t> rows_;          // n * height
  std::vector<std::uint16_t> body_;          // n * cells
  std::vector<std::uint16_t> free_;          // n * cells
  std::vector<std::uint16_t> free_pos_;      // n * cells
  std::vector<std::uint16_t> head_;          // n
  std::vector<std::uint16_t> food_;          // n
  std::vector<std::uint16_t> head_idx_;      // n
  std::vector<std::uint16_t> tail_idx_;      // n
  std::vector<std::int32_t> len_;            // n
  std::vector<std::int32_t> score_;          // n
  std::vector<std::int32_t> steps_;          // n
  std::vector<std::int32_t> steps_no_food_;  // n
  std::vector<std::int32_t> free_count_;     // n
  std::vector<std::uint8_t> dir_;            // n
  std::vector<std::uint8_t> done_;           // n
  std::vector<std::uint8_t> terminated_;     // n
  std::vector<std::uint8_t> truncated_;      // n
  std::vector<std::uint8_t> won_;            // n
  std::vector<std::uint8_t> needs_reset_;    // n
  std::vector<Xoshiro256> rng_;              // n

  std::vector<float> rewards_;         // n
  std::vector<float> returns_;         // n (output)
  std::vector<float> final_returns_;   // n (output, return of the terminal step)
  std::vector<float> episode_return_;  // n (internal accumulation)

  // Dense observation buffers, allocated only when requested.
  std::vector<std::uint8_t> obs_;          // n * 3 * cells
  std::vector<std::uint8_t> final_obs_;    // n * 3 * cells
  std::vector<std::uint64_t> final_rows_;  // n * height
  std::vector<std::uint16_t> final_head_;  // n
  std::vector<std::uint16_t> final_food_;  // n
  std::vector<std::int32_t> final_score_;  // n
  std::vector<std::int32_t> final_len_;    // n

  // Created after all state is initialized, so workers never observe a
  // half-initialized engine.
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace snake::detail
