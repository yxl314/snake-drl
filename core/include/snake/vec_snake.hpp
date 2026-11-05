#pragma once

#include <cstdint>
#include <memory>

#include "snake/cell.hpp"
#include "snake/config.hpp"
#include "snake/view.hpp"

namespace snake {

namespace detail {
class VecSnakeImpl;
}

// Batched snake engine.
//
// The class is a thin RAII facade over an internal implementation. Views
// returned by the accessors point into buffers owned by this instance and stay
// valid for its lifetime; step()/reset*() overwrite their contents, so copy if
// you need the previous values.
//
// A single instance is not safe for concurrent step()/reset*() calls. Use one
// instance per worker thread, or rely on the internal thread pool. This also
// applies to the Python bindings, which release the GIL around step().
class VecSnake {
 public:
  explicit VecSnake(const VecConfig& config);

  ~VecSnake();

  VecSnake(const VecSnake&) = delete;
  VecSnake& operator=(const VecSnake&) = delete;
  VecSnake(VecSnake&&) = delete;
  VecSnake& operator=(VecSnake&&) = delete;

  // actions[i] is the action for environment i:
  //   0 = straight, 1 = right, 2 = left (relative mode)
  // The pointer must reference at least num_envs() elements.
  void step(const std::int8_t* actions) noexcept;

  void reset_all() noexcept;
  // mask[i] != 0 resets environment i; the pointer must reference num_envs().
  void reset_mask(const std::uint8_t* mask) noexcept;
  // Reseeds RNG streams and rebuilds environments from scratch. reseed_all(seed)
  // gives environment i the seed `seed + i` (matching construction);
  // reseed_env(i, seed) gives environment i exactly `seed` and rebuilds only
  // that environment. Both leave the environment indistinguishable from a
  // freshly constructed instance with the same seed and clear all final_*
  // buffers.
  void reseed_all(std::uint64_t seed) noexcept;
  void reseed_env(int env, std::uint64_t seed);

  // The configuration captured at construction. Reseeding changes RNG state but
  // not this snapshot.
  const VecConfig& config() const noexcept;
  int num_envs() const noexcept;
  int width() const noexcept;
  int height() const noexcept;
  int cells() const noexcept;
  // Total threads including the caller; 1 means serial.
  int active_threads() const noexcept;

  // final_* views are meaningful only for environments whose latest dones()
  // value is 1. They are not cleared by a manual reset. An empty view means the
  // buffer does not exist for the configured observation/autoreset mode:
  //   final_observations:                        Grid + SameStep
  //   final_bitboards / final_head / final_food: SameStep
  //   final_scores / final_lengths / final_returns: always
  //
  // Encodings:
  //   GridView: (N, 3, H, W) uint8, channels 0=body, 1=head, 2=food.
  //   BoardView: (N, H) uint64, cell x is stored at bit (x ^ 7).
  //   CellView: packed cells, decode with snake::cell_x()/cell_y().
  [[nodiscard]] GridView observations() const noexcept;
  [[nodiscard]] GridView final_observations() const noexcept;
  [[nodiscard]] BoardView bitboards() const noexcept;
  [[nodiscard]] BoardView final_bitboards() const noexcept;

  [[nodiscard]] CellView head_cells() const noexcept;
  [[nodiscard]] CellView food_cells() const noexcept;
  [[nodiscard]] CellView final_head_cells() const noexcept;
  [[nodiscard]] CellView final_food_cells() const noexcept;

  [[nodiscard]] FloatView rewards() const noexcept;
  [[nodiscard]] FloatView returns() const noexcept;
  [[nodiscard]] FloatView final_returns() const noexcept;
  [[nodiscard]] Int32View scores() const noexcept;
  [[nodiscard]] Int32View lengths() const noexcept;
  [[nodiscard]] Int32View steps() const noexcept;
  [[nodiscard]] Int32View final_scores() const noexcept;
  [[nodiscard]] Int32View final_lengths() const noexcept;

  // Current heading per environment: 0 = up, 1 = right, 2 = down, 3 = left.
  [[nodiscard]] ByteView directions() const noexcept;

  [[nodiscard]] ByteView dones() const noexcept;
  [[nodiscard]] ByteView terminated() const noexcept;
  [[nodiscard]] ByteView truncated() const noexcept;
  [[nodiscard]] ByteView wins() const noexcept;
  [[nodiscard]] ByteView needs_reset() const noexcept;

  // Diagnostic check used by tests. It verifies the full invariant set of one
  // environment and may allocate temporary memory. An out-of-range index
  // returns false (it does not throw).
  [[nodiscard]] bool check_invariants(int env) const;
  [[nodiscard]] bool check_all_invariants() const;

 private:
  std::unique_ptr<detail::VecSnakeImpl> impl_;
};

}  // namespace snake
