#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "detail/engine.hpp"

namespace snake::detail {

// Test-only entry point for constructing arbitrary engine states.
//
// It is declared in engine.hpp as a friend (unconditionally, so the class
// definition is identical in every translation unit) but its implementation is
// only linked into snake_core_tests, never into the shipped library.
struct VecSnakeTestAccess {
  // Places one snake on environment 0. `cells` is given tail -> head; the ring
  // indices are derived from that order. All derived counters are kept
  // consistent so check_invariants() can be used immediately afterwards.
  static void place_snake(VecSnakeImpl& env, const std::vector<std::pair<int, int>>& cells,
                          std::uint8_t dir, std::uint16_t food, int steps = 0);

  // 3x3 board whose snake is one cell away from eating the final food.
  static void force_near_win(VecSnakeImpl& env);

  // Moves the current food to `cell`, keeping the free list and Grid obs in
  // sync. `cell` must be free. The overload targets a specific environment.
  static void force_food(VecSnakeImpl& env, std::uint16_t cell);
  static void force_food(VecSnakeImpl& env, int env_index, std::uint16_t cell);

  // Deliberate corruption of environment 0, used to prove that
  // check_invariants() actually rejects broken states.
  static void corrupt_duplicate_body(VecSnakeImpl& env);
  static void corrupt_free_count(VecSnakeImpl& env);
  static void corrupt_row_bitmap(VecSnakeImpl& env);
  static void corrupt_observation(VecSnakeImpl& env);
  static void corrupt_score(VecSnakeImpl& env);
  static void corrupt_head_index(VecSnakeImpl& env);
  static void corrupt_head_cell(VecSnakeImpl& env);
};

}  // namespace snake::detail
