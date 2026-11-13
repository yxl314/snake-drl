#include "test_access.hpp"

#include <cstddef>
#include <cstring>

#include "detail/bitboard.hpp"
#include "test_framework.hpp"

namespace snake::detail {

void VecSnakeTestAccess::place_snake(VecSnakeImpl& env,
                                     const std::vector<std::pair<int, int>>& cells,
                                     std::uint8_t dir, std::uint16_t food, int steps) {
  CHECK(env.num_envs() == 1);
  const int length = static_cast<int>(cells.size());
  CHECK(length >= 1 && length <= env.cells());

  auto free = env.free_cells(0);
  free.initialize_full(env.height());
  env.clear_rows(0);

  auto grid = env.grid_observation(0);
  if (!env.obs_.empty()) {
    std::memset(env.obs_.data(), 0,
                static_cast<std::size_t>(3) * static_cast<std::size_t>(env.cells()));
  }

  for (int part = 0; part < length; ++part) {
    const auto& coord = cells[static_cast<std::size_t>(part)];
    const std::uint16_t cell = pack_cell(coord.first, coord.second);
    env.body_[static_cast<std::size_t>(part)] = cell;
    row_set(env.rows_[static_cast<std::size_t>(coord.second)], coord.first);
    free.remove(cell);
    grid.set_body(cell, 1);
  }

  env.head_idx_[0] = static_cast<std::uint16_t>(length - 1);
  env.tail_idx_[0] = 0;
  env.len_[0] = length;
  env.score_[0] = length - env.config().initial_length;
  env.steps_[0] = steps;
  env.steps_no_food_[0] = steps;
  env.dir_[0] = dir;
  env.head_[0] = env.body_[static_cast<std::size_t>(length - 1)];
  env.food_[0] = food;
  env.done_[0] = 0;
  env.terminated_[0] = 0;
  env.truncated_[0] = 0;
  env.won_[0] = 0;
  env.needs_reset_[0] = 0;
  env.rewards_[0] = 0.0F;
  env.returns_[0] = 0.0F;
  env.episode_return_[0] = 0.0F;

  grid.set_head(env.head_[0], 1);
  if (food != kInvalidCell) {
    free.remove(food);
    grid.set_food(food, 1);
  }
  CHECK(food != kInvalidCell || length == env.cells());
  CHECK(free.size() == env.cells() - length - (food != kInvalidCell ? 1 : 0));
}

void VecSnakeTestAccess::force_near_win(VecSnakeImpl& env) {
  CHECK(env.width() == 3);
  CHECK(env.height() == 3);
  const std::vector<std::pair<int, int>> body = {
      {0, 0}, {1, 0}, {2, 0}, {2, 1}, {1, 1}, {0, 1}, {0, 2}, {1, 2},
  };
  place_snake(env, body, 1, pack_cell(2, 2), 10);
}

void VecSnakeTestAccess::force_food(VecSnakeImpl& env, std::uint16_t cell) {
  force_food(env, 0, cell);
}

void VecSnakeTestAccess::force_food(VecSnakeImpl& env, int env_index, std::uint16_t cell) {
  auto free = env.free_cells(env_index);
  const auto index = static_cast<std::size_t>(env_index);
  if (env.food_[index] != kInvalidCell) {
    free.add(env.food_[index]);
  }
  free.remove(cell);
  const std::uint16_t old_food = env.food_[index];
  env.food_[index] = cell;
  if (!env.obs_.empty()) {
    auto grid = env.grid_observation(env_index);
    if (old_food != kInvalidCell) {
      grid.set_food(old_food, 0);
    }
    grid.set_food(cell, 1);
  }
}

void VecSnakeTestAccess::corrupt_duplicate_body(VecSnakeImpl& env) { env.body_[1] = env.body_[0]; }

void VecSnakeTestAccess::corrupt_free_count(VecSnakeImpl& env) { ++env.free_count_[0]; }

void VecSnakeTestAccess::corrupt_row_bitmap(VecSnakeImpl& env) {
  row_clear(env.rows_[static_cast<std::size_t>(cell_y(env.head_[0]))], cell_x(env.head_[0]));
}

void VecSnakeTestAccess::corrupt_observation(VecSnakeImpl& env) {
  CHECK(!env.obs_.empty());
  env.obs_[0] ^= 1;
}

void VecSnakeTestAccess::corrupt_score(VecSnakeImpl& env) { ++env.score_[0]; }

void VecSnakeTestAccess::corrupt_head_index(VecSnakeImpl& env) {
  env.head_idx_[0] = static_cast<std::uint16_t>((env.head_idx_[0] + 1) % env.cells_);
}

void VecSnakeTestAccess::corrupt_head_cell(VecSnakeImpl& env) { env.head_[0] = env.food_[0]; }

}  // namespace snake::detail
