#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>

#include "detail/engine.hpp"

namespace snake::detail {
namespace {

// 四个方向的分量。std::array 与 C 数组在优化后生成同样的代码，但保留尺寸信息。
constexpr std::array<int, 4> kDx = {0, 1, 0, -1};
constexpr std::array<int, 4> kDy = {-1, 0, 1, 0};

// Step counters are public as int32. With both limits disabled a perfect policy
// can run long enough to overflow them, which would be UB, so saturate instead.
void increment_saturated(std::int32_t& value) noexcept {
  if (value < std::numeric_limits<std::int32_t>::max()) {
    ++value;
  }
}

}  // namespace

void VecSnakeImpl::init_env(int env) noexcept {
  auto free = free_cells(env);
  free.initialize_full(height_);
  clear_rows(env);
  if (!obs_.empty()) {
    std::memset(obs_slice(env), 0, 3 * idx(cells_));
  }

  reset_counters(env);
  final_score_[idx(env)] = 0;
  final_len_[idx(env)] = 0;
  final_returns_[idx(env)] = 0.0F;
  if (!final_obs_.empty()) {
    std::memset(final_obs_.data() + idx(env) * 3 * idx(cells_), 0, 3 * idx(cells_));
  }
  if (!final_rows_.empty()) {
    std::memset(final_rows_.data() + idx(env) * idx(height_), 0,
                sizeof(std::uint64_t) * idx(height_));
  }
  if (!final_head_.empty()) {
    final_head_[idx(env)] = kInvalidCell;
    final_food_[idx(env)] = kInvalidCell;
  }

  place_initial_snake(env);
  spawn_food(env);
}

void VecSnakeImpl::reset_env(int env) noexcept {
  const int old_length = len_[idx(env)];
  const int old_tail = tail_idx_[idx(env)];
  const std::uint16_t old_head = head_[idx(env)];
  const std::uint16_t old_food = food_[idx(env)];
  const std::uint16_t* body = body_slice(env);

  auto grid = grid_observation(env);
  grid.clear_environment(body, old_tail, old_length, old_head, old_food);

  auto free = free_cells(env);
  int cursor = old_tail;
  for (int moved = 0; moved < old_length; ++moved) {
    free.add(body[idx(cursor)]);
    if (++cursor == cells_) {
      cursor = 0;
    }
  }
  if (old_food != kInvalidCell) {
    free.add(old_food);
  }

  clear_rows(env);
  reset_counters(env);
  place_initial_snake(env);
  spawn_food(env);
}

void VecSnakeImpl::reset_counters(int env) noexcept {
  len_[idx(env)] = config_.initial_length;
  score_[idx(env)] = 0;
  steps_[idx(env)] = 0;
  steps_no_food_[idx(env)] = 0;
  dir_[idx(env)] = 1;
  head_[idx(env)] = 0;
  head_idx_[idx(env)] = 0;
  tail_idx_[idx(env)] = 0;
  food_[idx(env)] = kInvalidCell;
  done_[idx(env)] = 0;
  terminated_[idx(env)] = 0;
  truncated_[idx(env)] = 0;
  won_[idx(env)] = 0;
  needs_reset_[idx(env)] = 0;
  rewards_[idx(env)] = 0.0F;
  returns_[idx(env)] = 0.0F;
  episode_return_[idx(env)] = 0.0F;
}

void VecSnakeImpl::place_initial_snake(int env) noexcept {
  const int head_x = std::max(width_ / 2, config_.initial_length - 1);
  const int head_y = height_ / 2;
  std::uint16_t* body = body_slice(env);
  std::uint64_t* rows = row_slice(env);
  auto free = free_cells(env);
  auto grid = grid_observation(env);

  for (int part = 0; part < config_.initial_length; ++part) {
    const int x = head_x - (config_.initial_length - 1) + part;
    const std::uint16_t cell = pack_cell(x, head_y);
    body[idx(part)] = cell;
    free.remove(cell);
    row_set(rows[idx(head_y)], x);
    grid.set_body(cell, 1);
  }

  head_idx_[idx(env)] = static_cast<std::uint16_t>(config_.initial_length - 1);
  tail_idx_[idx(env)] = 0;
  head_[idx(env)] = body[idx(config_.initial_length - 1)];
  grid.set_head(head_[idx(env)], 1);
}

void VecSnakeImpl::spawn_food(int env) noexcept {
  auto free = free_cells(env);
  const std::uint16_t cell = free.take_random(rng_[idx(env)]);
  food_[idx(env)] = cell;
  if (cell != kInvalidCell) {
    grid_observation(env).set_food(cell, 1);
  }
}

void VecSnakeImpl::clear_rows(int env) noexcept {
  std::memset(row_slice(env), 0, sizeof(std::uint64_t) * idx(height_));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int VecSnakeImpl::resolve_direction(int env, int action) const noexcept {
  const int old_direction = dir_[idx(env)];
  if (config_.relative_actions) {
    if (action == 1) {
      return (old_direction + 1) & 3;
    }
    if (action == 2) {
      return (old_direction + 3) & 3;
    }
    return old_direction;
  }

  const int absolute = action & 3;
  if (absolute == ((old_direction + 2) & 3)) {
    return old_direction;
  }
  return absolute;
}

void VecSnakeImpl::step_one(int env, int action) noexcept {
  if (config_.autoreset_mode == AutoresetMode::NextStep && needs_reset_[idx(env)] != 0) {
    reset_env(env);
    rewards_[idx(env)] = 0.0F;
    returns_[idx(env)] = 0.0F;
    terminated_[idx(env)] = 0;
    truncated_[idx(env)] = 0;
    won_[idx(env)] = 0;
    done_[idx(env)] = 0;
    return;
  }
  if (config_.autoreset_mode == AutoresetMode::Disabled && needs_reset_[idx(env)] != 0) {
    // No step happened: leave the terminal transition intact. Zeroing done /
    // terminated / won / returns here would erase the terminal signal while
    // needs_reset stays set, and would desynchronize returns from the internal
    // episode accumulation. The caller decides when to reset.
    return;
  }

  const int direction = resolve_direction(env, action);
  dir_[idx(env)] = static_cast<std::uint8_t>(direction);

  const int x = cell_x(head_[idx(env)]);
  const int y = cell_y(head_[idx(env)]);
  // direction 在上一行由 resolve_direction 保证落在 [0, 4)，这里显式转成
  // 下标类型，避免 -Wsign-conversion 把 int 到 size_type 的转换当成可疑操作。
  const auto dir_index = static_cast<std::array<int, 4>::size_type>(direction);
  const int next_x = x + kDx[dir_index];
  const int next_y = y + kDy[dir_index];

  if (static_cast<unsigned>(next_x) >= static_cast<unsigned>(width_) ||
      static_cast<unsigned>(next_y) >= static_cast<unsigned>(height_)) {
    increment_saturated(steps_[idx(env)]);
    increment_saturated(steps_no_food_[idx(env)]);
    finish_step(env, config_.step_penalty + config_.death_penalty, true, false, false);
    return;
  }

  std::uint16_t* body = body_slice(env);
  std::uint64_t* rows = row_slice(env);
  auto free = free_cells(env);
  auto grid = grid_observation(env);

  const std::uint16_t new_cell = pack_cell(next_x, next_y);
  const std::uint16_t old_head = head_[idx(env)];
  const std::uint16_t old_tail = body[tail_idx_[idx(env)]];
  const bool ate = new_cell == food_[idx(env)];
  const bool into_tail = !ate && len_[idx(env)] > 1 && new_cell == old_tail;

  if (row_occupied(rows[idx(next_y)], next_x) && !into_tail) {
    increment_saturated(steps_[idx(env)]);
    increment_saturated(steps_no_food_[idx(env)]);
    finish_step(env, config_.step_penalty + config_.death_penalty, true, false, false);
    return;
  }

  increment_saturated(steps_[idx(env)]);
  increment_saturated(steps_no_food_[idx(env)]);
  float reward = config_.step_penalty;
  bool won = false;

  if (ate) {
    reward += config_.food_reward;
    ++score_[idx(env)];
    ++len_[idx(env)];
    steps_no_food_[idx(env)] = 0;

    grid.set_food(food_[idx(env)], 0);
    food_[idx(env)] = kInvalidCell;
    if (free.empty()) {
      won = true;
    } else {
      spawn_food(env);
    }
  } else {
    if (!into_tail) {
      free.remove(new_cell);
      free.add(old_tail);
      row_clear(rows[idx(cell_y(old_tail))], cell_x(old_tail));
      grid.set_body(old_tail, 0);
    }
    int tail = tail_idx_[idx(env)] + 1;
    if (tail == cells_) {
      tail = 0;
    }
    tail_idx_[idx(env)] = static_cast<std::uint16_t>(tail);
  }

  int head_index = head_idx_[idx(env)] + 1;
  if (head_index == cells_) {
    head_index = 0;
  }
  head_idx_[idx(env)] = static_cast<std::uint16_t>(head_index);
  body[idx(head_index)] = new_cell;
  if (!into_tail) {
    row_set(rows[idx(next_y)], next_x);
  }
  head_[idx(env)] = new_cell;

  grid.set_body(new_cell, 1);
  grid.set_head(old_head, 0);
  grid.set_head(new_cell, 1);

  if (won) {
    reward += config_.win_bonus;
  }

  bool is_truncated = false;
  if (!won) {
    const bool by_steps =
        config_.max_episode_steps > 0 && steps_[idx(env)] >= config_.max_episode_steps;
    const bool by_hunger = config_.max_steps_without_food > 0 &&
                           steps_no_food_[idx(env)] >= config_.max_steps_without_food;
    is_truncated = by_steps || by_hunger;
  }

  finish_step(env, reward, won, is_truncated, won);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void VecSnakeImpl::finish_step(int env, float reward, bool terminated, bool truncated,
                               bool won) noexcept {
  const float episode_return = episode_return_[idx(env)] + reward;
  episode_return_[idx(env)] = episode_return;

  if (terminated || truncated) {
    final_score_[idx(env)] = score_[idx(env)];
    final_len_[idx(env)] = len_[idx(env)];
    final_returns_[idx(env)] = episode_return;
    if (config_.autoreset_mode == AutoresetMode::SameStep) {
      capture_final(env);
      reset_env(env);
    } else {
      needs_reset_[idx(env)] = 1;
    }
  }

  rewards_[idx(env)] = reward;
  returns_[idx(env)] = episode_return;
  terminated_[idx(env)] = terminated ? 1 : 0;
  truncated_[idx(env)] = truncated ? 1 : 0;
  won_[idx(env)] = won ? 1 : 0;
  done_[idx(env)] = (terminated || truncated) ? 1 : 0;
}

void VecSnakeImpl::capture_final(int env) noexcept {
  if (config_.obs_mode == ObsMode::Grid && !final_obs_.empty()) {
    const std::size_t offset = idx(env) * 3 * idx(cells_);
    std::memcpy(final_obs_.data() + offset, obs_.data() + offset, 3 * idx(cells_));
  }
  if (!final_head_.empty()) {
    final_head_[idx(env)] = head_[idx(env)];
    final_food_[idx(env)] = food_[idx(env)];
  }
  if (!final_rows_.empty()) {
    const std::size_t offset = idx(env) * idx(height_);
    std::memcpy(final_rows_.data() + offset, rows_.data() + offset,
                sizeof(std::uint64_t) * idx(height_));
  }
}

void VecSnakeImpl::step(const std::int8_t* actions) noexcept {
  assert(actions != nullptr);
  pool_->parallel_for([this, actions](int begin, int end) noexcept {
    for (int env = begin; env < end; ++env) {
      step_one(env, actions[idx(env)]);
    }
  });
}

void VecSnakeImpl::reset_all() noexcept {
  pool_->parallel_for([this](int begin, int end) noexcept {
    for (int env = begin; env < end; ++env) {
      reset_env(env);
    }
  });
}

void VecSnakeImpl::reset_mask(const std::uint8_t* mask) noexcept {
  assert(mask != nullptr);
  pool_->parallel_for([this, mask](int begin, int end) noexcept {
    for (int env = begin; env < end; ++env) {
      if (mask[idx(env)] != 0) {
        reset_env(env);
      }
    }
  });
}

}  // namespace snake::detail
