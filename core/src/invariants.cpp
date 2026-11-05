#include <cstddef>
#include <cstdint>
#include <vector>

#include "detail/bitboard.hpp"
#include "detail/engine.hpp"

namespace snake::detail {

bool VecSnakeImpl::check_invariants(int env) const {
  if (env < 0 || env >= n_) {
    return false;
  }

  const int length_now = len_[idx(env)];
  const int score_now = score_[idx(env)];
  const int step_count = steps_[idx(env)];
  const int hungry_steps = steps_no_food_[idx(env)];
  const int head_index = head_idx_[idx(env)];
  const int tail_index = tail_idx_[idx(env)];
  const std::uint8_t is_done = done_[idx(env)];
  const std::uint8_t is_terminated = terminated_[idx(env)];
  const std::uint8_t is_truncated = truncated_[idx(env)];
  const std::uint8_t is_win = won_[idx(env)];
  const std::uint8_t reset_required = needs_reset_[idx(env)];

  // ---- flags and scalar bookkeeping ----
  if (length_now < 1 || length_now > cells_) {
    return false;
  }
  if (score_now < 0 || length_now != config_.initial_length + score_now) {
    return false;
  }
  if (head_index < 0 || head_index >= cells_ || tail_index < 0 || tail_index >= cells_) {
    return false;
  }
  if (dir_[idx(env)] > 3) {
    return false;
  }
  if (step_count < 0 || hungry_steps < 0 || hungry_steps > step_count) {
    return false;
  }
  if (config_.max_episode_steps > 0 && step_count > config_.max_episode_steps) {
    return false;
  }
  if (config_.max_steps_without_food > 0 && hungry_steps > config_.max_steps_without_food) {
    return false;
  }
  if (is_terminated && is_truncated) {
    return false;
  }
  if (is_done != ((is_terminated || is_truncated) ? 1 : 0)) {
    return false;
  }
  if (is_win && !is_terminated) {
    return false;
  }
  if (reset_required != 0 && is_done == 0) {
    return false;
  }

  // ---- body, free list, occupancy and observations ----
  const std::size_t body_offset = idx(env) * idx(cells_);
  const std::size_t rows_offset = idx(env) * idx(height_);
  const std::uint16_t* body = body_.data() + body_offset;
  const std::uint16_t* free_list = free_.data() + body_offset;
  const std::uint16_t* free_index = free_pos_.data() + body_offset;
  const std::uint64_t* rows = rows_.data() + rows_offset;

  std::vector<std::uint8_t> body_flag(idx(cells_), 0);
  std::vector<std::uint8_t> used(idx(cells_), 0);

  int cursor = tail_index;
  for (int part = 0; part < length_now; ++part) {
    const std::uint16_t cell = body[idx(cursor)];
    if (!is_valid_cell(cell, width_, height_)) {
      return false;
    }
    const int index = compact_index(cell, width_);
    if (body_flag[idx(index)] != 0) {
      return false;
    }
    body_flag[idx(index)] = 1;
    used[idx(index)] = 1;
    if (++cursor == cells_) {
      cursor = 0;
    }
  }

  if (head_[idx(env)] != body[idx(head_index)]) {
    return false;
  }
  if (head_index != (tail_index + length_now - 1) % cells_) {
    return false;
  }
  if (!is_valid_cell(head_[idx(env)], width_, height_) ||
      body_flag[idx(compact_index(head_[idx(env)], width_))] == 0) {
    return false;
  }

  const std::uint16_t food = food_[idx(env)];
  if (food == kInvalidCell) {
    if (length_now != cells_) {
      return false;
    }
  } else {
    if (length_now == cells_) {
      return false;
    }
    if (!is_valid_cell(food, width_, height_)) {
      return false;
    }
    const int index = compact_index(food, width_);
    if (used[idx(index)] != 0) {
      return false;
    }
    used[idx(index)] = 1;
  }

  const int expected_free = cells_ - length_now - (food != kInvalidCell ? 1 : 0);
  if (free_count_[idx(env)] != expected_free) {
    return false;
  }

  for (int index = 0; index < free_count_[idx(env)]; ++index) {
    const std::uint16_t cell = free_list[idx(index)];
    if (!is_valid_cell(cell, width_, height_)) {
      return false;
    }
    const int compact = compact_index(cell, width_);
    if (used[idx(compact)] != 0) {
      return false;
    }
    if (free_index[idx(compact)] != static_cast<std::uint16_t>(index)) {
      return false;
    }
    used[idx(compact)] = 1;
  }

  for (int index = 0; index < cells_; ++index) {
    if (used[idx(index)] == 0) {
      return false;
    }
  }

  std::uint64_t valid_mask = 0;
  for (int x = 0; x < width_; ++x) {
    valid_mask |= cell_bit(x);
  }
  for (int y = 0; y < height_; ++y) {
    if ((rows[idx(y)] & ~valid_mask) != 0) {
      return false;
    }
    for (int x = 0; x < width_; ++x) {
      const bool occupied = row_occupied(rows[idx(y)], x);
      const bool expected = body_flag[idx(y * width_ + x)] != 0;
      if (occupied != expected) {
        return false;
      }
    }
  }

  if (!obs_.empty()) {
    const std::uint8_t* obs = obs_.data() + idx(env) * 3 * idx(cells_);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const int index = y * width_ + x;
        const std::uint16_t cell = pack_cell(x, y);
        if (obs[idx(index)] != body_flag[idx(index)]) {
          return false;
        }
        if (obs[idx(cells_ + index)] != (head_[idx(env)] == cell ? 1 : 0)) {
          return false;
        }
        if (obs[idx(2 * cells_ + index)] != (food == cell ? 1 : 0)) {
          return false;
        }
      }
    }
  }

  return true;
}

bool VecSnakeImpl::check_all_invariants() const {
  for (int env = 0; env < n_; ++env) {
    if (!check_invariants(env)) {
      return false;
    }
  }
  return true;
}

}  // namespace snake::detail
