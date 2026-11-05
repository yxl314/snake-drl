#pragma once

#include <cassert>
#include <cstdint>

#include "detail/bitboard.hpp"
#include "snake/config.hpp"

namespace snake::detail {

// Non-owning view over one environment's (3,H,W) observation slice.
// When disabled (null data), every mutation is a no-op. This keeps a single
// step implementation for Grid, Bitboard and NoObs modes without branches at
// every call site.
class GridObservation {
 public:
  GridObservation(std::uint8_t* data, int width, int height) noexcept
      : data_(data), width_(width), height_(height), cells_(width * height) {}

  bool enabled() const noexcept { return data_ != nullptr; }

  std::uint8_t* data() const noexcept { return data_; }

  void set_body(std::uint16_t cell, std::uint8_t value) noexcept {
    if (data_ == nullptr) {
      return;
    }
    assert(is_valid_cell(cell, width_, height_));
    data_[compact_index(cell, width_)] = value;
  }

  void set_head(std::uint16_t cell, std::uint8_t value) noexcept {
    if (data_ == nullptr) {
      return;
    }
    assert(is_valid_cell(cell, width_, height_));
    data_[cells_ + compact_index(cell, width_)] = value;
  }

  void set_food(std::uint16_t cell, std::uint8_t value) noexcept {
    if (data_ == nullptr) {
      return;
    }
    assert(is_valid_cell(cell, width_, height_));
    data_[2 * cells_ + compact_index(cell, width_)] = value;
  }

  void clear_environment(const std::uint16_t* body, int tail_index, int length, std::uint16_t head,
                         std::uint16_t food) noexcept {
    if (data_ == nullptr) {
      return;
    }
    int cursor = tail_index;
    for (int i = 0; i < length; ++i) {
      set_body(body[cursor], 0);
      if (++cursor == cells_) {
        cursor = 0;
      }
    }
    set_head(head, 0);
    if (food != kInvalidCell) {
      set_food(food, 0);
    }
  }

 private:
  std::uint8_t* data_;
  int width_;
  [[maybe_unused]] int height_;
  int cells_;
};

}  // namespace snake::detail
