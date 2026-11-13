#pragma once

#include <cassert>
#include <cstdint>

#include "detail/bitboard.hpp"
#include "detail/rng.hpp"
#include "snake/config.hpp"

namespace snake::detail {

// Non-owning view over one environment's free-cell list and its inverse index.
// The list stores packed cells; positions stores the index for each compact
// cell. Removal is swap-remove, so it is O(1) and preserves uniformity.
class FreeCellSet {
 public:
  FreeCellSet(std::uint16_t* cells, std::uint16_t* positions, int* size, int capacity,
              int width) noexcept
      : cells_(cells), positions_(positions), size_(size), capacity_(capacity), width_(width) {}

  void initialize_full(int height) noexcept {
    int count = 0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width_; ++x) {
        const std::uint16_t cell = pack_cell(x, y);
        cells_[count] = cell;
        positions_[compact_index(cell, width_)] = static_cast<std::uint16_t>(count);
        ++count;
      }
    }
    *size_ = count;
  }

  void add(std::uint16_t cell) noexcept {
    assert(*size_ < capacity_);
    const int index = *size_;
    cells_[index] = cell;
    positions_[compact_index(cell, width_)] = static_cast<std::uint16_t>(index);
    *size_ = index + 1;
  }

  void remove(std::uint16_t cell) noexcept {
    assert(*size_ > 0);
    const int index = positions_[compact_index(cell, width_)];
    assert(index >= 0 && index < *size_);
    assert(cells_[index] == cell);
    remove_at(index);
  }

  std::uint16_t take_random(Xoshiro256& rng) noexcept {
    if (*size_ == 0) {
      return kInvalidCell;
    }
    const int index = static_cast<int>(rng.bounded(static_cast<std::uint32_t>(*size_)));
    const std::uint16_t cell = cells_[index];
    remove_at(index);
    return cell;
  }

  [[nodiscard]] int size() const noexcept { return *size_; }
  [[nodiscard]] bool empty() const noexcept { return *size_ == 0; }

 private:
  void remove_at(int index) noexcept {
    assert(index >= 0 && index < *size_);
    const int last = *size_ - 1;
    const std::uint16_t moved = cells_[last];
    cells_[index] = moved;
    positions_[compact_index(moved, width_)] = static_cast<std::uint16_t>(index);
    *size_ = last;
  }

  std::uint16_t* cells_;
  std::uint16_t* positions_;
  int* size_;
  [[maybe_unused]] int capacity_;
  int width_;
};

}  // namespace snake::detail
