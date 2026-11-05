#pragma once

#include <cstdint>

#include "snake/cell.hpp"
#include "snake/config.hpp"

namespace snake::detail {

static_assert(kMaxDimension <= 64, "Rows must fit in a single 64-bit word");

constexpr int compact_index(std::uint16_t cell, int width) noexcept {
  return cell_y(cell) * width + cell_x(cell);
}

// Cell x is stored at bit (x ^ 7). This reverses the bits inside each byte,
// making big-endian byte unpacking produce cells in ascending x order on
// little-endian hosts.
constexpr std::uint64_t cell_bit(int x) noexcept {
  return std::uint64_t{1} << (x ^ 7);
}

inline bool row_occupied(std::uint64_t row, int x) noexcept {
  return (row & cell_bit(x)) != 0;
}

inline void row_set(std::uint64_t& row, int x) noexcept {
  row |= cell_bit(x);
}

inline void row_clear(std::uint64_t& row, int x) noexcept {
  row &= ~cell_bit(x);
}

}  // namespace snake::detail
