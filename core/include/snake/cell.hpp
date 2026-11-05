#pragma once

#include <cstdint>

#include "snake/config.hpp"

namespace snake {

// Packed cell encoding used by head_cells()/food_cells():
//   cell = (y << kCellShift) | x,  x in [0, 63], y in [0, 63].
// kInvalidCell is reserved for "no cell".
inline constexpr int kCellShift = 6;
inline constexpr int kCellMask = (1 << kCellShift) - 1;
constexpr std::uint16_t pack_cell(int x, int y) noexcept {
  return static_cast<std::uint16_t>((y << kCellShift) | x);
}

constexpr int cell_x(std::uint16_t cell) noexcept {
  return cell & kCellMask;
}

constexpr int cell_y(std::uint16_t cell) noexcept {
  return cell >> kCellShift;
}

constexpr bool is_valid_cell(std::uint16_t cell, int width, int height) noexcept {
  return cell != kInvalidCell && cell_x(cell) < width && cell_y(cell) < height;
}

}  // namespace snake
