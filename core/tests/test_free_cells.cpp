#include <array>
#include <cstdint>

#include "detail/bitboard.hpp"
#include "detail/free_cells.hpp"
#include "detail/rng.hpp"
#include "test_framework.hpp"

void run_free_cell_tests() {
  using namespace snake;
  using namespace snake::detail;

  constexpr int width = 5;
  constexpr int height = 7;
  constexpr int cells = width * height;

  std::array<std::uint16_t, cells> list{};
  std::array<std::uint16_t, cells> positions{};
  int size = 0;
  FreeCellSet free(list.data(), positions.data(), &size, cells, width);
  free.initialize_full(height);

  CHECK(size == cells);
  CHECK(!free.empty());
  std::array<bool, cells> seen{};
  for (int index = 0; index < size; ++index) {
    const std::uint16_t cell = list[static_cast<std::size_t>(index)];
    CHECK(is_valid_cell(cell, width, height));
    const int compact = compact_index(cell, width);
    CHECK(!seen[static_cast<std::size_t>(compact)]);
    seen[static_cast<std::size_t>(compact)] = true;
    CHECK(positions[static_cast<std::size_t>(compact)] == index);
  }
  for (bool value : seen) {
    CHECK(value);
  }

  const std::uint16_t removed = pack_cell(2, 3);
  free.remove(removed);
  CHECK(size == cells - 1);
  free.add(removed);
  CHECK(size == cells);

  Xoshiro256 rng;
  rng.seed(123);
  std::array<bool, cells> taken{};
  for (int count = 0; count < cells; ++count) {
    const std::uint16_t cell = free.take_random(rng);
    CHECK(cell != snake::kInvalidCell);
    CHECK(is_valid_cell(cell, width, height));
    const int compact = compact_index(cell, width);
    CHECK(!taken[static_cast<std::size_t>(compact)]);
    taken[static_cast<std::size_t>(compact)] = true;
  }
  CHECK(free.empty());
  CHECK(free.size() == 0);
  CHECK(free.take_random(rng) == snake::kInvalidCell);
  for (bool value : taken) {
    CHECK(value);
  }
}
