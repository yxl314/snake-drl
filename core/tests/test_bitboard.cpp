#include <cstdint>

#include "detail/bitboard.hpp"
#include "snake/cell.hpp"
#include "test_framework.hpp"

void run_bitboard_tests() {
  using namespace snake;
  using namespace snake::detail;

  CHECK(pack_cell(0, 0) == 0);
  CHECK(pack_cell(3, 5) == ((5 << 6) | 3));
  CHECK(cell_x(pack_cell(17, 42)) == 17);
  CHECK(cell_y(pack_cell(17, 42)) == 42);

  for (int y = 0; y < 64; y += 7) {
    for (int x = 0; x < 64; x += 5) {
      const std::uint16_t cell = pack_cell(x, y);
      CHECK(cell_x(cell) == x);
      CHECK(cell_y(cell) == y);
      CHECK(cell_bit(x) == (std::uint64_t{1} << (x ^ 7)));
    }
  }

  CHECK(compact_index(pack_cell(0, 0), 16) == 0);
  CHECK(compact_index(pack_cell(15, 15), 16) == 255);
  CHECK(compact_index(pack_cell(2, 6), 10) == 62);
  CHECK(compact_index(pack_cell(63, 63), 64) == 4095);

  CHECK(is_valid_cell(pack_cell(0, 0), 1, 1));
  CHECK(!is_valid_cell(pack_cell(1, 0), 1, 1));
  CHECK(!is_valid_cell(pack_cell(0, 1), 1, 1));
  CHECK(!is_valid_cell(snake::kInvalidCell, 64, 64));

  std::uint64_t row = 0;
  row_set(row, 0);
  row_set(row, 7);
  row_set(row, 63);
  CHECK(row_occupied(row, 0));
  CHECK(row_occupied(row, 7));
  CHECK(row_occupied(row, 63));
  CHECK(!row_occupied(row, 1));

  row_clear(row, 7);
  CHECK(!row_occupied(row, 7));
  CHECK(row_occupied(row, 0));
  CHECK(row_occupied(row, 63));

  // Byte-order contract: each byte contains cells x=0..7 in MSB-first order.
  std::uint64_t byte_contract = 0;
  for (int x = 0; x < 8; ++x) {
    row_set(byte_contract, x);
  }
  CHECK(byte_contract == 0xFFULL);
}
