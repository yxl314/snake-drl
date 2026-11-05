#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "snake/span.hpp"
#include "snake/view.hpp"
#include "test_framework.hpp"

void run_view_tests() {
  using snake::BoardView;
  using snake::ByteView;
  using snake::CellView;
  using snake::FloatView;
  using snake::GridView;
  using snake::Int32View;
  using snake::MatrixView;
  using snake::Span;
  using snake::VectorView;

  // ---- Span ----
  {
    constexpr std::array<int, 4> values = {1, 2, 3, 4};
    const Span<const int> span(values.data(), values.size());
    CHECK(span.size() == 4);
    CHECK(!span.empty());
    CHECK(span[0] == 1);
    CHECK(span[3] == 4);
    CHECK(span.data() == values.data());

    const int sum = std::accumulate(span.begin(), span.end(), 0);
    CHECK(sum == 10);

    const Span<const int> empty;
    CHECK(empty.size() == 0);
    CHECK(empty.empty());
    CHECK(empty.data() == nullptr);
    CHECK(empty.begin() == empty.end());
  }

  // ---- VectorView ----
  {
    std::vector<std::uint8_t> values = {1, 0, 1};
    ByteView view(values.data(), static_cast<int>(values.size()));
    CHECK(view.valid());
    CHECK(view.size() == 3);
    CHECK(view[2] == 1);
    CHECK(view.span().size() == 3);
    CHECK(view.begin() != view.end());

    const ByteView empty;
    CHECK(!empty.valid());
    CHECK(empty.empty());
    CHECK(empty.size() == 0);
    CHECK(empty.data() == nullptr);
    CHECK(empty.span().empty());

    const ByteView null_data(nullptr, 8);
    CHECK(!null_data.valid());
    CHECK(null_data.size() == 0);

    const ByteView zero_size(values.data(), 0);
    CHECK(!zero_size.valid());
    CHECK(zero_size.size() == 0);
  }

  // ---- MatrixView ----
  {
    std::vector<std::uint64_t> words = {0b01, 0b10, 0b100, 0b1000, 0b10000, 0b100000};
    BoardView board(words.data(), 2, 3);
    CHECK(board.valid());
    CHECK(board.rows() == 2);
    CHECK(board.cols() == 3);
    CHECK(board.size() == 6);
    CHECK(board.at(1, 2) == 0b100000);
    CHECK(board[1].size() == 3);
    CHECK(board[1][0] == 0b1000);
    CHECK(board.span().size() == 6);

    const BoardView empty;
    CHECK(!empty.valid());
    CHECK(empty.empty());
    CHECK(empty.rows() == 0);
    CHECK(empty.cols() == 0);
    CHECK(empty.size() == 0);

    const MatrixView<std::uint64_t> invalid(words.data(), 0, 3);
    CHECK(!invalid.valid());
    CHECK(invalid.size() == 0);
  }

  // ---- GridView ----
  {
    constexpr int num_envs = 2;
    constexpr int height = 2;
    constexpr int width = 3;
    constexpr int env_size = 3 * height * width;
    std::vector<std::uint8_t> data(static_cast<std::size_t>(num_envs) * env_size, 0);

    GridView grid(data.data(), num_envs, height, width);
    CHECK(grid.valid());
    CHECK(grid.num_envs() == 2);
    CHECK(grid.channels() == 3);
    CHECK(grid.height() == 2);
    CHECK(grid.width() == 3);
    CHECK(grid.size() == static_cast<std::size_t>(num_envs) * env_size);

    // Plane 0 of env 1 starts after one full environment.
    const std::size_t base = static_cast<std::size_t>(env_size);
    data[base + 4] = 7;
    CHECK(grid.at(1, 0, 1, 1) == 7);
    CHECK(grid.plane(1, 0).size() == static_cast<std::size_t>(height * width));
    CHECK(grid.plane(1, 0)[4] == 7);
    CHECK(grid.environment(1).size() == static_cast<std::size_t>(env_size));
    CHECK(grid.environment(1)[4] == 7);
    CHECK(grid.span().size() == data.size());

    const GridView empty;
    CHECK(!empty.valid());
    CHECK(empty.empty());
    CHECK(empty.num_envs() == 0);
    CHECK(empty.channels() == 3);
    CHECK(empty.height() == 0);
    CHECK(empty.width() == 0);
    CHECK(empty.size() == 0);

    const GridView null_data(nullptr, 2, height, width);
    CHECK(!null_data.valid());
  }
}
