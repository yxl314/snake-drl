#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "snake/span.hpp"

namespace snake {

// Non-owning view over N contiguous elements, one per environment.
template <typename T>
class VectorView {
 public:
  using value_type = T;
  using size_type = std::size_t;

  constexpr VectorView() noexcept = default;
  constexpr VectorView(const T* data, int size) noexcept
      : data_(data != nullptr && size > 0 ? data : nullptr), size_(data_ != nullptr ? size : 0) {}

  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr int size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] constexpr bool valid() const noexcept { return data_ != nullptr; }

  [[nodiscard]] constexpr const T& operator[](int index) const noexcept {
    assert(index >= 0 && index < size_);
    return data_[index];
  }

  [[nodiscard]] Span<const T> span() const noexcept {
    return Span<const T>(data_, static_cast<size_type>(size_));
  }

  [[nodiscard]] constexpr const T* begin() const noexcept { return data_; }
  [[nodiscard]] constexpr const T* end() const noexcept { return data_ + size_; }

 private:
  const T* data_ = nullptr;
  int size_ = 0;
};

// Non-owning view over an (N, H) row-major matrix.
template <typename T>
class MatrixView {
 public:
  using value_type = T;
  using size_type = std::size_t;

  constexpr MatrixView() noexcept = default;
  constexpr MatrixView(const T* data, int rows, int cols) noexcept
      : data_(data != nullptr && rows > 0 && cols > 0 ? data : nullptr),
        rows_(data_ != nullptr ? rows : 0),
        cols_(data_ != nullptr ? cols : 0) {}

  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr int rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr int cols() const noexcept { return cols_; }
  [[nodiscard]] constexpr size_type size() const noexcept {
    return static_cast<size_type>(rows_) * static_cast<size_type>(cols_);
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return data_ == nullptr; }
  [[nodiscard]] constexpr bool valid() const noexcept { return data_ != nullptr; }

  [[nodiscard]] constexpr const T& at(int row, int col) const noexcept {
    assert(row >= 0 && row < rows_ && col >= 0 && col < cols_);
    return data_[static_cast<size_type>(row) * static_cast<size_type>(cols_) +
                 static_cast<size_type>(col)];
  }

  [[nodiscard]] Span<const T> operator[](int row) const noexcept {
    assert(row >= 0 && row < rows_);
    return Span<const T>(data_ + static_cast<size_type>(row) * static_cast<size_type>(cols_),
                         static_cast<size_type>(cols_));
  }

  [[nodiscard]] Span<const T> span() const noexcept { return Span<const T>(data_, size()); }

 private:
  const T* data_ = nullptr;
  int rows_ = 0;
  int cols_ = 0;
};

// Non-owning view over (N, 3, H, W) uint8 grid observations.
class GridView {
 public:
  using value_type = std::uint8_t;
  using size_type = std::size_t;

  static constexpr int kChannels = 3;

  constexpr GridView() noexcept = default;
  constexpr GridView(const std::uint8_t* data, int num_envs, int height, int width) noexcept
      : data_(data != nullptr && num_envs > 0 && height > 0 && width > 0 ? data : nullptr),
        num_envs_(data_ != nullptr ? num_envs : 0),
        height_(data_ != nullptr ? height : 0),
        width_(data_ != nullptr ? width : 0) {}

  [[nodiscard]] constexpr const std::uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] constexpr int num_envs() const noexcept { return num_envs_; }
  [[nodiscard]] constexpr int channels() const noexcept { return kChannels; }
  [[nodiscard]] constexpr int height() const noexcept { return height_; }
  [[nodiscard]] constexpr int width() const noexcept { return width_; }
  [[nodiscard]] constexpr size_type size() const noexcept {
    return static_cast<size_type>(num_envs_) * static_cast<size_type>(kChannels) *
           static_cast<size_type>(height_) * static_cast<size_type>(width_);
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return data_ == nullptr; }
  [[nodiscard]] constexpr bool valid() const noexcept { return data_ != nullptr; }

  [[nodiscard]] std::uint8_t at(int env, int channel, int y, int x) const noexcept {
    assert(env >= 0 && env < num_envs_);
    assert(channel >= 0 && channel < kChannels);
    assert(y >= 0 && y < height_);
    assert(x >= 0 && x < width_);
    return data_[index(env, channel, y, x)];
  }

  // One (H, W) channel for one environment.
  [[nodiscard]] Span<const std::uint8_t> plane(int env, int channel) const noexcept {
    assert(env >= 0 && env < num_envs_);
    assert(channel >= 0 && channel < kChannels);
    const size_type plane_size = static_cast<size_type>(height_) * static_cast<size_type>(width_);
    return Span<const std::uint8_t>(
        data_ + static_cast<size_type>(env) * static_cast<size_type>(kChannels) * plane_size +
            static_cast<size_type>(channel) * plane_size,
        plane_size);
  }

  // All three channels for one environment: (3, H, W).
  [[nodiscard]] Span<const std::uint8_t> environment(int env) const noexcept {
    assert(env >= 0 && env < num_envs_);
    const size_type env_size = static_cast<size_type>(kChannels) * static_cast<size_type>(height_) *
                               static_cast<size_type>(width_);
    return Span<const std::uint8_t>(data_ + static_cast<size_type>(env) * env_size, env_size);
  }

  [[nodiscard]] Span<const std::uint8_t> span() const noexcept {
    return Span<const std::uint8_t>(data_, size());
  }

 private:
  [[nodiscard]] size_type index(int env, int channel, int y, int x) const noexcept {
    const size_type plane_size = static_cast<size_type>(height_) * static_cast<size_type>(width_);
    return (static_cast<size_type>(env) * static_cast<size_type>(kChannels) +
            static_cast<size_type>(channel)) *
               plane_size +
           static_cast<size_type>(y) * static_cast<size_type>(width_) + static_cast<size_type>(x);
  }

  const std::uint8_t* data_ = nullptr;
  int num_envs_ = 0;
  int height_ = 0;
  int width_ = 0;
};

using ByteView = VectorView<std::uint8_t>;
using CellView = VectorView<std::uint16_t>;  // packed (y << 6) | x, see snake/cell.hpp
using FloatView = VectorView<float>;
using Int32View = VectorView<std::int32_t>;
using BoardView = MatrixView<std::uint64_t>;  // cell x lives at bit (x ^ 7)

}  // namespace snake
