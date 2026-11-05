#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace snake {

// Minimal read-only span for C++17. A Span never owns data; it is only valid
// while the buffer it points into is alive and not invalidated.
template <typename T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using iterator = T*;

  constexpr Span() noexcept = default;
  constexpr Span(T* data, size_type size) noexcept : data_(data), size_(size) {}

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] constexpr T& operator[](size_type index) const noexcept {
    assert(index < size_);
    return data_[index];
  }

  [[nodiscard]] constexpr iterator begin() const noexcept { return data_; }
  [[nodiscard]] constexpr iterator end() const noexcept { return data_ + size_; }

 private:
  T* data_ = nullptr;
  size_type size_ = 0;
};

}  // namespace snake
