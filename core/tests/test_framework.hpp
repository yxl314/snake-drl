#pragma once

#include <cmath>
#include <cstdio>
#include <type_traits>

namespace test {

inline int& failures() noexcept {
  static int value = 0;
  return value;
}

inline void expect(bool condition, const char* expression, const char* file, int line) {
  if (condition) {
    return;
  }
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", file, line, expression);
  ++failures();
}

template <typename T>
inline double widen(T value) noexcept {
  if constexpr (std::is_same_v<std::decay_t<T>, double>) {
    return value;
  } else {
    return static_cast<double>(value);
  }
}

template <typename A, typename B, typename C>
inline void expect_near(A actual, B expected, C tolerance, const char* expression, const char* file,
                        int line) {
  const double actual_value = widen(actual);
  const double expected_value = widen(expected);
  const double tolerance_value = widen(tolerance);
  if (std::fabs(actual_value - expected_value) <= tolerance_value) {
    return;
  }
  std::fprintf(stderr, "CHECK_NEAR failed at %s:%d: %s (actual=%f expected=%f)\n", file, line,
               expression, actual_value, expected_value);
  ++failures();
}

inline int summarize() noexcept {
  // cppcheck-suppress knownConditionTrueFalse  // failures() is mutated by CHECK
  if (failures() == 0) {
    std::printf("all tests passed\n");
    return 0;
  }
  std::fprintf(stderr, "%d check(s) failed\n", failures());
  return 1;
}

}  // namespace test

#define CHECK(expression) test::expect((expression), #expression, __FILE__, __LINE__)

#define CHECK_NEAR(actual, expected, tolerance) \
  test::expect_near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

#define CHECK_THROW(expression, exception_type) \
  do {                                          \
    bool caught = false;                        \
    try {                                       \
      expression;                               \
    } catch (const exception_type&) {           \
      caught = true;                            \
    } catch (...) {                             \
    }                                           \
    CHECK(caught);                              \
  } while (false)
