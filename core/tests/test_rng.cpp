#include <array>
#include <cstdint>

#include "detail/rng.hpp"
#include "test_framework.hpp"

void run_rng_tests() {
  using snake::detail::Xoshiro256;

  Xoshiro256 a;
  Xoshiro256 b;
  a.seed(123456789);
  b.seed(123456789);
  for (int i = 0; i < 1000; ++i) {
    CHECK(a.next() == b.next());
  }

  Xoshiro256 different;
  different.seed(123456790);
  CHECK(a.next() != different.next() || a.next() != different.next());

  Xoshiro256 rng;
  rng.seed(0);
  for (int i = 0; i < 10000; ++i) {
    const std::uint32_t value = rng.bounded(7);
    CHECK(value < 7);
  }
  CHECK(rng.bounded(1) == 0);

  // Determinism after reseeding.
  Xoshiro256 reseeded;
  reseeded.seed(999);
  const std::uint64_t first = reseeded.next();
  reseeded.seed(999);
  CHECK(reseeded.next() == first);

  // Rejection path must still produce values inside the bound.
  Xoshiro256 tight;
  tight.seed(42);
  for (std::uint32_t bound = 1; bound <= 64; ++bound) {
    for (int i = 0; i < 100; ++i) {
      CHECK(tight.bounded(bound) < bound);
    }
  }

  // Loose uniformity smoke test: catch gross transcription errors without
  // turning this into a statistical test.
  Xoshiro256 uniform;
  uniform.seed(2024);
  std::array<int, 16> buckets{};
  constexpr int kSamples = 16000;
  for (int i = 0; i < kSamples; ++i) {
    ++buckets[uniform.bounded(16)];
  }
  for (int count : buckets) {
    CHECK(count >= 700 && count <= 1300);  // expected 1000
  }
}
