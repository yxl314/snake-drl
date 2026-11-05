#pragma once

#include <cassert>
#include <cstdint>

namespace snake::detail {

inline std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Small, fast, per-environment RNG. No shared state, no locks.
class Xoshiro256 {
 public:
  // Default-constructed state is seeded so next() is never stuck at zero.
  Xoshiro256() noexcept { seed(0x9E3779B97F4A7C15ULL); }

  void seed(std::uint64_t value) noexcept {
    std::uint64_t state = value;
    state_[0] = splitmix64(state);
    state_[1] = splitmix64(state);
    state_[2] = splitmix64(state);
    state_[3] = splitmix64(state);
  }

  std::uint64_t next() noexcept {
    const std::uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
    const std::uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

  // Uniform in [0, bound). Expected fewer than two iterations (Lemire).
  std::uint32_t bounded(std::uint32_t bound) noexcept {
    assert(bound > 0);
    std::uint64_t product = static_cast<std::uint64_t>(static_cast<std::uint32_t>(next())) * bound;
    std::uint32_t low = static_cast<std::uint32_t>(product);
    if (low < bound) {
      const std::uint32_t threshold = (0U - bound) % bound;
      while (low < threshold) {
        product = static_cast<std::uint64_t>(static_cast<std::uint32_t>(next())) * bound;
        low = static_cast<std::uint32_t>(product);
      }
    }
    return static_cast<std::uint32_t>(product >> 32);
  }

 private:
  static std::uint64_t rotl(std::uint64_t value, int amount) noexcept {
    return (value << amount) | (value >> (64 - amount));
  }

  std::uint64_t state_[4]{};
};

}  // namespace snake::detail
