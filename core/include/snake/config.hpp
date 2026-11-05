#pragma once

#include <cstdint>

namespace snake {

inline constexpr int kMinDimension = 3;
inline constexpr int kMaxDimension = 64;
inline constexpr std::uint16_t kInvalidCell = 0xFFFF;

// How finished episodes are reset.
enum class AutoresetMode : std::uint8_t {
  NextStep = 0,  // reset on the next step; that step's action is ignored
  SameStep = 1,  // reset inside the terminal step
  Disabled = 2,  // caller resets with reset_mask()
};

// What the engine maintains for the caller.
enum class ObsMode : std::uint8_t {
  NoObs = 0,     // no dense observation; bitboards only
  Grid = 1,      // incremental (N,3,H,W) uint8 observation
  Bitboard = 2,  // (N,H) uint64 rows plus head/food cells
};

struct VecConfig {
  int num_envs = 64;

  int width = 16;
  int height = 16;
  int initial_length = 3;

  // Zero disables the corresponding truncation rule.
  int max_episode_steps = 2000;
  int max_steps_without_food = 0;

  float food_reward = 1.0F;
  float death_penalty = -1.0F;
  float step_penalty = 0.0F;
  float win_bonus = 2.0F;

  bool relative_actions = true;
  AutoresetMode autoreset_mode = AutoresetMode::SameStep;
  ObsMode obs_mode = ObsMode::Bitboard;

  std::uint64_t seed = 42;

  // 0 = auto (leaves one core for the caller/OS, capped at 5).
  // 1 = serial. >1 = explicit total threads including the caller.
  int num_threads = 0;
  int parallel_threshold = 128;
};

}  // namespace snake
