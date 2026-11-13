#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "detail/bitboard.hpp"
#include "detail/engine.hpp"
#include "snake/vec_snake.hpp"
#include "test_access.hpp"
#include "test_framework.hpp"

namespace {

snake::VecConfig make_config(snake::AutoresetMode autoreset, snake::ObsMode obs, int num_envs,
                             int width, int height) {
  snake::VecConfig config;
  config.num_envs = num_envs;
  config.width = width;
  config.height = height;
  config.initial_length = 3;
  config.max_episode_steps = 0;
  config.max_steps_without_food = 0;
  config.autoreset_mode = autoreset;
  config.obs_mode = obs;
  config.seed = 1234567;
  config.num_threads = 1;
  config.parallel_threshold = 1;
  return config;
}

// cppcheck-suppress constParameterReference  // mt19937::operator() is non-const
std::vector<std::int8_t> random_actions(int count, std::mt19937& rng) {
  std::vector<std::int8_t> actions(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    actions[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(rng() % 3);
  }
  return actions;
}

void stress(snake::ObsMode obs, snake::AutoresetMode autoreset, int num_envs, int width, int height,
            int steps, unsigned seed) {
  snake::VecSnake env(make_config(autoreset, obs, num_envs, width, height));
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> reset_mask(static_cast<std::size_t>(num_envs), 0);

  for (int step = 0; step < steps; ++step) {
    if (autoreset == snake::AutoresetMode::Disabled) {
      bool any = false;
      for (int env_index = 0; env_index < num_envs; ++env_index) {
        reset_mask[static_cast<std::size_t>(env_index)] = env.needs_reset()[env_index];
        any = any || env.needs_reset()[env_index] != 0;
      }
      if (any) {
        env.reset_mask(reset_mask.data());
      }
    }

    const auto actions = random_actions(num_envs, rng);
    env.step(actions.data());

    for (int env_index = 0; env_index < num_envs; ++env_index) {
      CHECK(std::isfinite(env.rewards()[env_index]));
      CHECK(env.dones()[env_index] == 0 || env.dones()[env_index] == 1);
      if (autoreset == snake::AutoresetMode::SameStep) {
        CHECK(env.needs_reset()[env_index] == 0);
      }
    }

    if (step % 97 == 0) {
      CHECK(env.check_all_invariants());
    }
  }

  CHECK(env.check_all_invariants());
  for (int env_index = 0; env_index < num_envs; ++env_index) {
    CHECK(env.lengths()[env_index] >= 1);
    CHECK(env.lengths()[env_index] <= env.cells());
    CHECK(env.scores()[env_index] >= 0);
  }
}

void test_wall_collision() {
  for (auto mode : {snake::AutoresetMode::SameStep, snake::AutoresetMode::NextStep}) {
    snake::detail::VecSnakeImpl env(make_config(mode, snake::ObsMode::Grid, 1, 16, 16));
    // Keep food off the straight path so the test is about wall collision only.
    snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
    const std::int8_t action = 0;

    for (int step = 0; step < 7; ++step) {
      env.step(&action);
      CHECK(env.dones()[0] == 0);
    }
    env.step(&action);
    CHECK(env.dones()[0] == 1);
    CHECK(env.terminated()[0] == 1);
    CHECK(env.truncated()[0] == 0);
    CHECK(env.wins()[0] == 0);
    CHECK(env.final_scores()[0] == 0);
    CHECK(env.final_lengths()[0] == 3);

    if (mode == snake::AutoresetMode::SameStep) {
      CHECK(env.needs_reset()[0] == 0);
      CHECK(env.lengths()[0] == 3);
      CHECK(env.steps()[0] == 0);
      env.step(&action);
      CHECK(env.dones()[0] == 0);
      CHECK(env.steps()[0] == 1);
    } else {
      CHECK(env.needs_reset()[0] == 1);
      env.step(&action);
      CHECK(env.dones()[0] == 0);
      CHECK(env.needs_reset()[0] == 0);
      CHECK(env.steps()[0] == 0);
      CHECK(env.lengths()[0] == 3);
    }
    CHECK(env.check_all_invariants());
  }
}

void test_disabled_reset_mask() {
  snake::VecSnake env(make_config(snake::AutoresetMode::Disabled, snake::ObsMode::Grid, 4, 16, 16));
  const std::vector<std::int8_t> actions(4, 0);
  for (int step = 0; step < 8; ++step) {
    env.step(actions.data());
  }
  CHECK(env.needs_reset()[0] == 1);
  CHECK(env.needs_reset()[1] == 1);
  CHECK(env.needs_reset()[2] == 1);
  CHECK(env.needs_reset()[3] == 1);

  const std::vector<std::uint8_t> mask = {1, 0, 1, 0};
  env.reset_mask(mask.data());
  CHECK(env.needs_reset()[0] == 0);
  CHECK(env.needs_reset()[1] == 1);
  CHECK(env.needs_reset()[2] == 0);
  CHECK(env.needs_reset()[3] == 1);
  CHECK(env.check_all_invariants());
}

void test_determinism_and_threads() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Bitboard, 32, 16, 16);
  config.num_threads = 1;
  snake::VecConfig parallel_config = config;
  parallel_config.num_threads = 4;

  snake::VecSnake serial_a(config);
  snake::VecSnake serial_b(config);
  snake::VecSnake parallel(parallel_config);
  std::mt19937 rng(777);
  const std::size_t rows =
      static_cast<std::size_t>(config.num_envs) * static_cast<std::size_t>(config.height);

  for (int step = 0; step < 3000; ++step) {
    const auto actions = random_actions(config.num_envs, rng);
    serial_a.step(actions.data());
    serial_b.step(actions.data());
    parallel.step(actions.data());
    CHECK(std::memcmp(serial_a.bitboards().data(), serial_b.bitboards().data(),
                      rows * sizeof(std::uint64_t)) == 0);
    CHECK(std::memcmp(serial_a.bitboards().data(), parallel.bitboards().data(),
                      rows * sizeof(std::uint64_t)) == 0);
    CHECK(std::memcmp(serial_a.rewards().data(), parallel.rewards().data(),
                      sizeof(float) * static_cast<std::size_t>(config.num_envs)) == 0);
  }
}

void test_grid_bitboard_consistency() {
  snake::VecConfig grid_config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 8, 16, 16);
  snake::VecConfig bit_config = grid_config;
  bit_config.obs_mode = snake::ObsMode::Bitboard;

  snake::VecSnake grid(grid_config);
  snake::VecSnake bit(bit_config);
  std::mt19937 rng(999);
  const std::size_t rows =
      static_cast<std::size_t>(grid_config.num_envs) * static_cast<std::size_t>(grid_config.height);

  for (int step = 0; step < 5000; ++step) {
    const auto actions = random_actions(grid_config.num_envs, rng);
    grid.step(actions.data());
    bit.step(actions.data());
    CHECK(std::memcmp(grid.bitboards().data(), bit.bitboards().data(),
                      rows * sizeof(std::uint64_t)) == 0);
    CHECK(std::memcmp(grid.head_cells().data(), bit.head_cells().data(),
                      sizeof(std::uint16_t) * static_cast<std::size_t>(grid_config.num_envs)) == 0);
    CHECK(std::memcmp(grid.food_cells().data(), bit.food_cells().data(),
                      sizeof(std::uint16_t) * static_cast<std::size_t>(grid_config.num_envs)) == 0);
    CHECK(std::memcmp(grid.scores().data(), bit.scores().data(),
                      sizeof(std::int32_t) * static_cast<std::size_t>(grid_config.num_envs)) == 0);
    CHECK(std::memcmp(grid.dones().data(), bit.dones().data(),
                      sizeof(std::uint8_t) * static_cast<std::size_t>(grid_config.num_envs)) == 0);
    if (step % 211 == 0) {
      CHECK(grid.check_all_invariants());
      CHECK(bit.check_all_invariants());
    }
  }
}

void test_odd_boards() {
  stress(snake::ObsMode::Grid, snake::AutoresetMode::SameStep, 16, 10, 7, 5000, 101);
  stress(snake::ObsMode::Bitboard, snake::AutoresetMode::SameStep, 16, 10, 7, 5000, 102);
  stress(snake::ObsMode::Grid, snake::AutoresetMode::SameStep, 8, 3, 64, 5000, 103);
  stress(snake::ObsMode::Bitboard, snake::AutoresetMode::SameStep, 8, 64, 3, 5000, 104);
  stress(snake::ObsMode::Grid, snake::AutoresetMode::NextStep, 16, 5, 5, 5000, 105);
}

void test_absolute_actions() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
  config.relative_actions = false;
  snake::VecSnake env(config);

  const int start_x = snake::cell_x(env.head_cells()[0]);
  const int start_y = snake::cell_y(env.head_cells()[0]);

  const std::int8_t down = 2;
  env.step(&down);
  CHECK(snake::cell_x(env.head_cells()[0]) == start_x);
  CHECK(snake::cell_y(env.head_cells()[0]) == start_y + 1);

  const std::int8_t opposite = 0;
  env.step(&opposite);
  CHECK(snake::cell_x(env.head_cells()[0]) == start_x);
  CHECK(snake::cell_y(env.head_cells()[0]) == start_y + 2);
  CHECK(env.check_all_invariants());
}

void test_step_limits() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 4, 16, 16);
  config.max_episode_steps = 5;
  snake::VecSnake env(config);
  const std::vector<std::int8_t> actions(4, 0);
  for (int step = 0; step < 5; ++step) {
    env.step(actions.data());
  }
  for (int env_index = 0; env_index < 4; ++env_index) {
    CHECK(env.dones()[env_index] == 1);
    CHECK(env.truncated()[env_index] == 1);
    CHECK(env.terminated()[env_index] == 0);
  }
}

void test_hunger_limit() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
  config.max_steps_without_food = 3;
  snake::detail::VecSnakeImpl env(config);
  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
  CHECK(env.check_invariants(0));

  const std::int8_t action = 0;
  for (int step = 0; step < 2; ++step) {
    env.step(&action);
    CHECK(env.truncated()[0] == 0);
  }
  env.step(&action);
  CHECK(env.truncated()[0] == 1);
  CHECK(env.terminated()[0] == 0);
  CHECK(env.dones()[0] == 1);
}

void test_auto_threads() {
  const unsigned hardware = std::thread::hardware_concurrency();
  const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
  const int expected_auto = std::max(1, std::min(available - 1, 5));

  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 512, 16, 16);
  config.num_threads = 0;
  config.parallel_threshold = 128;
  snake::VecSnake env(config);
  CHECK(env.active_threads() == std::min(expected_auto, config.num_envs));

  std::mt19937 rng(321);
  for (int step = 0; step < 200; ++step) {
    const auto actions = random_actions(config.num_envs, rng);
    env.step(actions.data());
  }
  CHECK(env.check_all_invariants());

  snake::VecConfig explicit_threads =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 4, 16, 16);
  explicit_threads.num_threads = 4;
  explicit_threads.parallel_threshold = 1;
  CHECK(snake::VecSnake(explicit_threads).active_threads() == 4);

  snake::VecConfig below_threshold =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 127, 16, 16);
  below_threshold.num_threads = 4;
  below_threshold.parallel_threshold = 128;
  CHECK(snake::VecSnake(below_threshold).active_threads() == 1);

  snake::VecConfig at_threshold =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 128, 16, 16);
  at_threshold.num_threads = 4;
  at_threshold.parallel_threshold = 128;
  CHECK(snake::VecSnake(at_threshold).active_threads() == 4);
}

void test_initial_length_boundary() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
  snake::VecSnake env(config);
  CHECK(env.check_invariants(0));
  const std::int8_t action = 0;
  env.step(&action);
  CHECK(env.dones()[0] == 1);
  CHECK(env.terminated()[0] == 1);
  CHECK(env.final_lengths()[0] == 3);
}

void test_invalid_config() {
  const auto expect_invalid = [](const snake::VecConfig& config) {
    // cppcheck-suppress unusedScopedObject  // the constructor is expected to throw
    CHECK_THROW(snake::VecSnake{config}, std::invalid_argument);
  };

  snake::VecConfig config;

  config = snake::VecConfig{};
  config.num_envs = 0;
  expect_invalid(config);
  config.num_envs = 65537;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.width = 2;
  expect_invalid(config);
  config = snake::VecConfig{};
  config.width = 65;
  expect_invalid(config);
  config = snake::VecConfig{};
  config.height = 2;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.initial_length = 0;
  expect_invalid(config);
  config = snake::VecConfig{};
  config.initial_length = config.width + 1;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.max_episode_steps = -1;
  expect_invalid(config);
  config = snake::VecConfig{};
  config.max_steps_without_food = -1;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.food_reward = std::numeric_limits<float>::infinity();
  expect_invalid(config);
  config = snake::VecConfig{};
  config.death_penalty = std::numeric_limits<float>::quiet_NaN();
  expect_invalid(config);
  config = snake::VecConfig{};
  config.step_penalty = -std::numeric_limits<float>::infinity();
  expect_invalid(config);
  config = snake::VecConfig{};
  config.win_bonus = std::numeric_limits<float>::infinity();
  expect_invalid(config);

  config = snake::VecConfig{};
  config.num_threads = -1;
  expect_invalid(config);
  config = snake::VecConfig{};
  config.num_threads = 257;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.parallel_threshold = 0;
  expect_invalid(config);

  config = snake::VecConfig{};
  config.autoreset_mode = static_cast<snake::AutoresetMode>(3);
  expect_invalid(config);
  config = snake::VecConfig{};
  config.obs_mode = static_cast<snake::ObsMode>(9);
  expect_invalid(config);

  // Valid boundaries must not throw.
  config = snake::VecConfig{};
  config.initial_length = 1;
  CHECK(snake::VecSnake(config).check_all_invariants());
  config = snake::VecConfig{};
  config.width = 64;
  config.height = 64;
  CHECK(snake::VecSnake(config).check_all_invariants());
  config = snake::VecConfig{};
  config.num_threads = 1;
  CHECK(snake::VecSnake(config).check_all_invariants());
}

void test_win_and_final_state() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
  snake::detail::VecSnakeImpl env(config);
  snake::detail::VecSnakeTestAccess::force_near_win(env);
  CHECK(env.check_invariants(0));

  const std::int8_t action = 0;  // head (1,2) moves right into food (2,2)
  env.step(&action);
  CHECK(env.dones()[0] == 1);
  CHECK(env.terminated()[0] == 1);
  CHECK(env.wins()[0] == 1);
  CHECK(env.final_scores()[0] == 6);
  CHECK(env.final_lengths()[0] == 9);
  CHECK_NEAR(env.rewards()[0], 3.0, 1e-6);

  // SameStep has already reset the live state.
  CHECK(env.lengths()[0] == 3);
  CHECK(env.scores()[0] == 0);
  CHECK(env.check_invariants(0));

  const auto final_obs = env.final_observations();
  CHECK(final_obs.valid());
  const auto cells = static_cast<std::size_t>(env.cells());
  int body_count = 0;
  int head_count = 0;
  int food_count = 0;
  for (std::size_t index = 0; index < cells; ++index) {
    body_count += final_obs.span()[index];
    head_count += final_obs.span()[cells + index];
    food_count += final_obs.span()[2 * cells + index];
  }
  CHECK(body_count == 9);
  CHECK(head_count == 1);
  CHECK(food_count == 0);
  CHECK(env.final_head_cells()[0] == snake::pack_cell(2, 2));
  CHECK(env.final_food_cells()[0] == snake::kInvalidCell);
}

void expect_same_state(const snake::VecSnake& actual, const snake::VecSnake& expected) {
  const int n = expected.num_envs();
  CHECK(actual.num_envs() == n);
  CHECK(actual.cells() == expected.cells());
  CHECK(actual.final_head_cells().valid() == expected.final_head_cells().valid());
  CHECK(actual.final_bitboards().valid() == expected.final_bitboards().valid());
  for (int env = 0; env < n; ++env) {
    CHECK(actual.head_cells()[env] == expected.head_cells()[env]);
    CHECK(actual.food_cells()[env] == expected.food_cells()[env]);
    CHECK(actual.scores()[env] == expected.scores()[env]);
    CHECK(actual.lengths()[env] == expected.lengths()[env]);
    CHECK(actual.steps()[env] == expected.steps()[env]);
    CHECK(actual.dones()[env] == expected.dones()[env]);
    CHECK(actual.terminated()[env] == expected.terminated()[env]);
    CHECK(actual.truncated()[env] == expected.truncated()[env]);
    CHECK(actual.wins()[env] == expected.wins()[env]);
    CHECK(actual.directions()[env] == expected.directions()[env]);
    CHECK_NEAR(actual.rewards()[env], expected.rewards()[env], 0.0);
    CHECK_NEAR(actual.returns()[env], expected.returns()[env], 0.0);
    CHECK_NEAR(actual.final_returns()[env], expected.final_returns()[env], 0.0);
    CHECK(actual.final_scores()[env] == expected.final_scores()[env]);
    CHECK(actual.final_lengths()[env] == expected.final_lengths()[env]);
    if (expected.final_head_cells().valid()) {
      CHECK(actual.final_head_cells()[env] == expected.final_head_cells()[env]);
      CHECK(actual.final_food_cells()[env] == expected.final_food_cells()[env]);
    }
  }
  const std::size_t words =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(expected.height());
  CHECK(std::memcmp(actual.bitboards().data(), expected.bitboards().data(),
                    sizeof(std::uint64_t) * words) == 0);
}

void test_reseed() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Bitboard, 8, 16, 16);
  snake::VecSnake fresh(config);
  snake::VecSnake env(config);

  // Move the engine away from its construction state, exercising autoresets.
  std::mt19937 rng(4242);
  for (int step = 0; step < 137; ++step) {
    const auto actions = random_actions(config.num_envs, rng);
    env.step(actions.data());
  }

  env.reseed_all(config.seed);
  expect_same_state(env, fresh);
  env.reseed_all(config.seed);  // idempotent
  expect_same_state(env, fresh);

  // The trajectory after reseeding must match the fresh engine step by step.
  for (int step = 0; step < 50; ++step) {
    const auto actions = random_actions(config.num_envs, rng);
    env.step(actions.data());
    fresh.step(actions.data());
    expect_same_state(env, fresh);
  }

  // reseed_env(i, seed + i) must match environment i of reseed_all(seed).
  snake::VecConfig many =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Bitboard, 16, 16, 16);
  snake::VecSnake all(many);
  all.reseed_all(1234);
  snake::VecSnake one(many);
  one.reseed_env(5, 1234 + 5);
  CHECK(one.food_cells()[5] == all.food_cells()[5]);
  CHECK(one.head_cells()[5] == all.head_cells()[5]);
  CHECK(one.bitboards().at(5, 0) == all.bitboards().at(5, 0));

  // A reseeded batch must not collapse every environment onto one food cell.
  std::set<std::uint16_t> foods;
  for (int env_index = 0; env_index < many.num_envs; ++env_index) {
    foods.insert(all.food_cells()[env_index]);
  }
  CHECK(foods.size() >= 8);
  CHECK(all.check_all_invariants());
}

}  // namespace

void test_win_bitboard() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Bitboard, 1, 3, 3);
  snake::detail::VecSnakeImpl env(config);
  snake::detail::VecSnakeTestAccess::force_near_win(env);
  CHECK(env.check_invariants(0));

  const std::int8_t action = 0;
  env.step(&action);
  CHECK(env.dones()[0] == 1);
  CHECK(env.wins()[0] == 1);
  CHECK(env.final_scores()[0] == 6);
  CHECK(env.final_lengths()[0] == 9);

  const auto rows = env.final_bitboards();
  CHECK(rows.valid());
  int occupied = 0;
  for (int y = 0; y < rows.rows(); ++y) {
    for (std::uint64_t value : rows[y]) {
      while (value != 0) {
        value &= value - 1;
        ++occupied;
      }
    }
  }
  CHECK(occupied == 9);
  CHECK(env.final_head_cells()[0] == snake::pack_cell(2, 2));
  CHECK(env.final_food_cells()[0] == snake::kInvalidCell);
  CHECK(env.lengths()[0] == 3);
  CHECK(env.check_invariants(0));
}

void test_reseed_env_validation() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Bitboard, 2, 16, 16);
  snake::VecSnake env(config);
  CHECK_THROW(env.reseed_env(-1, 1), std::out_of_range);
  CHECK_THROW(env.reseed_env(env.num_envs(), 1), std::out_of_range);
  env.reseed_env(0, 42);  // in range: must not throw
  CHECK(env.check_all_invariants());
}

void test_directions() {
  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
    snake::VecSnake env(config);
    CHECK(env.directions()[0] == 1);  // starts facing right
    const std::int8_t turn_right = 1;
    env.step(&turn_right);
    CHECK(env.directions()[0] == 2);
    const std::int8_t straight = 0;
    env.step(&straight);
    CHECK(env.directions()[0] == 2);
    const std::int8_t turn_left = 2;
    env.step(&turn_left);
    CHECK(env.directions()[0] == 1);
  }

  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
    config.relative_actions = false;
    snake::VecSnake env(config);
    const int start_x = snake::cell_x(env.head_cells()[0]);
    const int start_y = snake::cell_y(env.head_cells()[0]);

    const std::int8_t up = 0;
    env.step(&up);
    CHECK(env.directions()[0] == 0);
    CHECK(snake::cell_y(env.head_cells()[0]) == start_y - 1);

    const std::int8_t down = 2;  // 180-degree reversal is ignored
    env.step(&down);
    CHECK(env.directions()[0] == 0);
    CHECK(snake::cell_x(env.head_cells()[0]) == start_x);
    CHECK(snake::cell_y(env.head_cells()[0]) == start_y - 2);

    const std::int8_t left = 3;
    env.step(&left);
    CHECK(env.directions()[0] == 3);
    CHECK(snake::cell_x(env.head_cells()[0]) == start_x - 1);
  }
}

void test_self_collision() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
  snake::detail::VecSnakeImpl env(config);
  // Ring order: tail first, head last. Head (0,1) faces up into body cell (0,0).
  const std::vector<std::pair<int, int>> body = {{1, 1}, {1, 0}, {0, 0}, {0, 1}};
  snake::detail::VecSnakeTestAccess::place_snake(env, body, 0, snake::pack_cell(2, 2));
  CHECK(env.check_invariants(0));

  const std::int8_t straight = 0;
  env.step(&straight);
  CHECK(env.dones()[0] == 1);
  CHECK(env.terminated()[0] == 1);
  CHECK(env.truncated()[0] == 0);
  CHECK(env.wins()[0] == 0);
  CHECK(env.final_lengths()[0] == 4);
  CHECK(env.final_scores()[0] == 1);
  CHECK_NEAR(env.final_returns()[0], -1.0, 1e-6);
  CHECK(env.check_invariants(0));
}

void test_into_tail() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
  snake::detail::VecSnakeImpl env(config);
  // Head (1,1) moves left into the current tail cell (0,1): legal, no growth.
  const std::vector<std::pair<int, int>> body = {{0, 1}, {0, 0}, {1, 0}, {1, 1}};
  snake::detail::VecSnakeTestAccess::place_snake(env, body, 3, snake::pack_cell(2, 2));
  CHECK(env.check_invariants(0));

  const std::int8_t straight = 0;
  env.step(&straight);
  CHECK(env.dones()[0] == 0);
  CHECK(env.lengths()[0] == 4);
  CHECK(env.head_cells()[0] == snake::pack_cell(0, 1));
  CHECK(env.check_invariants(0));
}

void test_disabled_terminal_is_sticky() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::Disabled, snake::ObsMode::Grid, 2, 16, 16);
  snake::detail::VecSnakeImpl env(config);
  snake::detail::VecSnakeTestAccess::force_food(env, 0, snake::pack_cell(0, 0));
  snake::detail::VecSnakeTestAccess::force_food(env, 1, snake::pack_cell(0, 1));
  const std::vector<std::int8_t> actions(2, 0);
  for (int step = 0; step < 8; ++step) {
    env.step(actions.data());
  }
  CHECK(env.dones()[0] == 1);
  CHECK(env.dones()[1] == 1);
  CHECK(env.needs_reset()[0] == 1);
  CHECK(env.needs_reset()[1] == 1);

  const std::int32_t steps_before = env.steps()[0];
  const float returns_before = env.returns()[0];

  // A step without a reset is a no-op for a terminal environment: it must not
  // clear done/terminated/won or the accumulated return.
  env.step(actions.data());
  CHECK(env.dones()[0] == 1);
  CHECK(env.terminated()[0] == 1);
  CHECK(env.needs_reset()[0] == 1);
  CHECK(env.steps()[0] == steps_before);
  CHECK(env.returns()[0] == returns_before);
  CHECK(env.check_all_invariants());

  // Partial reset: env 1 stays terminal and must remain observable as such.
  const std::vector<std::uint8_t> mask = {1, 0};
  env.reset_mask(mask.data());
  CHECK(env.dones()[0] == 0);
  CHECK(env.needs_reset()[0] == 0);
  CHECK(env.dones()[1] == 1);
  CHECK(env.needs_reset()[1] == 1);
  CHECK(env.returns()[1] == returns_before);

  env.step(actions.data());
  CHECK(env.dones()[0] == 0);  // env 0 is playing again
  CHECK(env.dones()[1] == 1);  // env 1 is still terminal
  CHECK(env.returns()[1] == returns_before);
  CHECK(env.check_all_invariants());
}

void test_final_returns() {
  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
    snake::detail::VecSnakeImpl env(config);
    snake::detail::VecSnakeTestAccess::force_near_win(env);
    const std::int8_t straight = 0;
    env.step(&straight);
    CHECK(env.wins()[0] == 1);
    CHECK_NEAR(env.returns()[0], 3.0, 1e-6);
    CHECK_NEAR(env.final_returns()[0], 3.0, 1e-6);

    // After a reset the live return belongs to the new episode, while
    // final_returns still holds the finished one.
    env.reset_all();
    CHECK_NEAR(env.final_returns()[0], 3.0, 1e-6);
    CHECK_NEAR(env.returns()[0], 0.0, 1e-6);
  }

  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
    snake::detail::VecSnakeImpl env(config);
    snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
    const std::int8_t straight = 0;
    for (int step = 0; step < 8; ++step) {
      env.step(&straight);
    }
    CHECK(env.dones()[0] == 1);
    CHECK_NEAR(env.final_returns()[0], -1.0, 1e-6);
  }
}

void test_reward_config() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
  config.food_reward = 2.5F;
  config.death_penalty = -4.0F;
  config.step_penalty = -0.25F;
  snake::detail::VecSnakeImpl env(config);

  // Eat: step_penalty + food_reward.
  const int head_x = snake::cell_x(env.head_cells()[0]);
  const int head_y = snake::cell_y(env.head_cells()[0]);
  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(head_x + 1, head_y));
  CHECK(env.check_invariants(0));
  const std::int8_t straight = 0;
  env.step(&straight);
  CHECK_NEAR(env.rewards()[0], 2.25, 1e-6);
  CHECK(env.scores()[0] == 1);
  CHECK(env.lengths()[0] == config.initial_length + 1);

  // Death: place food away, turn up and run into the wall.
  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
  const std::int8_t turn_left = 2;  // right -> up
  env.step(&turn_left);
  float last_reward = 0.0F;
  for (int step = 0; step < 20 && env.dones()[0] == 0; ++step) {
    env.step(&straight);
    last_reward = env.rewards()[0];
  }
  CHECK(env.dones()[0] == 1);
  CHECK_NEAR(last_reward, -4.25, 1e-6);
  CHECK_NEAR(env.final_returns()[0], env.returns()[0], 1e-6);
  CHECK(env.check_invariants(0));
}

void test_invariant_checker_rejects_corruption() {
  using Access = snake::detail::VecSnakeTestAccess;

  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    CHECK(env.check_invariants(0));
    CHECK(!env.check_invariants(-1));
    CHECK(!env.check_invariants(env.num_envs()));
  }

  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_duplicate_body(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_free_count(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_row_bitmap(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_observation(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_score(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_head_index(env);
    CHECK(!env.check_invariants(0));
  }
  {
    snake::detail::VecSnakeImpl env(
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 8, 8));
    Access::corrupt_head_cell(env);
    CHECK(!env.check_invariants(0));
  }
}

void test_terminal_win_contract() {
  for (auto mode : {snake::AutoresetMode::NextStep, snake::AutoresetMode::Disabled}) {
    snake::detail::VecSnakeImpl env(make_config(mode, snake::ObsMode::Grid, 1, 3, 3));
    snake::detail::VecSnakeTestAccess::force_near_win(env);
    const std::int8_t straight = 0;
    env.step(&straight);

    CHECK(env.dones()[0] == 1);
    CHECK(env.wins()[0] == 1);
    CHECK(env.terminated()[0] == 1);
    CHECK(env.truncated()[0] == 0);
    CHECK(env.needs_reset()[0] == 1);
    CHECK(env.final_scores()[0] == 6);
    CHECK(env.final_lengths()[0] == 9);
    CHECK_NEAR(env.final_returns()[0], 3.0, 1e-6);
    CHECK(env.check_all_invariants());

    if (mode == snake::AutoresetMode::NextStep) {
      env.step(&straight);  // reset step: action ignored, outputs cleared
      CHECK(env.dones()[0] == 0);
      CHECK(env.wins()[0] == 0);
      CHECK(env.terminated()[0] == 0);
      CHECK(env.needs_reset()[0] == 0);
      CHECK(env.steps()[0] == 0);
      CHECK(env.lengths()[0] == 3);
      CHECK_NEAR(env.returns()[0], 0.0, 1e-6);
      CHECK(env.directions()[0] == 1);
    } else {
      const std::int32_t steps_before = env.steps()[0];
      env.step(&straight);  // sticky terminal state
      CHECK(env.dones()[0] == 1);
      CHECK(env.wins()[0] == 1);
      CHECK(env.needs_reset()[0] == 1);
      CHECK(env.steps()[0] == steps_before);
      CHECK_NEAR(env.returns()[0], 3.0, 1e-6);
    }
    CHECK(env.check_all_invariants());
  }
}

void test_terminal_truncation_contract() {
  for (auto mode : {snake::AutoresetMode::NextStep, snake::AutoresetMode::Disabled}) {
    {
      snake::VecConfig config = make_config(mode, snake::ObsMode::Grid, 1, 16, 16);
      config.max_episode_steps = 3;
      config.step_penalty = -0.5F;
      snake::detail::VecSnakeImpl env(config);
      snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
      const std::int8_t straight = 0;
      for (int step = 0; step < 3; ++step) {
        env.step(&straight);
      }
      CHECK(env.dones()[0] == 1);
      CHECK(env.truncated()[0] == 1);
      CHECK(env.terminated()[0] == 0);
      CHECK(env.wins()[0] == 0);
      CHECK(env.needs_reset()[0] == 1);
      CHECK(env.steps()[0] == 3);
      CHECK_NEAR(env.returns()[0], -1.5, 1e-6);
      CHECK_NEAR(env.final_returns()[0], -1.5, 1e-6);
      CHECK(env.check_all_invariants());

      if (mode == snake::AutoresetMode::NextStep) {
        env.step(&straight);
        CHECK(env.dones()[0] == 0);
        CHECK(env.truncated()[0] == 0);
        CHECK(env.steps()[0] == 0);
        CHECK_NEAR(env.returns()[0], 0.0, 1e-6);
      } else {
        env.step(&straight);
        CHECK(env.dones()[0] == 1);
        CHECK(env.truncated()[0] == 1);
        CHECK_NEAR(env.returns()[0], -1.5, 1e-6);
      }
    }

    {
      snake::VecConfig config = make_config(mode, snake::ObsMode::Grid, 1, 16, 16);
      config.max_steps_without_food = 3;
      config.step_penalty = -0.5F;
      snake::detail::VecSnakeImpl env(config);
      snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
      const std::int8_t straight = 0;
      for (int step = 0; step < 3; ++step) {
        env.step(&straight);
      }
      CHECK(env.dones()[0] == 1);
      CHECK(env.truncated()[0] == 1);
      CHECK(env.terminated()[0] == 0);
      CHECK(env.needs_reset()[0] == 1);
      CHECK(env.check_all_invariants());
      env.step(&straight);
      if (mode == snake::AutoresetMode::NextStep) {
        CHECK(env.dones()[0] == 0);
        CHECK(env.steps()[0] == 0);
      } else {
        CHECK(env.dones()[0] == 1);
        CHECK(env.truncated()[0] == 1);
      }
    }
  }
}

void test_exact_returns() {
  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
    config.step_penalty = -0.5F;
    config.food_reward = 5.0F;
    config.death_penalty = -7.0F;
    snake::detail::VecSnakeImpl env(config);

    const int head_x = snake::cell_x(env.head_cells()[0]);
    const int head_y = snake::cell_y(env.head_cells()[0]);
    snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(head_x + 1, head_y));
    const std::int8_t straight = 0;
    env.step(&straight);
    CHECK_NEAR(env.rewards()[0], 4.5, 1e-6);
    CHECK_NEAR(env.returns()[0], 4.5, 1e-6);

    snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
    float expected = env.returns()[0];
    for (int step = 0; step < 20 && env.dones()[0] == 0; ++step) {
      env.step(&straight);
      expected += env.rewards()[0];
      CHECK_NEAR(env.returns()[0], expected, 1e-6);
    }
    CHECK(env.dones()[0] == 1);
    CHECK_NEAR(expected, -6.0, 1e-6);
    CHECK_NEAR(env.final_returns()[0], -6.0, 1e-6);
  }

  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
    config.max_episode_steps = 3;
    config.step_penalty = -0.5F;
    snake::detail::VecSnakeImpl env(config);
    snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
    const std::int8_t straight = 0;
    for (int step = 0; step < 3; ++step) {
      env.step(&straight);
    }
    CHECK(env.truncated()[0] == 1);
    CHECK_NEAR(env.returns()[0], -1.5, 1e-6);
    CHECK_NEAR(env.final_returns()[0], -1.5, 1e-6);
  }

  {
    snake::VecConfig config =
        make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 3, 3);
    config.food_reward = 1.0F;
    config.win_bonus = 11.0F;
    config.step_penalty = 0.0F;
    snake::detail::VecSnakeImpl env(config);
    snake::detail::VecSnakeTestAccess::force_near_win(env);
    const std::int8_t straight = 0;
    env.step(&straight);
    CHECK(env.wins()[0] == 1);
    CHECK_NEAR(env.returns()[0], 12.0, 1e-6);
    CHECK_NEAR(env.final_returns()[0], 12.0, 1e-6);
  }
}

void test_hunger_counter_resets_on_eat() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
  config.max_steps_without_food = 3;
  snake::detail::VecSnakeImpl env(config);

  const int head_x = snake::cell_x(env.head_cells()[0]);
  const int head_y = snake::cell_y(env.head_cells()[0]);
  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(head_x + 1, head_y));
  const std::int8_t straight = 0;
  env.step(&straight);  // eat: resets the hunger counter
  CHECK(env.truncated()[0] == 0);

  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));
  env.step(&straight);
  CHECK(env.truncated()[0] == 0);
  env.step(&straight);
  CHECK(env.truncated()[0] == 0);
  env.step(&straight);
  CHECK(env.truncated()[0] == 1);
  CHECK(env.dones()[0] == 1);
}

void test_view_integration_matrix() {
  for (auto obs : {snake::ObsMode::NoObs, snake::ObsMode::Grid, snake::ObsMode::Bitboard}) {
    for (auto autoreset : {snake::AutoresetMode::SameStep, snake::AutoresetMode::NextStep,
                           snake::AutoresetMode::Disabled}) {
      snake::VecSnake env(make_config(autoreset, obs, 2, 4, 4));
      const bool grid = obs == snake::ObsMode::Grid;
      const bool same_step = autoreset == snake::AutoresetMode::SameStep;

      CHECK(env.bitboards().valid());
      CHECK(env.bitboards().rows() == 2);
      CHECK(env.bitboards().cols() == 4);
      CHECK(env.observations().valid() == grid);
      if (grid) {
        CHECK(env.observations().num_envs() == 2);
        CHECK(env.observations().channels() == 3);
        CHECK(env.observations().height() == 4);
        CHECK(env.observations().width() == 4);
      }
      CHECK(env.final_observations().valid() == (grid && same_step));
      CHECK(env.final_bitboards().valid() == same_step);
      CHECK(env.final_head_cells().valid() == same_step);
      CHECK(env.final_food_cells().valid() == same_step);
      CHECK(env.final_scores().valid());
      CHECK(env.final_lengths().valid());
      CHECK(env.final_returns().valid());
      CHECK(env.final_scores().size() == 2);
      CHECK(env.directions().valid());
      CHECK(env.head_cells().valid());
      CHECK(env.food_cells().valid());
    }
  }
}

void test_extreme_boards() {
  for (auto obs : {snake::ObsMode::Grid, snake::ObsMode::Bitboard}) {
    snake::VecSnake env(make_config(snake::AutoresetMode::SameStep, obs, 2, 64, 64));
    std::mt19937 rng(777);
    for (int step = 0; step < 1000; ++step) {
      const auto actions = random_actions(2, rng);
      env.step(actions.data());
      if (step % 97 == 0) {
        CHECK(env.check_all_invariants());
      }
    }
    CHECK(env.check_all_invariants());
  }

  for (auto mode : {snake::AutoresetMode::SameStep, snake::AutoresetMode::NextStep,
                    snake::AutoresetMode::Disabled}) {
    snake::VecConfig config = make_config(mode, snake::ObsMode::Grid, 2, 16, 16);
    config.initial_length = 1;
    snake::VecSnake env(config);
    std::mt19937 rng(778);
    for (int step = 0; step < 200; ++step) {
      const auto actions = random_actions(2, rng);
      env.step(actions.data());
      if (step % 31 == 0) {
        CHECK(env.check_all_invariants());
      }
    }
    CHECK(env.check_all_invariants());
  }
}

void test_zero_limits_disable_truncation() {
  snake::VecConfig config =
      make_config(snake::AutoresetMode::SameStep, snake::ObsMode::Grid, 1, 16, 16);
  config.relative_actions = false;
  config.max_episode_steps = 0;
  config.max_steps_without_food = 0;
  snake::detail::VecSnakeImpl env(config);
  snake::detail::VecSnakeTestAccess::force_food(env, snake::pack_cell(0, 0));

  // A safe 4x4 loop, repeated 8 times: 96 steps without a terminal state.
  std::vector<std::int8_t> actions;
  for (int cycle = 0; cycle < 8; ++cycle) {
    actions.insert(actions.end(), 3, static_cast<std::int8_t>(1));  // right
    actions.insert(actions.end(), 3, static_cast<std::int8_t>(2));  // down
    actions.insert(actions.end(), 3, static_cast<std::int8_t>(3));  // left
    actions.insert(actions.end(), 3, static_cast<std::int8_t>(0));  // up
  }
  CHECK(actions.size() == 96);
  for (std::int8_t action : actions) {
    env.step(&action);
  }
  CHECK(env.dones()[0] == 0);
  CHECK(env.steps()[0] == 96);
  CHECK(env.scores()[0] == 0);
  CHECK(env.lengths()[0] == config.initial_length);
  CHECK(env.check_all_invariants());
}

void run_engine_tests() {
  stress(snake::ObsMode::Grid, snake::AutoresetMode::SameStep, 16, 16, 16, 20000, 42);
  stress(snake::ObsMode::Grid, snake::AutoresetMode::NextStep, 16, 16, 16, 20000, 43);
  stress(snake::ObsMode::Bitboard, snake::AutoresetMode::SameStep, 16, 16, 16, 20000, 44);
  stress(snake::ObsMode::Bitboard, snake::AutoresetMode::NextStep, 16, 16, 16, 20000, 45);
  stress(snake::ObsMode::NoObs, snake::AutoresetMode::SameStep, 16, 16, 16, 20000, 46);
  stress(snake::ObsMode::Grid, snake::AutoresetMode::Disabled, 16, 16, 16, 20000, 47);

  test_wall_collision();
  test_disabled_reset_mask();
  test_determinism_and_threads();
  test_grid_bitboard_consistency();
  test_odd_boards();
  test_absolute_actions();
  test_step_limits();
  test_hunger_limit();
  test_auto_threads();
  test_initial_length_boundary();
  test_invalid_config();
  test_reseed();
  test_reseed_env_validation();
  test_directions();
  test_self_collision();
  test_into_tail();
  test_disabled_terminal_is_sticky();
  test_final_returns();
  test_reward_config();
  test_win_and_final_state();
  test_win_bitboard();
  test_invariant_checker_rejects_corruption();
  test_terminal_win_contract();
  test_terminal_truncation_contract();
  test_exact_returns();
  test_hunger_counter_resets_on_eat();
  test_view_integration_matrix();
  test_extreme_boards();
  test_zero_limits_disable_truncation();
}
