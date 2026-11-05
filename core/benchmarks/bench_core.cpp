#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "snake/vec_snake.hpp"

namespace {

struct Case {
  snake::ObsMode obs;
  int num_envs;
  int width;
  int height;
  int threads;
  int iterations;
};

struct Result {
  double env_steps_per_sec = 0.0;
  double batch_steps_per_sec = 0.0;
  double ns_per_env_step = 0.0;
  double done_rate = 0.0;
  int active_threads = 0;
};

constexpr int kRepetitions = 3;
constexpr int kWarmupSteps = 100;

const char* mode_name(snake::ObsMode mode) {
  switch (mode) {
    case snake::ObsMode::NoObs:
      return "none";
    case snake::ObsMode::Grid:
      return "grid";
    case snake::ObsMode::Bitboard:
      return "bitboard";
  }
  return "?";
}

// Same action stream for every mode/thread count within one (N, board) group,
// so rows are directly comparable.
std::uint64_t group_seed(const Case& c) {
  const std::uint64_t base = static_cast<std::uint64_t>(0x9E3779B97F4A7C15ULL);
  const std::uint64_t envs = static_cast<std::uint64_t>(c.num_envs) * std::uint64_t{1000003};
  const std::uint64_t width = static_cast<std::uint64_t>(c.width) * std::uint64_t{7919};
  const std::uint64_t height = static_cast<std::uint64_t>(c.height) * std::uint64_t{104729};
  return base ^ envs ^ width ^ height;
}

std::vector<std::int8_t> make_actions(const Case& c, std::uint64_t seed) {
  std::vector<std::int8_t> actions(static_cast<std::size_t>(c.num_envs) *
                                   static_cast<std::size_t>(c.iterations));
  std::uint64_t rng = seed != 0 ? seed : 1;
  for (auto& value : actions) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    value = static_cast<std::int8_t>(rng % 3);
  }
  return actions;
}

snake::VecConfig make_config(const Case& c) {
  snake::VecConfig cfg;
  cfg.num_envs = c.num_envs;
  cfg.width = c.width;
  cfg.height = c.height;
  cfg.initial_length = 3;
  cfg.max_episode_steps = 0;
  cfg.max_steps_without_food = 0;
  cfg.autoreset_mode = snake::AutoresetMode::SameStep;
  cfg.obs_mode = c.obs;
  cfg.seed = 987654321;
  cfg.num_threads = c.threads;
  cfg.parallel_threshold = 1;
  return cfg;
}

double run_once(const Case& c, const std::vector<std::int8_t>& actions, int* active_threads) {
  snake::VecSnake env(make_config(c));
  *active_threads = env.active_threads();

  for (int step = 0; step < kWarmupSteps; ++step) {
    const std::size_t offset =
        static_cast<std::size_t>(step % c.iterations) * static_cast<std::size_t>(c.num_envs);
    env.step(actions.data() + offset);
  }

  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < c.iterations; ++step) {
    const std::size_t offset =
        static_cast<std::size_t>(step) * static_cast<std::size_t>(c.num_envs);
    env.step(actions.data() + offset);
  }
  const auto end = std::chrono::steady_clock::now();

  const double seconds = std::chrono::duration<double>(end - start).count();
  return static_cast<double>(c.num_envs) * static_cast<double>(c.iterations) / seconds;
}

double measure_done_rate(const Case& c, const std::vector<std::int8_t>& actions) {
  snake::VecSnake env(make_config(c));
  std::uint64_t done_count = 0;
  for (int step = 0; step < c.iterations; ++step) {
    const std::size_t offset =
        static_cast<std::size_t>(step) * static_cast<std::size_t>(c.num_envs);
    env.step(actions.data() + offset);
    const auto dones = env.dones();
    for (int env_index = 0; env_index < c.num_envs; ++env_index) {
      done_count += dones[env_index];
    }
  }
  return static_cast<double>(done_count) /
         (static_cast<double>(c.num_envs) * static_cast<double>(c.iterations));
}

Result measure(const Case& c) {
  const std::vector<std::int8_t> actions = make_actions(c, group_seed(c));

  std::vector<double> rates;
  rates.reserve(kRepetitions);
  int active_threads = 0;
  for (int repetition = 0; repetition < kRepetitions; ++repetition) {
    rates.push_back(run_once(c, actions, &active_threads));
  }
  std::sort(rates.begin(), rates.end());

  Result result;
  result.env_steps_per_sec = rates[rates.size() / 2];
  result.batch_steps_per_sec = result.env_steps_per_sec / static_cast<double>(c.num_envs);
  result.ns_per_env_step = 1e9 / result.env_steps_per_sec;
  result.done_rate = measure_done_rate(c, actions);
  result.active_threads = active_threads;
  return result;
}

void print_result(const Case& c, const Result& r) {
  char thread_label[32];
  if (c.threads == 0) {
    std::snprintf(thread_label, sizeof(thread_label), "%d (auto)", r.active_threads);
  } else {
    std::snprintf(thread_label, sizeof(thread_label), "%d", r.active_threads);
  }
  std::printf(
      "%-10s N=%-5d board=%dx%d threads=%-9s | %8.2f M env/s | "
      "%10.1f batch/s | %7.2f ns/env-step | done=%5.1f%%\n",
      mode_name(c.obs), c.num_envs, c.width, c.height, thread_label, r.env_steps_per_sec / 1e6,
      r.batch_steps_per_sec, r.ns_per_env_step, r.done_rate * 100.0);
}

void run_case(snake::ObsMode obs, int num_envs, int width, int height, int threads,
              int iterations) {
  const Case c{obs, num_envs, width, height, threads, iterations};
  print_result(c, measure(c));
}

void run_group(int num_envs, int width, int height, int iterations, int threads) {
  for (snake::ObsMode obs :
       {snake::ObsMode::NoObs, snake::ObsMode::Grid, snake::ObsMode::Bitboard}) {
    run_case(obs, num_envs, width, height, threads, iterations);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    }
  }

#ifndef NDEBUG
  std::printf("WARNING: built without NDEBUG; throughput numbers are not representative.\n");
#endif
  std::printf("snake core benchmark (random actions, SameStep autoreset, median of %d runs)\n",
              kRepetitions);
  std::printf("machine reports %u hardware threads; auto leaves one core free\n\n",
              std::thread::hardware_concurrency());

  if (quick) {
    std::printf("16x16, serial vs auto:\n");
    run_group(1024, 16, 16, 1000, 1);
    run_group(1024, 16, 16, 4000, 0);
    return 0;
  }

  std::printf("16x16, serial:\n");
  for (int num_envs : {256, 1024, 4096}) {
    run_group(num_envs, 16, 16, 1000, 1);
  }

  std::printf("\n16x16, auto threads:\n");
  for (int num_envs : {256, 1024, 4096}) {
    run_group(num_envs, 16, 16, 4000, 0);
  }

  std::printf("\n64x64, serial:\n");
  for (int num_envs : {256, 1024}) {
    run_group(num_envs, 64, 64, 300, 1);
  }

  std::printf("\n64x64, auto threads:\n");
  for (int num_envs : {256, 1024}) {
    run_group(num_envs, 64, 64, 1200, 0);
  }

  std::printf("\n4096 envs, 16x16, bitboard, thread scaling:\n");
  for (int threads : {1, 2, 4, 8}) {
    run_case(snake::ObsMode::Bitboard, 4096, 16, 16, threads, 2000);
  }
  return 0;
}
