#include "detail/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>

namespace snake::detail {
namespace {

constexpr int kMaxEnvCount = 65536;
constexpr int kMaxThreadCount = 256;
constexpr int kAutoThreadCap = 5;  // leaves room for the caller on wide machines

void validate_config(const VecConfig& config) {
  if (config.num_envs < 1 || config.num_envs > kMaxEnvCount) {
    throw std::invalid_argument("num_envs must be in [1, 65536]");
  }
  if (config.width < kMinDimension || config.width > kMaxDimension ||
      config.height < kMinDimension || config.height > kMaxDimension) {
    throw std::invalid_argument("width/height must be in [3, 64]");
  }
  if (config.initial_length < 1 || config.initial_length > config.width ||
      config.initial_length >= config.width * config.height) {
    throw std::invalid_argument(
        "initial_length must be in [1, width] and smaller than width*height");
  }
  if (config.max_episode_steps < 0 || config.max_steps_without_food < 0) {
    throw std::invalid_argument("step limits must be non-negative");
  }
  if (!std::isfinite(config.food_reward) || !std::isfinite(config.death_penalty) ||
      !std::isfinite(config.step_penalty) || !std::isfinite(config.win_bonus)) {
    throw std::invalid_argument("rewards must be finite");
  }
  if (config.num_threads < 0 || config.num_threads > kMaxThreadCount) {
    throw std::invalid_argument("num_threads must be in [0, 256]");
  }
  if (config.parallel_threshold < 1) {
    throw std::invalid_argument("parallel_threshold must be >= 1");
  }
  const int autoreset = static_cast<int>(config.autoreset_mode);
  if (autoreset < 0 || autoreset > 2) {
    throw std::invalid_argument("invalid autoreset_mode");
  }
  const int obs = static_cast<int>(config.obs_mode);
  if (obs < 0 || obs > 2) {
    throw std::invalid_argument("invalid obs_mode");
  }
}

int resolve_total_threads(const VecConfig& config, int num_envs) noexcept {
  int requested = config.num_threads;
  if (requested == 0) {
    const unsigned hardware = std::thread::hardware_concurrency();
    const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
    // Leave one core for the caller and OS; cap at 5 on wide machines.
    requested = std::max(1, std::min(available - 1, kAutoThreadCap));
  }
  if (requested <= 1 || num_envs < config.parallel_threshold) {
    return 1;
  }
  return std::min(requested, num_envs);
}

}  // namespace

VecSnakeImpl::VecSnakeImpl(const VecConfig& config) : config_(config) {
  validate_config(config_);

  n_ = config_.num_envs;
  width_ = config_.width;
  height_ = config_.height;
  cells_ = width_ * height_;

  const std::size_t env_count = static_cast<std::size_t>(n_);
  const std::size_t cell_count = static_cast<std::size_t>(cells_);
  const std::size_t row_count = static_cast<std::size_t>(height_);

  rows_.assign(env_count * row_count, 0);
  body_.assign(env_count * cell_count, 0);
  free_.assign(env_count * cell_count, 0);
  free_pos_.assign(env_count * cell_count, 0);
  head_.assign(env_count, 0);
  food_.assign(env_count, kInvalidCell);
  head_idx_.assign(env_count, 0);
  tail_idx_.assign(env_count, 0);
  len_.assign(env_count, config_.initial_length);
  score_.assign(env_count, 0);
  steps_.assign(env_count, 0);
  steps_no_food_.assign(env_count, 0);
  free_count_.assign(env_count, 0);
  dir_.assign(env_count, 1);
  done_.assign(env_count, 0);
  terminated_.assign(env_count, 0);
  truncated_.assign(env_count, 0);
  won_.assign(env_count, 0);
  needs_reset_.assign(env_count, 0);
  rng_.resize(env_count);

  rewards_.assign(env_count, 0.0F);
  returns_.assign(env_count, 0.0F);
  final_returns_.assign(env_count, 0.0F);
  episode_return_.assign(env_count, 0.0F);

  final_score_.assign(env_count, 0);
  final_len_.assign(env_count, 0);

  if (config_.autoreset_mode == AutoresetMode::SameStep) {
    final_head_.assign(env_count, kInvalidCell);
    final_food_.assign(env_count, kInvalidCell);
    // The occupancy mirror exists for every observation mode, so preserve it
    // for NoObs/Grid too: with SameStep the live board is reset inside the
    // terminal step and this is the only copy of the terminal state.
    final_rows_.assign(env_count * row_count, 0);
  }
  if (config_.obs_mode == ObsMode::Grid) {
    obs_.assign(env_count * 3 * cell_count, 0);
    if (config_.autoreset_mode == AutoresetMode::SameStep) {
      final_obs_.assign(env_count * 3 * cell_count, 0);
    }
  }

  for (int env = 0; env < n_; ++env) {
    rng_[idx(env)].seed(config_.seed + static_cast<std::uint64_t>(env));
    init_env(env);
  }

  pool_ = std::make_unique<ThreadPool>(resolve_total_threads(config_, n_), n_);
}

void VecSnakeImpl::reseed_all(std::uint64_t seed) noexcept {
  for (int env = 0; env < n_; ++env) {
    rng_[idx(env)].seed(seed + static_cast<std::uint64_t>(env));
  }
  // Rebuild canonically. reset_all() would reuse the history-dependent free
  // list and therefore make the same seed produce different episodes.
  pool_->parallel_for([this](int begin, int end) noexcept {
    for (int env = begin; env < end; ++env) {
      init_env(env);
    }
  });
}

void VecSnakeImpl::reseed_env(int env, std::uint64_t seed) {
  if (env < 0 || env >= n_) {
    throw std::out_of_range("environment index out of range");
  }
  rng_[idx(env)].seed(seed);
  // Rebuild this environment from scratch for the same reason as reseed_all.
  init_env(env);
}

}  // namespace snake::detail
