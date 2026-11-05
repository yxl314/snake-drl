#include "snake/vec_snake.hpp"

#include <cassert>
#include <memory>

#include "detail/engine.hpp"

namespace snake {

VecSnake::VecSnake(const VecConfig& config)
    : impl_(std::make_unique<detail::VecSnakeImpl>(config)) {}

VecSnake::~VecSnake() = default;

void VecSnake::step(const std::int8_t* actions) noexcept {
  assert(actions != nullptr);
  impl_->step(actions);
}

void VecSnake::reset_all() noexcept {
  impl_->reset_all();
}

void VecSnake::reset_mask(const std::uint8_t* mask) noexcept {
  assert(mask != nullptr);
  impl_->reset_mask(mask);
}

void VecSnake::reseed_all(std::uint64_t seed) noexcept {
  impl_->reseed_all(seed);
}

void VecSnake::reseed_env(int env, std::uint64_t seed) {
  impl_->reseed_env(env, seed);
}

const VecConfig& VecSnake::config() const noexcept {
  return impl_->config();
}

int VecSnake::num_envs() const noexcept {
  return impl_->num_envs();
}

int VecSnake::width() const noexcept {
  return impl_->width();
}

int VecSnake::height() const noexcept {
  return impl_->height();
}

int VecSnake::cells() const noexcept {
  return impl_->cells();
}

int VecSnake::active_threads() const noexcept {
  return impl_->active_threads();
}

GridView VecSnake::observations() const noexcept {
  return impl_->observations();
}

GridView VecSnake::final_observations() const noexcept {
  return impl_->final_observations();
}

BoardView VecSnake::bitboards() const noexcept {
  return impl_->bitboards();
}

BoardView VecSnake::final_bitboards() const noexcept {
  return impl_->final_bitboards();
}

CellView VecSnake::head_cells() const noexcept {
  return impl_->head_cells();
}

CellView VecSnake::food_cells() const noexcept {
  return impl_->food_cells();
}

CellView VecSnake::final_head_cells() const noexcept {
  return impl_->final_head_cells();
}

CellView VecSnake::final_food_cells() const noexcept {
  return impl_->final_food_cells();
}

FloatView VecSnake::rewards() const noexcept {
  return impl_->rewards();
}

FloatView VecSnake::returns() const noexcept {
  return impl_->returns();
}

FloatView VecSnake::final_returns() const noexcept {
  return impl_->final_returns();
}

ByteView VecSnake::directions() const noexcept {
  return impl_->directions();
}

Int32View VecSnake::scores() const noexcept {
  return impl_->scores();
}

Int32View VecSnake::lengths() const noexcept {
  return impl_->lengths();
}

Int32View VecSnake::steps() const noexcept {
  return impl_->steps();
}

Int32View VecSnake::final_scores() const noexcept {
  return impl_->final_scores();
}

Int32View VecSnake::final_lengths() const noexcept {
  return impl_->final_lengths();
}

ByteView VecSnake::dones() const noexcept {
  return impl_->dones();
}

ByteView VecSnake::terminated() const noexcept {
  return impl_->terminated();
}

ByteView VecSnake::truncated() const noexcept {
  return impl_->truncated();
}

ByteView VecSnake::wins() const noexcept {
  return impl_->wins();
}

ByteView VecSnake::needs_reset() const noexcept {
  return impl_->needs_reset();
}

bool VecSnake::check_invariants(int env) const {
  return impl_->check_invariants(env);
}

bool VecSnake::check_all_invariants() const {
  return impl_->check_all_invariants();
}

}  // namespace snake
