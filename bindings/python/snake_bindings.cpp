// Python bindings for the snake core engine.
//
// Design notes:
//  * Every buffer getter returns a zero-copy numpy view. The `owner` argument
//    is passed as numpy's base object: pybind11 3.x silently copies the data
//    when a pointer is given without a base, and the base also keeps the
//    VecSnake instance alive for as long as the array exists.
//  * step()/reset*() release the GIL around the C++ call, but the buffer
//    protocol lookup (py::buffer_info) happens while the GIL is still held.
//  * Views stay valid for the lifetime of the VecSnake instance; step()/reset()
//    overwrite their contents, so copy if the previous values are needed.
//  * A single VecSnake must not be driven from multiple Python threads at the
//    same time, even though step() releases the GIL.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "snake/vec_snake.hpp"

namespace py = pybind11;
using snake::VecSnake;

namespace {

template <typename T>
py::array_t<T> view(py::handle owner, std::vector<py::ssize_t> shape, const T* ptr) {
  // The base object is required: pybind11 copies the data when a pointer is
  // passed without one, and it also keeps the engine alive while the view
  // exists. The core exposes const buffers, so the array is read-only.
  py::array_t<T> array(std::move(shape), ptr, owner);
  array.attr("flags").attr("writeable") = false;
  return array;
}

py::array_t<std::uint8_t> obs_view(py::handle owner, snake::GridView grid) {
  if (!grid.valid()) {
    throw std::runtime_error("observations() is only available with ObsMode::Grid");
  }
  return view<std::uint8_t>(owner, {grid.num_envs(), grid.channels(), grid.height(), grid.width()},
                            grid.data());
}

py::array_t<std::uint64_t> bitboard_view(py::handle owner, snake::BoardView board) {
  // Occupancy is maintained for every observation mode, so this never depends
  // on ObsMode.
  if (!board.valid()) {
    throw std::runtime_error("bitboards() is unavailable");
  }
  return view<std::uint64_t>(owner, {board.rows(), board.cols()}, board.data());
}

py::object optional_obs(py::handle owner, snake::GridView grid) {
  if (!grid.valid()) {
    return py::none();
  }
  return obs_view(owner, grid);
}

py::object optional_bitboards(py::handle owner, snake::BoardView board) {
  if (!board.valid()) {
    return py::none();
  }
  return bitboard_view(owner, board);
}

template <typename T, snake::VectorView<T> (VecSnake::*Getter)() const noexcept>
py::array_t<T> column(py::object self) {
  const auto& env = self.cast<const VecSnake&>();
  const snake::VectorView<T> values = (env.*Getter)();
  return view<T>(self, {values.size()}, values.data());
}

template <typename T, snake::VectorView<T> (VecSnake::*Getter)() const noexcept>
py::object optional_column(py::object self) {
  const auto& env = self.cast<const VecSnake&>();
  const snake::VectorView<T> values = (env.*Getter)();
  if (!values.valid()) {
    return py::none();
  }
  return view<T>(self, {values.size()}, values.data());
}

}  // namespace

PYBIND11_MODULE(_snake_core, m) {
  m.doc() = "CPU batched snake engine (low-level bindings)";

  py::enum_<snake::ObsMode>(m, "ObsMode")
      .value("NoObs", snake::ObsMode::NoObs)
      .value("Grid", snake::ObsMode::Grid)
      .value("Bitboard", snake::ObsMode::Bitboard);

  py::enum_<snake::AutoresetMode>(m, "AutoresetMode")
      .value("NextStep", snake::AutoresetMode::NextStep)
      .value("SameStep", snake::AutoresetMode::SameStep)
      .value("Disabled", snake::AutoresetMode::Disabled);

  py::class_<snake::VecConfig>(m, "VecConfig")
      .def(py::init<>())
      .def_readwrite("num_envs", &snake::VecConfig::num_envs)
      .def_readwrite("width", &snake::VecConfig::width)
      .def_readwrite("height", &snake::VecConfig::height)
      .def_readwrite("initial_length", &snake::VecConfig::initial_length)
      .def_readwrite("max_episode_steps", &snake::VecConfig::max_episode_steps)
      .def_readwrite("max_steps_without_food", &snake::VecConfig::max_steps_without_food)
      .def_readwrite("food_reward", &snake::VecConfig::food_reward)
      .def_readwrite("death_penalty", &snake::VecConfig::death_penalty)
      .def_readwrite("step_penalty", &snake::VecConfig::step_penalty)
      .def_readwrite("win_bonus", &snake::VecConfig::win_bonus)
      .def_readwrite("relative_actions", &snake::VecConfig::relative_actions)
      .def_readwrite("autoreset_mode", &snake::VecConfig::autoreset_mode)
      .def_readwrite("obs_mode", &snake::VecConfig::obs_mode)
      .def_readwrite("seed", &snake::VecConfig::seed)
      .def_readwrite("num_threads", &snake::VecConfig::num_threads)
      .def_readwrite("parallel_threshold", &snake::VecConfig::parallel_threshold);

  py::class_<VecSnake, std::shared_ptr<VecSnake>>(
      m, "VecSnake",
      "Batched snake engine. Use one instance per thread; concurrent step() "
      "calls on the same instance are not safe even though the GIL is released.")
      .def(py::init<const snake::VecConfig&>(), py::arg("config"))

      // Write side. Buffer lookup under the GIL, engine call without it.
      .def(
          "step",
          [](VecSnake& env,
             py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> actions) {
            py::buffer_info buf = actions.request();
            if (buf.size != env.num_envs()) {
              throw std::invalid_argument("actions must have num_envs elements");
            }
            const auto* ptr = static_cast<const std::int8_t*>(buf.ptr);
            {
              py::gil_scoped_release unlock;
              env.step(ptr);
            }
          },
          py::arg("actions"), "Advance all environments by one step.")
      .def(
          "reset_all",
          [](VecSnake& env) {
            py::gil_scoped_release unlock;
            env.reset_all();
          },
          "Reset every environment.")
      .def(
          "reset_mask",
          [](VecSnake& env,
             py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask) {
            py::buffer_info buf = mask.request();
            if (buf.size != env.num_envs()) {
              throw std::invalid_argument("mask must have num_envs elements");
            }
            const auto* ptr = static_cast<const std::uint8_t*>(buf.ptr);
            py::gil_scoped_release unlock;
            env.reset_mask(ptr);
          },
          py::arg("mask"), "Reset the environments whose mask entry is non-zero.")
      .def("reseed_all", &VecSnake::reseed_all, py::arg("seed"))
      .def("reseed_env", &VecSnake::reseed_env, py::arg("env"), py::arg("seed"))

      .def_property_readonly("num_envs", &VecSnake::num_envs)
      .def_property_readonly("width", &VecSnake::width)
      .def_property_readonly("height", &VecSnake::height)
      .def_property_readonly("cells", &VecSnake::cells)
      .def_property_readonly("active_threads", &VecSnake::active_threads)
      .def_property_readonly("config", &VecSnake::config, py::return_value_policy::copy)

      // Observation views. `keep_alive` is redundant with the base object but
      // documents intent and protects against future refactors.
      .def(
          "observations",
          [](py::object self) {
            const auto& env = self.cast<const VecSnake&>();
            return obs_view(self, env.observations());
          },
          py::keep_alive<0, 1>(), "Grid observations, (N,3,H,W) uint8.")
      .def(
          "final_observations",
          [](py::object self) -> py::object {
            const auto& env = self.cast<const VecSnake&>();
            return optional_obs(self, env.final_observations());
          },
          py::keep_alive<0, 1>(),
          "Terminal Grid observations, (N,3,H,W) uint8; None unless Grid + SameStep.")
      .def(
          "bitboards",
          [](py::object self) {
            const auto& env = self.cast<const VecSnake&>();
            return bitboard_view(self, env.bitboards());
          },
          py::keep_alive<0, 1>(), "Body occupancy, (N,H) uint64; available in every mode.")
      .def(
          "final_bitboards",
          [](py::object self) -> py::object {
            const auto& env = self.cast<const VecSnake&>();
            return optional_bitboards(self, env.final_bitboards());
          },
          py::keep_alive<0, 1>(), "Terminal body occupancy, (N,H) uint64; None unless SameStep.")

      .def("rewards", &column<float, &VecSnake::rewards>, py::keep_alive<0, 1>())
      .def("returns", &column<float, &VecSnake::returns>, py::keep_alive<0, 1>())
      .def("final_returns", &column<float, &VecSnake::final_returns>, py::keep_alive<0, 1>())
      .def("dones", &column<std::uint8_t, &VecSnake::dones>, py::keep_alive<0, 1>())
      .def("terminated", &column<std::uint8_t, &VecSnake::terminated>, py::keep_alive<0, 1>())
      .def("truncated", &column<std::uint8_t, &VecSnake::truncated>, py::keep_alive<0, 1>())
      .def("wins", &column<std::uint8_t, &VecSnake::wins>, py::keep_alive<0, 1>())
      .def("needs_reset", &column<std::uint8_t, &VecSnake::needs_reset>, py::keep_alive<0, 1>())
      .def("scores", &column<std::int32_t, &VecSnake::scores>, py::keep_alive<0, 1>())
      .def("lengths", &column<std::int32_t, &VecSnake::lengths>, py::keep_alive<0, 1>())
      .def("steps", &column<std::int32_t, &VecSnake::steps>, py::keep_alive<0, 1>())
      .def("final_scores", &column<std::int32_t, &VecSnake::final_scores>, py::keep_alive<0, 1>())
      .def("final_lengths", &column<std::int32_t, &VecSnake::final_lengths>, py::keep_alive<0, 1>())

      .def("head_cells", &column<std::uint16_t, &VecSnake::head_cells>, py::keep_alive<0, 1>())
      .def("directions", &column<std::uint8_t, &VecSnake::directions>, py::keep_alive<0, 1>(),
           "Heading per env: 0=up, 1=right, 2=down, 3=left.")
      .def("food_cells", &column<std::uint16_t, &VecSnake::food_cells>, py::keep_alive<0, 1>())
      .def("final_head_cells", &optional_column<std::uint16_t, &VecSnake::final_head_cells>,
           py::keep_alive<0, 1>(), "Terminal packed head cells, (N,) uint16; None unless SameStep.")
      .def("final_food_cells", &optional_column<std::uint16_t, &VecSnake::final_food_cells>,
           py::keep_alive<0, 1>(), "Terminal packed food cells, (N,) uint16; None unless SameStep.")

      .def("check_all_invariants", &VecSnake::check_all_invariants);
}
