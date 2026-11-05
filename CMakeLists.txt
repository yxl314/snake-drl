cmake_minimum_required(VERSION 3.20)
project(snake_drl LANGUAGES CXX)

# Needed so `ctest` from the top-level build directory discovers core tests.
enable_testing()

# Monorepo entry point. The engine lives in core/ and stays framework-agnostic;
# built as a standalone project it can be configured, installed and consumed
# without this file.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/core/cmake")
add_subdirectory(core)

# Python bindings and the training/play side live beside core/ so the engine
# itself stays framework-agnostic.
option(SNAKE_BUILD_PYTHON "Build pybind11 bindings" OFF)
if(SNAKE_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()
