cmake_minimum_required(VERSION 3.20)
project(snake_core VERSION 1.1.0 LANGUAGES CXX)

option(SNAKE_BUILD_TESTS "Build the C++ test suite" ON)
option(SNAKE_BUILD_BENCHMARKS "Build the C++ benchmark" ON)
option(SNAKE_ENABLE_IPO "Enable interprocedural optimization in Release" OFF)
option(SNAKE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(SNAKE_STRICT_WARNINGS "Enable the extended warning set" ON)
option(SNAKE_ENABLE_INSTALL "Generate install/export rules for the C++ package" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SnakeWarnings)

include(GNUInstallDirs)

add_library(snake_core STATIC
  src/vec_snake.cpp
  src/engine.cpp
  src/environment.cpp
  src/invariants.cpp
  src/thread_pool.cpp
)
set_target_properties(snake_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(snake_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(snake_core PUBLIC cxx_std_17)

snake_configure_warnings(snake_core)

if(SNAKE_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SNAKE_IPO_SUPPORTED OUTPUT SNAKE_IPO_MESSAGE)
  if(SNAKE_IPO_SUPPORTED)
    set_target_properties(snake_core PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(STATUS "IPO not available: ${SNAKE_IPO_MESSAGE}")
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(snake_core PUBLIC Threads::Threads)

if(SNAKE_BUILD_TESTS)
  enable_testing()

  add_executable(snake_core_tests
    tests/test_main.cpp
    tests/test_bitboard.cpp
    tests/test_free_cells.cpp
    tests/test_rng.cpp
    tests/test_thread_pool.cpp
    tests/test_view.cpp
    tests/test_engine.cpp
    tests/support/test_access.cpp
  )
  target_link_libraries(snake_core_tests PRIVATE snake_core)
  target_include_directories(snake_core_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/support
  )
  snake_configure_warnings(snake_core_tests)
  if(SNAKE_IPO_SUPPORTED)
    set_target_properties(snake_core_tests PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
  add_test(NAME snake_core_tests COMMAND snake_core_tests)
endif()

if(SNAKE_BUILD_BENCHMARKS)
  add_executable(snake_benchmark benchmarks/bench_core.cpp)
  target_link_libraries(snake_benchmark PRIVATE snake_core)
  snake_configure_warnings(snake_benchmark)
  if(SNAKE_IPO_SUPPORTED)
    set_target_properties(snake_benchmark PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

if(SNAKE_ENABLE_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS snake_core
    EXPORT SnakeCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY include/snake DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT SnakeCoreTargets
    FILE SnakeCoreTargets.cmake
    NAMESPACE snake::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snake_core
  )

  configure_package_config_file(
    cmake/snake_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/snake_coreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snake_core
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/snake_coreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/snake_coreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/snake_coreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snake_core
  )
endif()
