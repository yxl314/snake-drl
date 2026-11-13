# stacklog 的内嵌副本：随本仓库一起分发，不依赖外部安装。
#
# 只构建静态库本身；安装/导出规则由上层项目负责（本副本不对外安装）。

add_library(stacklog STATIC src/stacklog.cpp)
add_library(snake::stacklog ALIAS stacklog)

# 位置无关：本库有 thread_local 状态，snake_core 会被链进 Python 扩展 .so，
# 非 PIC 的 TLS 会产生 local-exec 重定位，链接器拒绝。
set_target_properties(stacklog PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(stacklog PUBLIC cxx_std_11)

target_include_directories(stacklog PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stacklog PRIVATE -Wall -Wextra -Wpedantic)

endif()

find_package(Threads REQUIRED)
target_link_libraries(stacklog PUBLIC Threads::Threads)

# 随 snake_core 一起导出：snake_core 以 PRIVATE 方式静态链接它，链接期仍需
# 该目标可见，所以必须进同一个导出集。
include(GNUInstallDirs)
install(TARGETS stacklog
  EXPORT SnakeCoreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/stacklog DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
