cmake_minimum_required(VERSION 3.16)

# stacklog 的内嵌副本（snake-drl 使用子集）。
#
# 同步自上游 stacklog commit 8fd01bc；与上游相比只保留构建静态库所需的
# include/、src/、cmake/ 与本文，文件内容保持上游一致。
#
# 与上游 CMakeLists.txt 的差异（保持 snake_core 集成）：
#   1. 安装规则由上层项目负责：库与头文件随 snake_core 一起安装，目标以
#      stacklog:: 命名空间进入独立的 StacklogTargets 导出集，由
#      snake_coreConfig.cmake 一并加载；
#   2. 上游的 tests/examples/benchmarks 子工程不随本副本分发，故不注册；
#   3. 保留 MSVC 的 /W4 /utf-8 分支（上游 Windows 支持的约定），MinGW/g++ 走
#      -Wall -Wextra -Wpedantic。
project(stacklog
    VERSION 0.2.0
    DESCRIPTION "面向 C++ / Qt 项目的轻量日志库"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# 平台端口：核心 stacklog.cpp 不含平台宏，适配器按平台二选一编译。
if(WIN32)
    set(STACKLOG_PLATFORM_SOURCE src/platform_win32.cpp)
else()
    set(STACKLOG_PLATFORM_SOURCE src/platform_posix.cpp)
endif()

add_library(stacklog STATIC src/stacklog.cpp ${STACKLOG_PLATFORM_SOURCE})
add_library(stacklog::stacklog ALIAS stacklog)

# 静态库也要位置无关：本库持有 thread_local 状态，被链进共享对象（例如
# Python 扩展 .so）时，非 PIC 的目标文件会产生 local-exec 的 TLS 重定位，
# 链接器会直接拒绝：
#   relocation R_X86_64_TPOFF32 ... can not be used when making a shared object
set_target_properties(stacklog PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(stacklog PUBLIC cxx_std_11)

target_include_directories(stacklog PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stacklog PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    # 与上游 Windows 支持一致：/W4 严格告警，/utf-8 固定源码与执行字符集（本库
    # 源码含中文注释与错误文本，不依赖系统代码页）；不加 /WX，第三方头文件的
    # 告警不阻断构建。
    target_compile_options(stacklog PRIVATE /W4 /utf-8)
endif()

find_package(Threads REQUIRED)
target_link_libraries(stacklog PUBLIC Threads::Threads)

# snake_core 以 PRIVATE 方式静态链接它，链接期仍需该目标可见；安装时用独立
# 导出集与上游命名空间导出，snake_coreConfig.cmake 负责在 SnakeCoreTargets
# 之前加载它（安装/导出规则由上层项目负责）。
install(TARGETS stacklog
    EXPORT StacklogTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/stacklog DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT StacklogTargets
    FILE stacklogTargets.cmake
    NAMESPACE stacklog::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stacklog)
