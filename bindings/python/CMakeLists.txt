set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_snake_core snake_bindings.cpp)
target_link_libraries(_snake_core PRIVATE snake_core)
snake_configure_warnings(_snake_core)

if(MINGW)
  # Windows 上的扩展模块必须自带 GCC 运行时：Python 3.8 起扩展的依赖 DLL
  # 不再走 PATH 搜索，动态依赖 libgcc_s_seh-1 / libstdc++-6 / libwinpthread-1
  # 会让 `import snake_drl` 直接报 "DLL load failed"，除非用户手工
  # os.add_dll_directory 或装 MSYS2 运行时。`-static` 让 g++ 把这三个运行时库
  # 全部静态链入（只写 -static-libgcc/-static-libstdc++ 时 libwinpthread 仍然
  # 是动态依赖，已用 objdump 验证）；这样 wheel 自包含，普通 Windows Python
  # 解压即用。只影响 MinGW，MSVC/POSIX 分支不变。
  target_link_options(_snake_core PRIVATE -static)
endif()

install(TARGETS _snake_core LIBRARY DESTINATION snake_drl)
