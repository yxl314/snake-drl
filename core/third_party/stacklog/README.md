# stacklog（内嵌副本）

本目录是 [stacklog](../../../../stacklog) 的使用子集，随 `snake_core` 一起分发。

- 同步自上游 commit `8fd01bc`（公开 API 与 0.2.0 版本不变；本次更新为上游
  平台层拆分的后续排版/Doxygen 整理与 cxx 工具链配置统一）。
- 只保留构建静态库实际需要的 `include/`、`src/`、`cmake/`、`CMakeLists.txt`
  与 `LICENSE`；上游的 tests/、examples/、benchmarks/、docs/ 与门禁配置不随
  snake-drl 分发。
- 版本见 `include/stacklog/version.hpp`；许可见 `LICENSE`。
- 只构建静态库；安装与导出由上层项目负责（见下面“与上游的差异”）。

## 与上游的差异

同步后本目录与上游的文件内容一致，只多一个**裁剪过的 `CMakeLists.txt`**：

1. 安装规则由上层项目负责：库与头文件随 `snake_core` 一起安装，目标以
   `stacklog::stacklog` 进入独立的 `StacklogTargets` 导出集，由
   `snake_coreConfig.cmake` 一并加载（不生成独立的 stacklog CMake 包）；
2. 不注册上游的 tests/examples/benchmarks 子工程（这些目录不在副本内）；
3. 保留上游 Windows 支持的 MSVC `/W4 /utf-8` 分支；MinGW/g++ 走
   `-Wall -Wextra -Wpedantic`；
4. 打开 `POSITION_INDEPENDENT_CODE`：本库内部有 `thread_local` 状态，
   `snake_core` 会被链进 Python 扩展 `.so`，非 PIC 的 TLS 访问会产生
   local-exec 重定位，链接器直接报错。

升级方式：从上游仓库重新拷贝 `include/`、`src/`、`cmake/`、`LICENSE`，保持本目录
只多出 `CMakeLists.txt` 与这份说明。

## Windows 要点（来自上游 `docs/windows支持.md`）

- `config.dir` / `config.base_name` 在 Windows 上按 **UTF-8** 解释，库内部转 UTF-16
  后调用宽字符文件 API；`\\?\` 扩展长度路径必须使用反斜杠。
- 平台适配器：`src/platform_win32.cpp` 在 `WIN32` 构建里替代
  `src/platform_posix.cpp`；目标是 STATIC 库，不需要 `__declspec(...)`。
- MSVC 编译必须加 `/utf-8`（源码是 UTF-8 无 BOM 且含中文）：vendored
  `CMakeLists.txt` 已按上游配置 `/W4 /utf-8`；MinGW/g++ 无需额外选项。
- POSIX 专属场景（外部删除/替换已打开的文件、`/dev/full`、symlink）在 Windows
  不适用，上游测试已按 `_WIN32` 跳过；本目录只 vendor 库源码，不含 tests。

校验同步内容可用 `git hash-object` 对比上游 commit `8fd01bc`：

```text
include/stacklog/stacklog.hpp   034a515cfe8f9c55af0a30471f86a51cec7ed77a
include/stacklog/version.hpp    f6afd8b3df4fd7ea90e761c9fee28bcdec9f5768
src/stacklog.cpp                b0a79720bc7d8c8019b2e5bc3aa59b9a1781b9e9
src/platform.hpp                518bf67309af86aeb5d6d3bcb51d197b0ef80eb1
src/platform_posix.cpp          4af374314ea271aec9e4500441da0288a460d399
src/platform_win32.cpp          858b0106e19c43e2d252f52f9707f943054ad151
src/platform_dirs.hpp           2789872363e4a5c8da0f7d6ac3cc273e8143789d
src/platform_time.hpp           7a49e517b8a752a3ca5a496d9b60377358c43d09
src/platform_paths.hpp          ba04560202394e73c408a226f0e7b32c45ff7ddb
src/platform_stream.hpp         bc5c2e93e3dc01b107f6ca4d8b5643d388ae89b3
cmake/stacklogConfig.cmake.in   744510bf3e6e62a32872d4f4baf158bb9b3ca843
```
