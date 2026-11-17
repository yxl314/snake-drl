# stacklog（内嵌副本）

本目录是 [stacklog](../../../../stacklog) 的副本，随 `snake_core` 一起分发。

- 同步自上游 commit `0189fe3`（平台层拆分：核心 `stacklog.cpp` 不再含平台宏，
  POSIX/Windows 适配器按平台分别编译；公开 API 与 0.2.0 版本不变）。
- 版本见 `include/stacklog/version.hpp`
- 许可见 `LICENSE`
- 只构建静态库；安装与导出由上层项目负责

与上游相比只有两处差异：(a) 多一个只构建静态库的 `CMakeLists.txt`（按 `WIN32`
选择 `src/platform_posix.cpp` 或 `src/platform_win32.cpp`）；(b) 该
`CMakeLists.txt` 显式打开 `POSITION_INDEPENDENT_CODE`。原因：本库内部有
`thread_local` 状态，`snake_core` 会被链进 Python 扩展 `.so`，非 PIC 的 TLS 访问
会产生 local-exec 重定位，链接器直接报错。

升级方式：从上游仓库重新拷贝 `include/`、`src/`、`LICENSE`，保持本目录只多出
`CMakeLists.txt` 与这份说明。

## Windows 要点（来自上游 `docs/windows支持.md`）

- `config.dir` / `config.base_name` 在 Windows 上按 **UTF-8** 解释，库内部转 UTF-16
  后调用宽字符文件 API；`\\?\` 扩展长度路径必须使用反斜杠。
- 平台适配器：`src/platform_win32.cpp` 在 `WIN32` 构建里替代
  `src/platform_posix.cpp`；目标是 STATIC 库，不需要 `__declspec(...)`。
- MSVC 编译必须加 `/utf-8`（源码是 UTF-8 无 BOM 且含中文）：vendored
  `CMakeLists.txt` 已按上游配置 `/W4 /utf-8`；MinGW/g++ 无需额外选项。
- POSIX 专属场景（外部删除/替换已打开的文件、`/dev/full`、symlink）在 Windows
  不适用，上游测试已按 `_WIN32` 跳过；本目录只 vendor 库源码，不含 tests。

校验同步内容可用 `git hash-object` 对比上游 commit `0189fe3`：

```text
include/stacklog/stacklog.hpp   db708904dec486929e2d5df3d12bd485fae67991
include/stacklog/version.hpp    d5258796e2c594a67852f7bc635ad2ffa346bcde
src/stacklog.cpp                c2f7d6fc1206da7e9a8dbe9a6b115410cdca3a1a
src/platform.hpp                2e15c232f2cdfc0bc3474dde5e8911811535c72a
src/platform_posix.cpp          cf9426ae076480a50990c61abc478365e8f382d1
src/platform_win32.cpp          1de436880ee27205d1a11b4ad17fc8ac95bdf3e4
src/platform_dirs.hpp           77821843b5a612ef8d389d9be1906b49e9283e2a
src/platform_time.hpp           ea835305e152a30b5ac9ab5b3426bed6a522e6e9
src/platform_paths.hpp          69b26cb0d1e2391bae6c8b2440dc5ee9bd709abb
src/platform_stream.hpp         4222bf99e9df6e12ce8f80bc88e938640a084a4f
```
