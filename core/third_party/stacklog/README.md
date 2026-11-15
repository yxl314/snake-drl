# stacklog（内嵌副本）

本目录是 [stacklog](../../../../stacklog) 的副本，随 `snake_core` 一起分发。

- 同步自上游 commit `d96bcfa`（其源文件与新增 Windows 支持的 `ae17172` 一致，
  后续只多了文档改动）：含 UTF-8 路径转宽字符、CTest/qmake 的 Windows 支持。
- 版本见 `include/stacklog/version.hpp`
- 许可见 `LICENSE`
- 只构建静态库；安装与导出由上层项目负责

与上游相比只有两处差异：(a) 多一个只构建静态库的 `CMakeLists.txt`；
(b) 该 `CMakeLists.txt` 显式打开 `POSITION_INDEPENDENT_CODE`。原因：本库内部有
`thread_local` 状态，`snake_core` 会被链进 Python 扩展 `.so`，非 PIC 的 TLS 访问
会产生 local-exec 重定位，链接器直接报错。

升级方式：从上游仓库重新拷贝 `include/`、`src/`、`LICENSE`，保持本目录只多出
`CMakeLists.txt` 与这份说明。

## Windows 要点（来自上游 `docs/windows支持.md`）

- `config.dir` / `config.base_name` 在 Windows 上按 **UTF-8** 解释，库内部转 UTF-16
  后调用宽字符文件 API；`\\?\` 扩展长度路径必须使用反斜杠。
- 目标是 STATIC 库，不需要 `__declspec(dllexport/dllimport)`，直接链入即可。
- MSVC 编译必须加 `/utf-8`（源码是 UTF-8 无 BOM 且含中文）：vendored
  `CMakeLists.txt` 已按上游配置 `/W4 /utf-8`；MinGW/g++ 无需额外选项。
- POSIX 专属场景（外部删除/替换已打开的文件、`/dev/full`、symlink）在 Windows
  不适用，上游测试已按 `_WIN32` 跳过；本目录只 vendor 库源码，不含 tests。

校验同步内容可用 `git hash-object` 对比上游 commit `d96bcfa`：

```text
src/stacklog.cpp                aba78f4a947ef85c0ceee360669f6025430547ad
include/stacklog/stacklog.hpp   1f279d59c805d777533ec1495226ab38b62ff082
include/stacklog/version.hpp    d5258796e2c594a67852f7bc635ad2ffa346bcde
```
