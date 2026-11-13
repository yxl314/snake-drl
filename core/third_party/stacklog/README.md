# stacklog（内嵌副本）

本目录是 [stacklog](../../../../stacklog) 的副本，随 `snake_core` 一起分发。

- 版本见 `include/stacklog/version.hpp`
- 许可见 `LICENSE`
- 只构建静态库；安装与导出由上层项目负责

与上游相比只有两处差异：(a) 多一个只构建静态库的 `CMakeLists.txt`；
(b) 该 `CMakeLists.txt` 显式打开 `POSITION_INDEPENDENT_CODE`。原因：本库内部有
`thread_local` 状态，`snake_core` 会被链进 Python 扩展 `.so`，非 PIC 的 TLS 访问
会产生 local-exec 重定位，链接器直接报错。

升级方式：从上游仓库重新拷贝 `include/`、`src/`、`LICENSE`，保持本目录只多出
`CMakeLists.txt` 与这份说明。
