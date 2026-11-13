# stacklog（内嵌副本）

本目录是 [stacklog](../../../../stacklog) 的副本，随 `snake_core` 一起分发。

- 版本见 `include/stacklog/version.hpp`
- 许可见 `LICENSE`
- 只构建静态库；安装与导出由上层项目负责

升级方式：从上游仓库重新拷贝 `include/`、`src/`、`LICENSE`，保持本目录只多出
`CMakeLists.txt` 与这份说明。
