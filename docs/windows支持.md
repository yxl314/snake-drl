# Windows 支持

本文记录 `snake-drl` 在 Windows 上的支持范围、实现要点与真机验证结果。结论基于
2026-11-15 的一台 Windows 11 x64 机器（原生 CPython 3.14 + MSYS2 UCRT64），以及同一
份代码在 Linux 上的回归。

## 1. 结论

| 门禁 | Linux | Windows | 说明 |
|---|---|---|---|
| `check.py core`（构建 + ctest） | ✅ | ✅ | MinGW-w64 g++ 16.2 + Ninja；ctest 1/1 通过 |
| `check.py core`（排版 standardize） | ✅ | ✅ | 28 个文件 0 改动 |
| `check.py core`（clang-tidy） | ✅ | ✅ | cxx_audit 自动检测 MinGW 并补 `--target`；项目侧补 assert noreturn 声明，0 告警 |
| `check.py core`（CPD） | ✅ | ✅ | PMD 7.29.0 + JRE 21 真跑，无重复 |
| `check.py core`（Valgrind） | ✅ | ⚠️ SKIPPED | Valgrind 没有 Windows 版本，cxx_audit 打印原因后跳过 |
| `check.py python` | ✅ | ✅ | pytest 105 passed / 4 skipped |
| `check.py lint` | ✅ | ✅ | ruff 全部通过 |
| 扩展模块 import | ✅ | ✅ | `.pyd` 静态链接 MinGW 运行时，不依赖 MSYS2 PATH |

一句话结论：**Windows 上与 Linux 一致——core（构建/ctest/排版/clang-tidy/CPD）、
python、lint 全部通过；Valgrind 因平台不适用 SKIPPED。`python check.py all` 退出 0。**

## 2. 已验证环境

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Windows 11 x64（10.0.26300） |
| Python | CPython 3.14.10（原生 Windows 构建，`python.exe` + `python -m venv`） |
| C++ 工具链 | MSYS2 UCRT64：g++ 16.2.0、CMake 4.5.0、Ninja 1.13.2 |
| clang-tidy / clang-format | 21.1.6 / 21.1.8（PyPI 轮子，放在 `PATH` 中 MSYS2 之前） |
| CPD 依赖 | PMD 7.29.0 + Temurin JRE 21.0.13 |
| torch | 2.15.0+cpu（`--index-url https://download.pytorch.org/whl/cpu`） |
| numpy / matplotlib / Pillow | 2.5.4 / 3.11.2 / 13.0.0 |
| pytest / ruff / PyQt6 | 9.2.0 / 0.17.5 / 6.12.1 |
| 验证日期 | 2026-11-15 |

## 3. 安装与用法

```powershell
# 1) 原生 Windows Python 3.14；MSYS2 UCRT64 的 g++/cmake/ninja 在 PATH 中
py -m venv .venv
.venv\Scripts\pip install -e ".[plot,play,dev]"

# 2) 门禁所需的工具版本（与 Linux 基线一致；MSYS2 自带的是 23.x，不匹配）
.venv\Scripts\pip install clang-tidy==21.1.6 clang-format==21.1.8

# 3) 跑门禁（cmd/PowerShell 均可，不需要 sh）
python check.py all
```

CPD 需要 `pmd` 与 `java` 在 `PATH` 中（cxx_audit 的约定：两任一缺失会把 CPD 标成
SKIPPED 并打印原因，不算失败）。`check.py` 的 venv 查找顺序是
`SNAKE_DRL_PYTHON` → `.venv\Scripts\python.exe` → 报错并提示如何创建 venv。

## 4. 平台实现要点（本仓库内）

1. **跨平台门禁入口**：新增 `check.py` 承担 `all|core|python|lint` 分派，`check.sh`
   退化为 POSIX 薄壳（`exec python3 check.py "$@"`）。Windows 没有 `sh`，因此 core
   阶段改为逐条运行 `core/check.sh` 内部的等价命令（cxx_ci 的 ci/standardize +
   cxx_audit 的 cpp_audit）；POSIX 仍然直接调 `core/check.sh all`，输出与退出码不变。
   退出码语义与旧脚本一致：0 通过 / 1 有阶段失败 / 2 用法错误 / 130 用户中断。
2. **扩展模块自包含**：MinGW 构建的 `_snake_core.pyd` 用 `-static` 链接
   `libgcc/libstdc++/libwinpthread`。Python 3.8 起扩展模块的依赖 DLL 搜索不再走
   `PATH`，动态依赖这些运行库会让 `import snake_drl` 直接报 "DLL load failed"
   （除非用户手工 `os.add_dll_directory` 或安装 MSYS2 运行时）。静态链接后
   `objdump -p` 只剩 `KERNEL32`、系统 UCRT `api-ms-win-crt-*` 与 `python314.dll`。
3. **vendored stacklog 同步上游 `0189fe3`**：平台层拆分（`platform_posix.cpp` /
   `platform_win32.cpp`，公开 API 不变），带 UTF-8→UTF-16 路径转换、Windows
   轮转与 CTest 支持；内嵌副本的 `CMakeLists.txt` 按平台选择适配器，并保留上游
   一致的 MSVC `/W4 /utf-8` 分支（MinGW/g++ 不需要额外选项）。
4. **clang-tidy opt-in 检查的测试豁免**：`core/tests/test_engine.cpp` 里有两处
   “故意构造非法枚举值”的输入，clang-tidy 21 的
   `clang-analyzer-optin.core.EnumCastOutOfRange` 会误报。加了带理由的
   `NOLINTNEXTLINE`（Linux 同版本也会报这两条，与 Windows 无关），不放宽
   `.clang-tidy` 的检查集。
5. **clang-tidy 的 MinGW 适配**：cxx_audit ≥ `87fb102` 会从编译数据库自动检测
   MinGW/GCC，给 clang-tidy 追加 `--target=x86_64-w64-windows-gnu`
   `--gcc-toolchain=<ucrt64>`，并生成一份剔除 GCC 私有参数的数据库。但 MinGW/UCRT
   的 `assert` 失败函数（`_assert` / `_wassert`）声明缺少 noreturn，clang-analyzer
   会把 assert 失败当成继续执行，误报空指针解引用/未定义返回。仓库侧新增
   `core/cmake/analysis_mingw_assert.h`（只在 `_WIN32 && __clang__` 下补声明属性），
   由 `core/audit.toml` 的 `[tidy].extra_args` 强制包含；普通编译（g++/MSVC）与
   POSIX 分析都不受影响。

## 5. 已知限制

- **clang 工具版本必须与门禁基线对齐**：基线是 clang-tidy/clang-format
  **21.1.8**（Linux 系统版本；Windows 用 PyPI 同系列 `clang-tidy==21.1.6` +
  `clang-format==21.1.8`）。安装命令：

  ```powershell
  .venv\Scripts\pip install clang-tidy==21.1.6 clang-format==21.1.8
  ```

  并且 Python 的 `Scripts` 目录要在 `PATH` 中排在 MSYS2 之前（否则会解析到
  MSYS2 UCRT64 自带的 **23.1.5**）。23.1.5 新增/收紧了若干检查，与基线差异明显
  （历史实测会产生 24 条版本差异告警），**不能用于本门禁**。版本 pin 就记在这里，
  不额外加 requirements 文件。
- **CPD/Valgrind**：CPD 在 Windows 上可用，但要求 `pmd` 与 `java` 都在 `PATH`
  （PMD 的 `pmd.bat` 只认 PATH 里的 java，`JAVA_HOME` 无效）；Valgrind 没有原生
  Windows 版本，cxx_audit 会 SKIPPED 并打印原因。`audit.toml` 里的 ASan 仍是项目级
  关闭状态（原环境限制），本次未改变该决定。
- **MSVC 未验证**：构建验证使用 MSYS2 UCRT64 的 MinGW-w64 g++；CMake 与内嵌
  stacklog 保留了 MSVC 分支（`/W4 /utf-8`），但没有真机跑过。
- **PyQt6 真实窗口交互未验证**：PyQt6 6.12.1 可以安装并 import，`QT_QPA_PLATFORM=offscreen`
  下能成功创建 `QApplication`；但验证机是 SSH 无桌面环境，`snake-drl play` 的真实
  窗口显示、方向键/暂停/重开等交互没有在 Windows 桌面上跑过。
- **中文路径**：验证目录使用 ASCII 路径；`stacklog` 上游已支持含中文路径的 UTF-8
  约定，但本仓库没有在中文工作目录下重新验证 C++ 构建。

### 历史记录（已修复，保留追溯）

- `d95dc45`：Windows tidy 曾报 2 条 `bugprone-exception-escape` 伪告警——PyPI
  clang-tidy 默认 target 是 MSVC，分析 MinGW 构建时按 MSVC STL 推断异常路径。
- cxx_audit `f830277` 自动为 MinGW 编译数据库补 `--target` 后，上述 2 条消失，但
  又暴露出 4 条 `clang-analyzer` 伪告警（MinGW/UCRT 的 assert 失败函数缺 noreturn
  标注）。
- `db00cd1`：加入 `core/cmake/analysis_mingw_assert.h` 的 noreturn 声明补丁后，
  tidy 变成 `0 个错误，0 个警告`；最终 `python check.py all` 退出码 0。以上两轮
  历史失败均不再是现状，仅作追溯保留。

## 6. 验证记录

Windows 侧从 `git archive HEAD` 部署到干净目录（ASCII 路径、空 venv）实测：

```powershell
# 铺开：python -m venv .venv
#       pip install torch（CPU 轮子）+ matplotlib/pillow/ruff/pytest/PyQt6
#       pip install -e .                 # MinGW 构建 core + 扩展
# 实测耗时：113 s（含首次下载/编译）
python check.py all                     # 实测 80 s，退出码 0
```

关键输出（节选）：

```text
[ci]   configure  ok       6.26s
[ci]   build      ok       3.44s
[ci]   test       ok       10.14s       # ctest 1/1 passed, snake_core_tests
[standardize] processed: 28  changed: 0  unchanged: 28  failed: 0
[OK] tidy: 14 个文件，0 个错误，0 个警告
[OK] cpd: 退出码=0，重复=否
[SKIPPED] valgrind: Windows 上没有原生 Valgrind ...
105 passed, 4 skipped in 19.46s
All checks passed!                              # ruff
[check] summary:
  core: passed
  python: passed
  lint: passed
[check] passed
```

tidy 报告头会记录 MinGW 兼容参数，便于核对：

```text
# 文件: 14  配置: .clang-tidy  编译库外跳过: 0
# MinGW 兼容参数: --extra-arg=--target=x86_64-w64-windows-gnu --extra-arg=--gcc-toolchain=C:/msys64/ucrt64
```

扩展模块自包含验证：`objdump -p _snake_core.cp314-win_amd64.pyd` 的依赖只有
`KERNEL32.dll`、系统 UCRT `api-ms-win-crt-*.dll` 和 `python314.dll`；把 `PATH` 清到
只剩 `C:\Windows\System32;C:\Windows` 后 `import snake_drl` 仍然成功。

PyQt6 6.12.1 / Qt 6.12.0 在 `QT_QPA_PLATFORM=offscreen` 下可以创建
`QApplication`；SSH 无桌面，真实窗口交互未验证。

Linux 回归：`./check.sh all` 在改动后仍然全绿（core / python / lint 三项 passed，
clang-tidy/format 21.1.8 与 17.0.6 都验证过 0 告警）。
