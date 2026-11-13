# snake-drl-core

CPU-only, high-throughput batched snake engine for reinforcement learning.

The core owns all game logic: state, collision, RNG, rewards, termination and
observation generation. It has no CUDA dependency, no device memory and no
host-device synchronization. Training frameworks are external consumers; the
core is deliberately framework-agnostic.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Options:

- `-DSNAKE_BUILD_TESTS=ON|OFF`
- `-DSNAKE_BUILD_BENCHMARKS=ON|OFF`
- `-DSNAKE_ENABLE_IPO=ON|OFF`
- `-DSNAKE_WARNINGS_AS_ERRORS=ON|OFF`
- `-DSNAKE_STRICT_WARNINGS=ON|OFF` (extended `-Wconversion`/`-Wsign-conversion` set)
- `-DSNAKE_ENABLE_INSTALL=ON|OFF` (generate `cmake --install` / `find_package` rules)

CMake presets:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Debug, UBSan, ASan and TSan presets are in `CMakePresets.json`. The extended
warning set (`-Wconversion`, `-Wsign-conversion`, ...) is on by default;
`release-werror` additionally treats warnings as errors.

## 门禁

```bash
./check.sh          # 等价于 ./check.sh all
./check.sh ci       # 构建 + ctest
./check.sh style    # clang-format / clang-tidy 排版检查
./check.sh audit    # clang-tidy + CPD + Valgrind
```

三个工具来自同级目录（`../../cxx_ci`、`../../cxx_audit`），不安装到系统。
`JOBS=` 可覆盖并行度，缺省为本机核数。

检查集在 `.clang-tidy` 里，`modernize-*` 全面启用；本项目是 C++17，链接期
相关检查按该标准配置。

## 日志

`third_party/stacklog` 是上游副本，随仓库分发。日志是**内部实现细节**：
`snake_core` 以 PRIVATE 方式静态链接它，公开头文件不含 stacklog，库本身也
从不调用 `stacklog::init()`——由消费者（测试、训练/评估程序、绑定）决定
日志目录与级别。

只在低频边界打日志：构造、重播种、不定点检查失败。`step`/`reset*`、位棋盘
与食物采样等热路径一行都不打。

## Layout

```text
include/snake/         public config, cell codec, views and VecSnake facade
src/                   engine, environment rules, invariants, thread pool
src/detail/            internal bitboard, free-cell, grid, RNG, pool headers
tests/                 module tests and engine stress tests
benchmarks/            C++ throughput benchmark
docs/                  final design snapshot
```

## Usage

```cpp
#include <cstdint>
#include <vector>

#include "snake/vec_snake.hpp"

snake::VecConfig cfg;
cfg.num_envs = 1024;
cfg.width = 16;
cfg.height = 16;
cfg.obs_mode = snake::ObsMode::Bitboard;
cfg.autoreset_mode = snake::AutoresetMode::SameStep;
cfg.num_threads = 0;  // auto

snake::VecSnake env(cfg);
env.reset_all();

std::vector<int8_t> actions(cfg.num_envs, 0);  // 0=straight, 1=right, 2=left
env.step(actions.data());

// Views are non-owning, read-only and self-describing.
snake::BoardView rows = env.bitboards();        // (N, H) uint64
snake::CellView head = env.head_cells();        // (N,) uint16
snake::CellView food = env.food_cells();        // (N,) uint16
snake::FloatView rewards = env.rewards();       // (N,) float
snake::FloatView returns = env.returns();       // (N,) float
snake::FloatView final_returns = env.final_returns();
snake::ByteView directions = env.directions();  // 0=up,1=right,2=down,3=left
snake::ByteView dones = env.dones();            // (N,) uint8

const std::uint64_t row_0_of_env_0 = rows.at(0, 0);
const std::uint16_t head_of_env_1 = head[1];
```

For dense observations use `cfg.obs_mode = snake::ObsMode::Grid` and read
`env.observations()`, a `snake::GridView` with `(N,3,H,W)` uint8 layout and
`plane(env, channel)` / `at(env, channel, y, x)` helpers. Unsupported modes
return an empty view (`valid() == false`).

All views refer to internal buffers and stay valid for the lifetime of the
`VecSnake` instance; `step`/`reset` overwrite their contents, so copy data if
you need the previous values. `final_*` views are only
meaningful when the corresponding environment reported `dones == 1` on the
latest step.

`returns()` is the cumulative return of the episode the latest step belonged
to. With `SameStep` the live `scores()`/`lengths()` already belong to the new
episode after a terminal step; use `final_scores()`/`final_lengths()`/
`final_returns()` for the finished one.

Reset and seeding:

```cpp
env.reset_all();
std::vector<std::uint8_t> mask(cfg.num_envs, 0);
mask[0] = 1;
env.reset_mask(mask.data());

// Rebuilds every environment from scratch. Environment i is seeded with
// `seed + i`, exactly like construction, so the result is reproducible and
// idempotent.
env.reseed_all(123);

// Rebuilds only environment 0, using exactly this seed.
env.reseed_env(0, 456);
```

## Observations

| Mode | Data |
|---|---|
| `Bitboard` (default) | `rows (N,H) uint64` + head/food packed cells |
| `Grid` | incremental `(N,3,H,W) uint8`: body, head, food |
| `NoObs` | no dense observation; bitboards and metadata only |

Bitboard rows store cell `x` at bit `x ^ 7`, matching big-endian byte
unpacking on little-endian hosts (the multi-byte words are read in native
memory order, so a big-endian build would need a different mapping).

Use the public codec from `snake/cell.hpp` instead of re-deriving the layout:

```cpp
const std::uint16_t cell = env.food_cells()[0];
const int x = snake::cell_x(cell);
const int y = snake::cell_y(cell);
const std::uint16_t packed = snake::pack_cell(x, y);
```

`final_*` availability (empty view in C++ when unavailable):

| View | Available when |
|---|---|
| `final_observations` | Grid + SameStep |
| `final_bitboards`, `final_head_cells`, `final_food_cells` | SameStep |
| `final_scores`, `final_lengths`, `final_returns` | always |

## Autoreset

| Mode | Behavior |
|---|---|
| `SameStep` (default) | terminal step also resets; final state is captured in `final_*` |
| `NextStep` | reset happens on the next step; that action is ignored |
| `Disabled` | caller resets with a mask; until then `step` is a no-op that keeps the terminal outputs intact |

## Install

```bash
cmake --install build --prefix /opt/snake
```

Consumers can then use:

```cmake
find_package(snake_core CONFIG REQUIRED)
target_link_libraries(app PRIVATE snake::snake_core)
```

## Tests

```bash
ctest --test-dir build --output-on-failure
# or
./build/snake_core_tests
```

## Benchmark

```bash
./build/snake_benchmark
# quick run
./build/snake_benchmark --quick
```

The benchmark covers Bitboard/Grid/NoObs, serial vs. automatic threads and
multiple environment counts/board sizes. Run it on the target machine; the
design does not hard-code machine-specific numbers.

## Design

See [`docs/设计方案.md`](docs/设计方案.md) for the complete final design:
data structures, O(1) step path, CPU thread pool, observations, API, tests and
acceptance criteria.
