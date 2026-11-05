# snake-drl

CPU-only, high-throughput batched snake engine for reinforcement learning, plus
a PyQt6 human-play mode and pybind11 bindings.

## Layout

```text
core/                 C++ engine (no Python/CUDA dependency)
bindings/python/      pybind11 bindings -> _snake_core
python/snake_drl/     Python package (bindings re-export + play mode)
```

`core/` builds and installs on its own; the top-level `CMakeLists.txt` is only a
monorepo entry point that adds the bindings when `SNAKE_BUILD_PYTHON=ON`.

## C++ build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or from `core/` alone:

```bash
cd core
cmake --preset release
cmake --build --preset release
ctest --preset release
```

`core/` also supports `cmake --install` and `find_package(snake_core)`; set
`-DSNAKE_ENABLE_INSTALL=OFF` when you only need the library in-tree.

## Python install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[play]"
```

`[play]` pulls in PyQt6. Without it, `import snake_drl` and the bindings still
work for training.

After changing C++ code, rerun `pip install -e .` (or `cmake --build build`)
because the extension is compiled.

## Human play

```bash
snake-play
# or
python -m snake_drl.play
```

Options: `--width`, `--height`, `--seed`, `--speed 1-7`.

| Key | Action |
|---|---|
| Arrow keys / WASD | Turn |
| Space | Pause / start |
| R | Restart |
| `+` / `-` | Speed up / down |
| Esc | Quit |

Controls map to absolute directions, so the engine ignores a 180° reversal.

## Python usage

```python
import numpy as np

from snake_drl import ObsMode, VecConfig, VecSnake

cfg = VecConfig()
cfg.num_envs = 1024
cfg.obs_mode = ObsMode.Grid
env = VecSnake(cfg)
env.reset_all()

obs = env.observations()        # (N,3,H,W) uint8, read-only zero-copy view
env.step(np.zeros(cfg.num_envs, dtype=np.int8))
rewards = env.rewards()         # (N,) float32
directions = env.directions()   # (N,) uint8, 0=up,1=right,2=down,3=left
```

Views alias engine memory, are read-only, and are invalidated by the next
`step`/`reset`. Copy before storing them.

## Quality gates

```bash
cd core
cmake --preset release-werror     # strict warnings (-Wconversion/-Wsign-conversion) + -Werror
cmake --build --preset release-werror
ctest --preset release-werror
cmake --preset ubsan && cmake --build --preset ubsan && ctest --preset ubsan
```
