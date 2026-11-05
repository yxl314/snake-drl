# snake-drl

A batched snake engine written in C++, plus a modular deep-RL layer on top of it:

- **Engine** (`core/`): CPU-only, bitboard collision, persistent thread pool,
  zero-copy observation views, ~10–70 M env-steps/s depending on batch size.
- **Algorithms**: PPO and the DQN family (DQN, DDQN, D3QN), uniform and
  prioritized experience replay.
- **Networks**: MLP, CNN, GRU, Transformer, ConvGRU, ConvTransformer — all
  interchangeable with either algorithm family.
- **Devices**: CPU and CUDA, with the data pipeline written to avoid allocation,
  synchronization and thread-oversubscription hazards.
- **Human play**: PyQt6 front-end (`snake-play`).

```text
core/                 C++ engine (framework-agnostic, no CUDA)
bindings/python/      pybind11 bindings -> _snake_core
python/snake_drl/     engine re-export, DRL layer and play mode
  spaces.py           ObsSpec / ActionSpec / StepBatch
  registry.py         name -> factory registry
  config.py           typed configs + YAML loading
  envs/               VectorEnv, snake adapter, toy MDPs
  nn/                 encoders, heads, policies, FastLinear
  buffers/            rollout buffer, uniform replay, prioritized replay
  algorithms/         PPO, DQN family, GAE, schedules
  runner.py           the only component that talks to an environment
  trainer/            trainer loop, evaluator, logger, checkpoints, factory
  cli.py              `snake-drl train|eval|play|bench`
configs/              example YAML configs
docs/                 design document (Chinese) with benchmark data
tests/                pytest suite (CPU, seconds)
```

## Build

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[play,dev]"
```

`pip install -e .` compiles the C++ engine through scikit-build-core. Re-run it
after changing `core/` or `bindings/`.

## Train

```bash
# Dueling double DQN + prioritized replay + CNN on a 10x10 board
snake-drl train -c configs/d3qn_cnn.yaml

# small and fast on CPU
snake-drl train -c configs/dqn_mlp.yaml --set train.total_steps=20000

# override anything from the command line
snake-drl train -c configs/ppo_cnn.yaml --set algorithm.lr=1e-4 --set run.device=cpu
```

Outputs land in `runs/<name>-<timestamp>/`: `config.yaml`, `metrics.jsonl`,
`checkpoint_*.pt` and `latest.pt`.

```bash
# evaluate a checkpoint (greedy policy, fresh environments)
snake-drl eval -m runs/d3qn_cnn-*/latest.pt --episodes 50

# measure engine + policy throughput for your machine
snake-drl bench -c configs/d3qn_cnn.yaml

# play yourself
snake-play
```

## Design in one paragraph

The engine emits a zero-copy `(N,3,H,W)` uint8 view per step. A single
`VectorRunner` keeps the per-environment frame window each encoder asks for,
uploads frames to the device (pinned + `non_blocking` on CUDA) and reports the
*true* successor observation of terminal/truncated steps. Buffers store uint8
observations and copy whole minibatches at a time; algorithms only know narrow
protocols (`ActorCriticPolicy`, `QPolicy`, `RolloutBuffer`, `ReplayBuffer`), so
any encoder works with any algorithm. A composition root assembles everything
from a typed YAML config; see [the design document](docs/设计方案-模块化DRL.md).

## Swap a component

```python
from snake_drl.nn import ENCODERS, build_policy
from snake_drl.buffers import BUFFERS

print(ENCODERS.names())   # mlp, cnn, gru, transformer, convgru, convtransformer
print(BUFFERS.names())    # uniform, prioritized
```

Add your own by registering a factory and naming it in the YAML:

```python
from snake_drl.nn.encoders import Encoder

@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    def __init__(self, spec, cfg):
        super().__init__(cfg.frames, cfg.feature_dim)
        ...
    def forward(self, obs):
        ...
```

## Configuration

Every field is validated (`dacite` strict mode rejects unknown keys). The
algorithm block is dispatched on `type`: `ppo`, `dqn`, `ddqn` or `d3qn` (the
last three share `DQNConfig`; explicit `double_q` / `dueling` win over the
preset). Example:

```yaml
env:   {type: snake, num_envs: 256, width: 10, height: 10, max_episode_steps: 500}
network:
  encoder: {type: cnn, frames: 1, feature_dim: 256, cnn_channels: [32, 64, 64]}
buffer: {type: prioritized, capacity: 100000, alpha: 0.6, beta0: 0.4}
algorithm: {type: d3qn, lr: 1e-4, batch_size: 256, train_freq: 4, gamma: 0.99}
train: {total_steps: 500000, log_every: 5000, eval_every: 50000}
```

## Performance notes

Measured on an 8-vCPU aarch64 box (details in the design document):

- `torch.set_num_threads` defaults to half the cores. Using all 8 next to the
  engine's pool made a 196 KiB copy ~60x slower and end-to-end steps ~2x slower.
  Override with `run.torch_threads`.
- `nn.Linear` (i.e. `addmm`) measured 15–157x slower than `matmul` on this
  build, so linear layers use `FastLinear`; set `SNAKE_DRL_LINEAR=addmm` to opt
  out.
- Replay stores uint8 observations (293 MiB vs 1.2 GiB for float32 at 200k
  transitions) and samples with `index_select(..., out=...)`.
- DQN updates are counted in *transitions* (`train_freq`), not vectorized steps,
  so the update rate does not silently shrink when `num_envs` grows.

## Tests

```bash
pytest -q
```

Covers config validation, all six encoders, policy/head math, buffer wraparound,
prioritized-sampling correctness, GAE against a naive reference, DQN/DDQN/D3QN
targets, terminal/truncation handling, the runner window contract, checkpoint
round-trips and short learning runs on toy MDPs.

## C++ engine quality gates

```bash
cd core
cmake --preset release && cmake --build --preset release && ctest --preset release
cmake --preset release-werror && cmake --build --preset release-werror
cmake --preset ubsan && cmake --build --preset ubsan && ctest --preset ubsan
```
