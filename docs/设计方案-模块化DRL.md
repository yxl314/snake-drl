# snake-drl 模块化 DRL 层设计（v2 · 定稿）

本文件是 v1 方案经过 `桌面/tmp/drl-lab` 实验修订后的版本。v1 假设、实验数据、
被推翻的结论和被保留的结论都记录在第 0 节；后续章节描述最终实现。

---

## 0. 实验反馈：v1 → v2

### 0.1 被实验推翻或修正的假设

| # | v1 的假设 | 实验 | 结论 → v2 的修订 |
|---|---|---|---|
| 1 | 位棋盘观测更快，训练层可默认用它 | E1：1024 环境 `np.unpackbits` 285 µs，再加拷贝 426 µs；直接拷贝 grid 只要 25.8 µs | 训练层**只暴露 grid**；位棋盘留在引擎里。`EnvConfig.obs_mode` 只接受 `grid` |
| 2 | torch 线程数取满核数 | E2：8 线程时 196 KiB `copy_` 从 4.2 µs 变成 427 µs，端到端 step 慢约 1.5–2× | 默认 `torch_threads = min(8, cores//2)`；实测 4 线程 + 引擎 1–2 线程最优 |
| 3 | 回放直接存 float32 | E4：200k 转移、3×16×16、obs+next_obs：float32 1.17 GiB vs uint8 293 MiB；采样 256 条 uint8 仅 6.6 µs | 回放存 **uint8**，只在网络入口转 float |
| 4 | PER 用逐样本循环的 sum tree | E4：向量化后 256 样本 67 µs；逐条更新 256 个优先级 5.3 ms | 分层 numpy 数组 + `np.add.at` 的向量化树 |
| 5 | 六种网络都是"可选组件"，成本相近 | E5：同输入下 CNN 8 ms，ConvTransformer 417 ms（fwd） | 保留六种，但 CPU 默认 cnn/mlp；成本写入文档并用于选默认值 |
| 6 | 逐环境循环的隐状态 RNN | 接口实验 + 复杂度分析 | 采用**窗口式**时序编码（见 §7），算法与回放无需感知时序 |
| 7 | `nn.Linear` 是基线实现，无需关心 | 实现期实测：`addmm`/`nn.Linear` 比 `matmul` 慢 **15–157×**（所有形状） | 引入 `FastLinear`（保持参数布局，仅换 forward）；可用 `SNAKE_DRL_LINEAR=addmm` 关闭 |
| 8 | DQN 每个"向量步"做一次梯度更新 | 与 SB3 对照：N=16 时更新次数只有单环境实现的 1/16，玩具 MDP 上完全不收敛 | 更新频率按**转移数**计（`train_freq`），与环境数解耦 |

### 0.2 实验中发现并修复的缺陷

1. **SumTree 根节点漂移为负**：同一样本在一个 batch 里出现多次时，
   `delta = new - old` 被重复累加。修复：`np.unique` 去重后再传播。
   如果不修，训练几千步后 `total` 变负，PER 直接崩溃。
2. **Runner 返回的观测被原地覆盖**：窗口原地左移会让调用方拿到的 `obs`
   变成下一步的窗口。修复：双窗口交替写，`obs` 指向旧窗口，永不原地修改。
3. **逐向量步更新**（见上表 #8）。
4. **日志器在 eval 之前关闭**：回调顺序错误。修复：独立的
   `CloseLoggerCallback` 追加在最后。
5. **截断 bootstrap**：早期实现把 `done`（terminated∨truncated）当作终止掩码，
   截断时不再 bootstrap，Q 有系统性低估。修复：缓冲区存 `terminated` 与
   `truncated` 两列，`next_obs` 在终止/截断步统一取引擎的 `final_*`。
6. **PPO 每次 rollout 都 reset 环境**：episode 被硬切成 `n_steps`，智能体无法跨
   rollout 维持一局。修复：runner 的状态跨 iterate 保留（与 DQN 一致）。
   修复后 300k 步的评估回报由 ±0.25 的平台变为 0.31→0.44→0.63→0.65→0.80 单调上升。

### 0.3 被实验确认的 v1 决策

- 窗口式时序编码可行，六种编码器 × 两种算法族（PPO / Q）全部跑通。
- uint8 + 预分配 + `copy_` 的采集路径没有分配热点。
- 分层 / 协议 / 注册表 + 显式装配的依赖注入方式，使测试可以注入假环境与假策略。

### 0.4 实验环境

8 vCPU aarch64（无 CUDA），15 GiB 内存，Python 3.12，PyTorch 2.15.0+cpu。
所有数字都在同一台机器上测得；表中绝对值受当时负载影响，**比例关系**是决策依据。

---

## 1. 目标与非目标

目标：

- 算法：PPO、DQN、DDQN、D3QN；离线回放支持均匀与优先级采样。
- 网络：MLP、CNN、GRU、Transformer、ConvGRU、ConvTransformer，且与算法族正交。
- 设备：CPU / CUDA，数据路径按"避免分配、避免同步、避免线程超订"设计。
- 工程：低耦合、可替换、依赖注入、可测试、可扩展。

非目标：分布式训练；连续动作；多智能体；重写 C++ 引擎；引入 Lightning /
Hydra / Ray 这类会接管生命周期的框架（配置用 OmegaConf，校验用 dacite，
都只做"解析 + 校验"，不做装配）。

---

## 2. 分层与依赖规则

```text
core (C++, 无 torch)  ←  _snake_core (pybind11)  ←  envs  ←  runner
                                                          ↖
                                            buffers  ←  algorithms  ←  trainer  ←  cli
                                                ↑             ↑
                                           nn (policy) ───────┘
```

| 层 | 可以依赖 | 不允许依赖 |
|---|---|---|
| `envs` | numpy、`_snake_core` | torch、buffers、算法 |
| `spaces` | 标准库 | 任何具体实现 |
| `nn` | torch、spaces、config | envs、buffers、算法 |
| `buffers` | torch、spaces、config | nn、算法、envs |
| `runner` | envs、spaces、torch | buffers、算法 |
| `algorithms` | torch、**协议**（Policy/Buffer/Runner） | 具体编码器、具体回放实现、CLI |
| `trainer` | 全部协议 + 装配 | CLI 之外的副作用 |
| `cli` | 装配根 | — |

`algorithms/` 里没有一处 `isinstance(encoder, CNNEncoder)` 之类的判断：
算法只看到 `ActorCriticPolicy` / `QPolicy` / `ReplayBuffer` / `RolloutBuffer`
的协议。`runner.py` 放在包顶层而不是 `trainer/` 下，就是为了避免
`algorithms → trainer → algorithms` 的循环导入。

---

## 3. 一次训练的数据流

```text
engine.step(int8 actions)                     C++，释放 GIL
  → observations()  (N,3,H,W) uint8 视图     零拷贝
  → VectorRunner
       · 维护 (N,T,3,H,W) 窗口（T 由编码器声明）
       · pinned staging + non_blocking 上传（CUDA 时）
       · next_obs = 终止/截断步用 final_obs，其余用新帧
  → StepBatch(obs, next_obs, reward, terminated, truncated)
  → PPO: RolloutBuffer[t] ← copy_ ；DQN: ReplayBuffer.add ← copy_
  → 算法更新（minibatch 索引切片 / index_select(out=)）
  → callbacks：日志 / 评估 / checkpoint
```

关键语义：**autoreset 的终止步**里，环境窗口已经属于新 episode，但学习者拿到的
`next_obs` 是终止帧（`final_obs`）。终止步不 bootstrap；截断步用终止帧 bootstrap。
这一条同时决定 GAE 与 DQN target 的正确性。

---

## 4. 张量契约（全库唯一约定）

| 名称 | 形状 | dtype | 说明 |
|---|---|---|---|
| `obs` | `(B, T, *frame)` | 环境 dtype（uint8 / float32） | `T` = 编码器感受帧数 |
| `next_obs` | 同上 | 同上 | 终止/截断步为终局帧 |
| `action` | `(B,)` | int64 | 离散 |
| `reward` | `(B,)` | float32 | |
| `terminated` | `(B,)` | bool | 不 bootstrap |
| `truncated` | `(B,)` | bool | **要** bootstrap |
| `weights` | `(B,)` | float32 | PER 重要性权重，均匀回放恒为 1 |

`ObsSpec.shape` 是单帧形状（`(C,H,W)` 或 `(D,)`），`ObsSpec.scale` 在网络入口
一次性乘上（snake 的 0/1 平面用 1.0，0..255 的观测可设 1/255）。
`frames` 属于编码器而不是算法：算法永远只看到"每环境每步一个观测"。

---

## 5. 组件、注册表与依赖注入

四件套：**协议 + 注册表 + 配置类 + 工厂**。

```python
# 每个层把注册表放在独立的 registry.py，避免"实现模块 -> 包 __init__ -> 实现模块"的环
from snake_drl.nn.encoders.registry import ENCODERS   # mlp/cnn/gru/transformer/convgru/convtransformer
from snake_drl.buffers.registry import BUFFERS        # uniform/prioritized
from snake_drl.envs.registry import ENVS              # snake/bandit/corridor
```

编码器实现拆成 `nn/encoders/{mlp,cnn,gru,transformer,convgru,convtransformer}.py`，
每个文件只依赖 `Encoder` 基类、注册表和 `FastLinear`；新增一种网络不需要改任何既有文件。

装配根 `trainer/factory.py` 是唯一知道"谁和谁怎么接"的地方：

```python
env     = make_env(cfg.env)
policy  = build_policy(cfg.network, env.obs_spec, env.action_spec,
                       family="actor_critic" | "q", device=device, dueling=...)
runner  = VectorRunner(env, policy.frames, device, pin_memory=...)
buffer  = RolloutBuffer(...) | build_replay(cfg.buffer, ...)
algo    = PPO(cfg.algorithm, policy, runner, buffer, optimizer, device)
        | DQN(cfg.algorithm, policy, target, runner, replay, optimizer, device,
              epsilon=LinearSchedule(...), beta=LinearSchedule(...))
```

算法构造参数全部显式传入，测试可以注入假 runner、假回放、常量 epsilon/beta。
`Trainer` 只依赖 `Algorithm.iterate()`；日志、评估、checkpoint 都是回调。

---

## 6. 环境层

- `VectorEnv` 协议：`reset() -> (N,*frame)`、`step(actions) -> StepResult`，
  纯 numpy，不知道 torch 的存在。
- `SnakeVecEnv`：映射 `EnvConfig`，固定 `AutoresetMode.SameStep` + `ObsMode.Grid`，
  动作缓冲预分配（`self._actions[:] = actions`），`dones.any()` 时才取终局观测。
- `BanditEnv` / `CorridorEnv`：确定性玩具 MDP，用于单测和算法正确性验证；
  两者都实现与引擎一致的 autoreset 语义。
- 动作模式：默认相对动作（直行/右转/左转，3 个），可选绝对方向（4 个）。
- **统一的工厂签名**：所有环境都是 `factory(cfg: EnvConfig, *, seed: int | None)`，
  由 `ENVS.create(cfg.type, cfg=cfg, seed=seed)` 调用；不存在 snake 与玩具环境的特殊分支。
  玩具环境的额外参数走 `env.kwargs`（`length`/`n_ctx`），`max_episode_steps` 与其他环境一样
  取自顶层字段——早期版本里玩具环境忽略这个字段，导致"永不结束的评估"。
- `SnakeVecEnv.close()` 可重复调用，关闭后再用会抛出明确的 `RuntimeError`，而不是
  `AttributeError: 'NoneType'`；`VectorRunner` 有同样的守卫。
- 编码器基类的 `forward` 是 `@abstractmethod`，忘记实现会在实例化时立刻报错。

---

## 7. 网络层

### 7.1 六种编码器

统一接口：`(B, T, *frame) -> (B, feature_dim)`。时间维在编码器内部处理，
所以 PPO 与 DQN 共用同一个 policy/buffer 契约。

| 类型 | 结构 | 输入 | 说明 |
|---|---|---|---|
| `mlp` | flatten(T·D) → [Linear+LN+SiLU]×k | image/vector | 帧拼接为特征维 |
| `cnn` | T 折进通道 → 3 层 stride-2 conv → adaptive pool → Linear | image | 帧堆叠在通道轴 |
| `gru` | 逐帧 Linear → GRU(T) → h_last | image/vector | 窗口内时序 |
| `transformer` | 逐帧 Linear + 可学习位置编码 → TransformerEncoder → 末 token | image/vector | |
| `convgru` | conv stem → 二维隐状态 ConvGRU 扫描 T 帧 → mean pool | image | 真正的卷积门控（r/z/n 分 x、h 两路） |
| `convtransformer` | conv patch embedding → (帧×patch) token + 帧/位置编码 → Transformer | image | 时空注意力 |

测量（B=256、T=4、3×16×16、4 线程；v1 原型实现，机器负载中等，绝对值仅供比例参考。
实现期的 `FastLinear` 主要改善线性层，编码器的相对排序不变）：

| 编码器 | fwd (ms) | 单步训练 (ms) | 参数量 |
|---|---:|---:|---:|
| mlp | 8.1 | 15.0 | 886k |
| cnn | 8.1 | 25.8 | 92k |
| gru | 13.7 | 20.3 | 214k |
| transformer | 31.9 | 61.6 | 381k |
| convgru | 91.0 | 316.3 | 250k |
| convtransformer | 416.8 | 831.7 | 323k |

结论：ConvTransformer 在 CPU 上比 CNN 慢约 50×，因此示例配置用 cnn/mlp；
时序/注意力编码器适合 GPU 或小棋盘，文档中明确标注。

### 7.2 头部与策略

- `ActorCriticHead`：`logits = Linear(F,A)`、`value = Linear(F,1)`（共享编码器）。
- `QHead`：`Linear(F,A)`。
- `DuelingQHead`：`V(F,1) + A(F,A) - A.mean(-1)`，均值扣除保证 V 可辨识。
- `ActorCriticPolicy`：`forward -> (logits, value)`；`act/deterministic`、`value`、
  `evaluate_actions`、`greedy`。
- `QPolicy`：`forward -> Q`；`act(epsilon)`、`greedy`。

正交初始化：编码器与隐藏层 gain=√2，actor/Q 输出 gain=0.01，value 输出 gain=1.0。

### 7.3 `FastLinear`

`nn.Linear` 走 `addmm`。在本机 torch 2.15 / aarch64 上，`addmm` 比 `matmul`
慢 15–157 倍（4 线程）：

| (M,K)×(K,N) | nn.Linear | matmul | 比值 |
|---|---:|---:|---:|
| (16,64)×(64,64) | 7.03 ms | 0.045 ms | 158× |
| (256,64)×(64,64) | 3.95 ms | 0.051 ms | 78× |
| (256,256)×(256,128) | 8.80 ms | 0.122 ms | 72× |
| (1024,128)×(128,64) | 2.83 ms | 0.132 ms | 21× |
| (256,768)×(768,256) | 6.68 ms | 0.444 ms | 15× |

`FastLinear` 继承 `nn.Linear`（参数布局、state_dict、初始化都不变），只把
forward 改成 `x @ W.T + b`。最大数值差异 1.2e-6。`SNAKE_DRL_LINEAR=addmm`
可回退。Conv2d 不受影响（走 conv 路径）。

它同时用于编码器**和输出头**：头层的 `(256,256)×(256,3)` 是同一个 `addmm` 热点，
实测 PPO 的 `act` 降低约 15%、DQN 单次梯度步降低约 18%。

---

## 8. 算法层

### 8.1 PPO

- 采集：`n_steps × num_envs`，全程 `no_grad` + `policy.eval()`。
- GAE：`δ = r + γ·V(s')·(1-terminated) - V(s)`；`s'` 对截断步取终局帧的 V，
  由 `v_next[truncated] = policy.value(next_obs[truncated])` 单独计算
  （只对截断子集做一次前向，不用为所有步重算）。
- GAE 的两类掩码不能混用：`terminated` 只用来抑制 `δ` 里的自举；递推的截断用
  `terminated | truncated`。截断步本身仍自举，但它之后的 transition 属于新回合
  （autoreset），若让它流入前一步的 advantage，就会把下一回合的回报借过来。
- advantage 标准化用 `std(unbiased=False)`：单样本 rollout 的标准差是 0 而不是 NaN。
- 更新：`epochs × minibatches`，advantage 标准化、ratio clip、可选 value clip、
  熵奖励、梯度裁剪、可选线性学习率、可选 `target_kl` 提前停止。
- 指标累积在设备张量里，一次 iterate 只同步一次。

### 8.2 DQN / DDQN / D3QN

同一实现的三个预设：

| 预设 | double_q | dueling |
|---|---|---|
| `dqn` | 否 | 否 |
| `ddqn` | 是 | 否 |
| `d3qn` | 是 | 是 |

```text
target = r + γ · (1 - terminated) · bootstrap
bootstrap = max_a Q_target(s')                      # DQN
          = Q_target(s', argmax_a Q_online(s'))     # DDQN
Q = V + A - mean(A)                                  # dueling head
```

- 目标网络：`tau == 1` 时每 `target_update` 个环境步硬同步；`tau < 1` 时**每个梯度步**做一次
  Polyak 平均（不是每个采集周期——`train_freq=4` 时两者差 4 倍）。
- 截断：`terminated=False` 时照常 bootstrap，使用 `next_obs`（终局帧）。
- 梯度裁剪：`max_grad_norm <= 0` 关闭（不再有独立的 `grad_clip` 布尔开关，避免两套语义）。
- 更新频率：`train_freq` 以**转移数**计。一个向量步收集 `num_envs` 条转移，
  攒够 `train_freq` 条做 `gradient_steps` 次更新；`learning_starts` 之前不学习，
  同时限制"欠账"上限，避免开训瞬间突发。
- PER：`td_error` 回写优先级；loss 乘重要性权重。
- 探索：`LinearSchedule` 注入，评估时用 `greedy`。

### 8.3 调度器

`epsilon` 与 `beta` 都是 `Callable[[int], float]` 注入，算法不关心退火方式；
测试传 `Constant`，工厂传 `LinearSchedule`。

---

## 9. 缓冲层

### 9.1 RolloutBuffer（on-policy）

- 预分配 `[S,N,...]` 的 obs / next_obs / action / reward / logprob / value /
  terminated / truncated，采集时 `copy_` 原地写。
- `flat()` 返回 `(S·N, ...)` **视图**；每个 epoch 生成一次 `randperm`，
  minibatch 用切片，obs 用 `index_select(out=预分配)`，不整批拷贝。
- 内存：`2 · S · N · T · frame_bytes`。

### 9.2 均匀回放

- 环形缓冲，`add` 支持批量写入并自动处理回绕。
- 观测按环境 dtype 存储（snake 为 uint8）；采样用
  `index_select(..., out=预分配)`，`torch.as_tensor(idx)` 只搬运索引。
- 同时保存 `terminated` 与 `truncated`，为 n-step/分析与正确的 bootstrap 留接口。
- 设备：`buffer.device=auto` 跟随计算设备；CPU 缓冲在 CUDA 可用时 pinned，
  采样后 `to(device, non_blocking=True)`。

### 9.3 优先级回放

- 分层 numpy 数组的 sum tree；采样按层向量化下降，一批样本 67 µs（256 条）。
- 优先级 `p = (|td| + eps)^alpha`，新样本取当前最大优先级。
- 重要性权重 `w = (N·P)^(-β)` 归一化到最大值为 1，`β` 从 `beta0` 线性退火到 1。
- 采样可能重复命中同一 slot，更新优先级前先 `np.unique` 去重（见 §0.2）。

---

## 10. 训练编排

- `Trainer`：`while env_steps < total_steps: metrics = algorithm.iterate()`，
  然后依次触发回调。不包含任何算法分支。
- `Evaluator`：独立环境实例、贪心策略、统计 episode 回报/长度；
  不依赖环境是否暴露 score，因此对任意环境通用。`train.eval_max_steps` 按**每个环境**
  计步（不是整批总和），否则 16 个并行评估环境 × 2000 步的回合永远跑不完；
  一个回合都没完成时会发 `RuntimeWarning` 并返回 NaN，而不是静默给出空结果。
- `MetricLogger`：JSONL + 控制台，可选 TensorBoard（惰性 import）。
- `CheckpointCallback`：`config + algorithm.state_dict() + counters`，
  `torch.load(map_location=...)` 跨设备恢复；`latest.pt` 始终指向最后一次。
- `CloseLoggerCallback` 最后执行，保证 eval 的末行能落盘。

---

## 11. 性能设计与实测

### 11.1 采集路径（E1/E2/E3）

| 项目 | 实测 | 设计含义 |
|---|---|---|
| 引擎 step（1024 envs, grid） | 23–40 µs | 引擎不是瓶颈 |
| grid 观测拷贝 1024 envs | 25.8 µs（30 GB/s） | 与 step 同量级，因此观测必须少拷贝 |
| 位棋盘解码 | 285 µs（+拷贝 426 µs） | 比直接拷 grid 慢 11–16×，弃用 |
| 窗口平移（256 envs, T=4, 4 线程） | ~20 µs，原地 `copy_` | 预分配 + 原地平移，不做 `cat` 分配 |
| 单帧窗口（T=1） | 双缓冲交换后 `runner.step` 4.84 ms → **0.775 ms** | 不为 `(B,1,C,H,W)` 做跨步写 |
| CPU 采集拷贝 | torch `copy_` **2.07 ms** vs numpy `copyto` **0.094 ms**（76 KiB，紧随引擎并行段） | CPU 路径改用 numpy 赋值；与 torch 线程数解耦 |
| 视图创建 | 1.8 µs | 每次 step 的 Python 开销可接受 |

端到端（env step + 采集 + 256 批 CNN 前向）：

| env_threads \ torch_threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 1 | 13.6 ms | 20.9 | 10.1 | 18.8 |
| 2 | 13.9 | 10.3 | **6.8** | 12.1 |
| 4 | 13.6 | 11.0 | 10.4 | 14.2 |

（N=256，单位 ms/step；越低越好。8 线程一律最差。）

### 11.2 回放（E4）

| 项目 | uint8 | float32 |
|---|---:|---:|
| 200k 转移内存（obs+next） | 293 MiB | 1172 MiB |
| 采样 256 条（4 线程） | 6.6 µs | 6.6 µs + 转换 8 µs |
| 线程=8 时采样 | 407 µs | 421 µs |

PER：采样 256 条 67 µs、更新 256 个优先级 40 µs，相对一次 DQN 更新可忽略。

### 11.3 硬性规范（代码中已落实）

1. 采集阶段 `torch.no_grad()` + `policy.eval()`；更新阶段 `policy.train()`。
2. 缓冲区、窗口、minibatch 输出全部预分配；热路径无 `torch.zeros`/`cat`。
3. 指标累积在张量里，一次 iterate 只做一次 device→host 同步。
4. `optimizer.zero_grad(set_to_none=True)`；梯度裁剪可选。
5. `torch.set_num_threads` 保守默认，并与引擎线程池协调（`run.torch_threads`）。
6. 回放/窗口存 uint8；只在网络入口转 float32。
7. CUDA 下 pinned staging + `non_blocking=True`；小标量也走 pinned，
   避免每步同步把流水线打断。
8. 线性层走 `FastLinear`；`SNAKE_DRL_LINEAR=addmm` 可关闭。
9. 普通 DQN 不再计算 online 网络的 next-state Q（DDQN 才需要），省掉一次整批前向；
   PPO 的自举只走 `head.value`，不计算 actor 头。
10. 回放只 pin 采样输出、不 pin 整块存储（100k 转移的 pinned 池会把显存/锁页内存吃光）；
    `buffer.device=cpu` 时 `ReplayBatch.to(device)` 负责异步搬移。
11. checkpoint 的 `latest.pt` 用 `shutil.copyfile` 流式复制，不再把整个文件读进内存两次。
12. 不默认开 `torch.compile` / AMP：前者首次编译开销大且 aarch64 稳定性未知，
   后者只对 CUDA 有意义；都通过配置留作扩展。
10. T=1 的窗口用双缓冲（`_frame` / `_frame_alt`）交换，避免向
    `(B,1,C,H,W)` 的跨步切片写入；T>1 才做整窗平移。

### 11.4 内存公式

- rollout：`2 · S · N · T · frame_bytes`（obs + next_obs）。
- replay：`2 · capacity · T · frame_bytes`。
- 例：16×16、3 通道、T=1、S=128、N=256 → rollout ≈ 48 MiB；
  capacity=100k 的回放 ≈ 146 MiB（uint8，obs+next_obs）。

---

## 12. 配置与 CLI

配置是 dataclass；YAML 用 OmegaConf 读取与合并，`dacite` 严格模式校验
（未知字段直接报错），算法块按 `type` 分发到 `PPOConfig` / `DQNConfig`。

```bash
snake-drl train -c configs/d3qn_cnn.yaml --set algorithm.lr=1e-4
snake-drl eval  -m runs/<run>/latest.pt --episodes 50 --set env.num_envs=8
snake-drl bench -c configs/dqn_mlp.yaml
snake-play
```

命令行覆盖支持 `key.path=value`，类型由 OmegaConf 解析；缺少 `=` 会直接报错
（而不是把 `None` 送进类型转换）。字符串布尔按严格规则解析（`"false"` 是 False，
`bool("false")` 是 True 这个坑不会再出现）。

配置只校验**字段的形状**，不校验"有哪些实现"：`network.encoder.type` / `buffer.type`
的合法名字由各自的注册表决定，因此用户注册的自定义编码器/回放在 YAML 里可以直接写。
蛇特有的校验（棋盘尺寸、`obs_mode`、`action_mode`）放在 `SnakeVecEnv` 里，自定义环境
不会被这些规则误伤。

`eval` 的覆盖**作用在 checkpoint 自带的配置之上**（不是另找一份配置）；结构性覆盖
（`network.*`、`env.width/height`）会让权重与配置对不上，此时会给出明确错误而不是
torch 的 `size mismatch` 堆栈。`eval` 不再接受单独的 `-c`（那会重新定义网络结构）。

---

## 13. 测试

`pytest -q`（CPU，约 1–2 分钟，含学习用例）：

| 文件 | 覆盖 |
|---|---|
| `test_config.py` | 默认值、算法分发、预设与显式开关、未知字段、YAML 覆盖、非法值 |
| `test_spaces.py` | ObsSpec/ActionSpec 校验与窗口形状 |
| `test_encoders.py` | 六种编码器 × image/vector × frames、非法组合、梯度有限 |
| `test_policies.py` | logits/value 形状、logprob 与 evaluate_actions 一致、dueling 分解、scale 不修改输入 |
| `test_buffers.py` | sum tree 分布/重复索引、均匀回放回绕、PER 权重与优先级、rollout 满/复位 |
| `test_env.py` | snake 规格、死亡惩罚、截断、玩具环境语义 |
| `test_runner.py` | 窗口正确性、终止帧 `next_obs`、新 episode 复位、向量观测 |
| `test_algorithms.py` | DQN/DDQN 目标、截断 bootstrap、GAE vs 朴素实现、PER 回写、四算法在老虎机上收敛 |
| `test_trainer.py` | 端到端训练、指标/评估/checkpoint 文件、断点恢复、步数上限、total_steps 一致性、eval 覆盖优先级 |

共 85 个用例（4 个"时序编码器需要 frames>=2"的预期跳过）。CI 级别的命令：

```bash
pytest -q
ruff check python/snake_drl tests
ruff format --check python/snake_drl tests --exclude python/snake_drl/play
```

---

## 14. 扩展点

- **新编码器**：实现 `Encoder` 并 `@ENCODERS.register("name")`，在 YAML 里写 `type: name`。
- **新回放**：实现 `add/sample/update_priorities/__len__` 并注册到 `BUFFERS`。
- **新环境**：实现 `VectorEnv`（numpy），注册到 `ENVS`；复杂参数走 `env.kwargs`。
- **新算法**：实现 `Algorithm.iterate()`，在工厂里装配它的策略族与缓冲区。
- **新调度**：任何 `Callable[[int], float]` 都可以传给 DQN 的 `epsilon` / `beta`。
- **新回调**：日志/评估/checkpoint 都是 `Callback`，可自由增删。

---

## 15. 已知限制

1. 时序编码采用**窗口**而非携带隐状态：记忆深度受 `frames` 限制，
   观测内存随 `frames` 线性增长。接口上保留升级空间（`Encoder.forward` 的第一维是时间）。
2. 回放同时保存 `obs` 与 `next_obs`（2× 内存），换取截断 bootstrap 的严格正确。
3. 未实现 n-step 回报、分布式 DQN、PPO 的循环策略、连续动作。
4. ConvGRU / ConvTransformer 在 CPU 上偏慢，默认配置不启用。
5. 本开发机无 CUDA，GPU 路径按设备无关方式实现（pinned/异步/AMP 开关），
   但未在本机做 GPU 基准；`snake-drl bench` 可在目标机器上复测。
6. snake 的观测是 body/head/food 三个平面，朝向可由蛇头与相邻身体格推断；
   若需要显式方向通道，可通过环境包装器扩展（不影响算法层）。
7. checkpoint 保存策略/目标网络/优化器/计数器，**不保存回放内容**（100k 转移约 150 MiB）；
   恢复后从空回放继续。恢复是库级能力（`load_state_dict`），CLI 没有 `--resume`——
   需要时用同一个 config 重建算法再加载即可；评估走 `load_policy`。


---

## 16. 端到端验证（E7）

任务：8×8 棋盘、256 并行环境、每局最多 300 步、CPU、MLP 编码器。
这不是调参基准，只用于验证"引擎 → 采集 → 回放 → 更新 → 评估 → checkpoint"
整条链路能真正学起来。随机策略基线：**回报 -0.06、存活 2 步**（撞墙即死）。

| 算法 | 步数 | 评估回报（20 局） | 单局最大回报 | 平均存活 |
|---|---:|---:|---:|---:|
| 随机策略 | — | -0.06 | — | 2 |
| D3QN + MLP | 25k | 0.72 | 1 | 229 |
| D3QN + MLP | 50k | 1.16 | 3 | 129 |
| D3QN + MLP | 75k | 2.61 | 7 | 144 |
| D3QN + MLP | 100k | 1.84 | 5 | 201 |
| D3QN + MLP（CLI eval 复测） | 100k | 2.16 | 6 | 203 |
| PPO + MLP | 65k | 0.31 | 1 | 300 |
| PPO + MLP | 130k | 0.44 | 1 | 268 |
| PPO + MLP | 200k | 0.63 | 1 | 226 |
| PPO + MLP | 260k | 0.65 | 1 | 201 |
| PPO + MLP | 300k | 0.80 | 1 | 187 |

结论：

- 离线算法（D3QN + PER）在 10 万步内明显优于 PPO，符合稀疏奖励、长回合任务的
  样本效率预期；PPO 需要更多样本或奖励塑形。
- 两者都显著高于随机基线，且 D3QN 的"存活 + 进食"行为出现得很早。
- 100k 步的 checkpoint 用 `snake-drl eval` 复测得到 2.16，与训练期评估一致，
  说明评估路径与训练路径使用同一套观测/动作语义。

复现：

```bash
snake-drl train -c configs/dqn_mlp.yaml --set algorithm.type=d3qn \
  --set algorithm.lr=3e-4 --set algorithm.train_freq=4 \
  --set algorithm.eps_decay_steps=60000 --set env.width=8 --set env.height=8 \
  --set train.total_steps=100000

snake-drl eval -m runs/<run>/latest.pt --episodes 20
```

---

## 17. 代码质量修订（v2.1）

在不改变行为的前提下（69 个测试全绿）按六个维度做了一次系统性审查与重构。

### 17.1 模块化设计

- `nn/encoders.py`（338 行、6 个类）拆成 `nn/encoders/` 包：`base.py` + 每种网络一个
  文件 + `registry.py`。新增一种网络不需要修改任何既有文件。
- 三个注册表各自独立成 `registry.py`（`nn/encoders/registry.py`、`buffers/registry.py`、
  `envs/registry.py`），从结构上消除"实现模块 → 包 `__init__` → 实现模块"的循环导入。
- `buffers/base.py` 拆成 `batch.py`（数据容器）/ `protocols.py`（算法依赖的协议）/ `registry.py`。
- 算法共用的优化器步骤抽到 `algorithms/optim.py`，两处重复的
  `zero_grad/backward/clip/step` 收敛为一处。
- `StepBatch` 从 `spaces.py` 移到 `runner.py`：它携带 `torch.Tensor`，而 `spaces` 要
  保持框架无关（envs 层至今没有一行 `import torch`）。

### 17.2 可读性与可维护性

- 公共类/函数全部有 docstring，实现细节一律 `_` 前缀。
- 魔法数字命名：`_MAX_BANKED_TRIGGERS`、`_STAT_KEYS`、`_TEMPORAL`、`_ALL`。
- PPO 约 90 行的 `_update` 拆成 `_advantages` / `_run_epochs` / `_value_loss`，
  DQN 的 `_iterate` 拆出 `_credit_updates`。
- 类型标注现代化：`list[int]`、`X | None`、`collections.abc.Callable`。
- `ruff`（`E,F,W,I,UP,B,SIM`）与 `ruff format` 全绿，规则写进 `pyproject.toml`。
- `mypy` 对 49 个源文件零错误：策略族按具体类收窄（`PPO.policy: ActorCriticPolicy`、
  `DQN.policy: QPolicy`），检查点负载用 `dict[str, Any]`，删掉了全部 `type: ignore`。
  `play/` 与 `cli.py` 排除：本机 PyQt6 的 `.pyi` 在 mypy 2.x 下无法解析
  （`'/' parameter must appear before '*'`），而这两处是薄适配层。

### 17.3 底层屏蔽

- `SnakeVecEnv` 的引擎句柄私有化（`_engine` + `_require_engine()`），关闭后再用会得到
  `RuntimeError: this SnakeVecEnv has been closed`，而不是 `AttributeError: 'NoneType'`；
  `close()` 可重复调用。
- 回放的存储张量、采样输出缓冲、sum tree 全部私有，对外只有
  `add / sample / update_priorities / __len__`。
- 顶层 `snake_drl` 只导出 17 个名字（引擎 + 配置 + 训练入口 + 三个注册表），
  `build_policy` 之类的装配细节留在 `snake_drl.nn`。

### 17.4 语义一致性

- **`updates` 统一为"优化器步数"**：此前 PPO 每次 `iterate` 记 1、DQN 每个梯度步记 1，
  两个数字不可比。现在两者都按优化器步计，另加 `iterations` 表示 `iterate()` 调用次数；
  checkpoint 同时保存两者。
- **`total_steps` 单一来源**：`Trainer` 负责写入并校验 `algorithm.total_steps`，
  与算法构造时的值不一致直接报错（学习率调度不再可能静默失效）。
- **环境工厂签名统一**：全部是 `factory(cfg, *, seed=None)`；玩具环境的
  `max_episode_steps` 与引擎环境语义一致（早期版本会让评估永不结束）。
- 删除 `grad_clip`：它和 `max_grad_norm <= 0` 是同一个开关的两种写法，配置里只保留一种。
  迁移：v2.1 之前保存的 checkpoint 的 config 里含该字段，严格解析会拒绝；重新训练即可
  （旧 checkpoint 仅存在于本地实验目录，未随仓库分发）。
- `eval` 的 `--set` 覆盖作用在 **checkpoint 自带配置** 之上，不再接受会重新定义网络结构的 `-c`。

### 17.5 最小暴露

删除以下未被使用或已经过时的公开面：`TOY_ENVS`、`parse_override`、`make_rng`、
`Runtime.is_cuda`、`ObsSpec.windowed`、`RolloutProtocol`、两个缓冲区上的 `nbytes`、
`Algorithm.n_actions` / `Algorithm.frames`、`buffers/base.py`、`DQN._last_metrics`。

### 17.6 性能隐患

- 普通 DQN 不再计算 online 网络的 next-state Q：只有 DDQN/D3QN 需要它。
  同机同负载下 `_update` 从 36.6 ms 降到 18.7 ms（省掉一次整批前向）。
- PPO 自举只走 `ActorCriticHead.value`，不再顺带计算 actor 头。
- 回放只 pin 采样输出，不 pin 整块存储：100k 转移的 pinned 池会耗尽锁页内存，
  而异步拷贝只要求源张量是 pinned。
- `ReplayBatch.to(device)` 统一设备语义：修复了 `buffer.device=cpu` + GPU 训练时
  直接把 CPU 张量喂给 GPU 网络的问题。
- checkpoint 的 `latest.pt` 改为 `shutil.copyfile` 流式复制，不再把整个文件读入内存两次。
- 指标继续"一次 iterate 一次同步"；PPO 的统计量改为一次 `stack` + 一次 `tolist()`。
- 训练结束时不再重复评估/重复写 checkpoint：`EvalCallback` 与 `CheckpointCallback`
  记录最后一步，`on_end` 只在最后一步尚未处理时才执行。

### 17.7 审查中发现的缺陷

- **PPO 的梯度裁剪只生效一次**：`_run_epochs` 把 `policy.parameters()` 生成器存进局部变量，
  第一次 `clip_grad_norm_` 就把它耗尽，后续 minibatch 实际不再裁剪。
  现在两个算法都在构造时物化 `self._parameters = list(policy.parameters())`，
  `optimizer_step` 的文档也写明"必须可重复迭代"；新增的回归测试用 spy 统计每次裁剪的
  参数个数，旧实现会在第二次之后出现 0。
- 训练最后一步重复评估/重复写 checkpoint（已由 §17.6 的末步记录去掉）。

### 17.8 第二轮审查（Python 端）

这一轮只修"有真实场景"的问题，未引入任何新抽象：

- **`epochs: 0` 直接崩溃**（`torch.stack` 不接受空列表）。一个不跑 epoch 的 PPO 没有意义，
  改为配置校验 `epochs >= 1`，而不是在算法里加空列表分支。
- **评估预算按整批计步导致"越好越评不出来"**：`eval_max_steps` 原意是总步数，
  16 个评估环境 × 2000 步回合时只跑了 312 步/环境，一个回合都完不成，返回 NaN。
  现在按**每个环境**计步，并在零回合完成时发出 `RuntimeWarning`。
- **metrics.jsonl 里写 NaN 不是合法 JSON**：严格解析器（如带 `parse_constant` 的 json、
  部分日志工具）会拒绝。非有限值改写为 `null`。
- **`network.head` 与算法可能矛盾且被静默忽略**：`head: dueling_q` + `algorithm.type: dqn`
  实际建的是非 dueling 网络。现在直接报错并提示改用 `d3qn` / `dueling: true`。
- **软目标网络的更新频率错了**：`tau < 1` 原本每个采集周期更新一次，`train_freq=4`
  时比"每个梯度步更新"慢 4 倍。现在在每次梯度步后做 Polyak 平均。
- **同一秒启动的两个 run 会共用一个输出目录**（并行扫参时真实存在）：目录名冲突时
  自动加后缀，且用 `mkdir(exist_ok=False)` 做到跨进程无竞争。
- 删除两处没有调用方的"扩展钩子"：`save_checkpoint(extra=...)`、`SumTree.capacity`。


### 17.9 第三轮：多视角评审（算法 / 数据路径 / API / 性能）

四个并行评审各自给出证据，逐条取舍如下。

**采纳（有真实场景且能复现）**

| 问题 | 场景 | 修复 |
|---|---|---|
| GAE 在截断步没有切断递推 | 任一有回合上限的环境，截断时把下一回合的回报借给前一步（实测单步 advantage 偏 0.67） | `compute_gae` 增加 `truncated` 参数，递推用 `terminated\|truncated`，`δ` 仍用 `terminated` |
| `adv.std()` 在单样本 rollout 上是 NaN | `num_envs=1, n_steps=1` 的冒烟配置，第一次更新就把全部权重写成 NaN | 改为 `std(unbiased=False)` |
| 配置白名单挡住了自定义组件 | 注册 `my_encoder` 后写进 YAML 直接被配置层拒绝，与 README 的扩展说明矛盾 | 删除硬编码白名单，由注册表给出"可用名字"错误 |
| `network.head` 与算法矛盾时被静默忽略 | `head: dueling_q` + `type: dqn` 实际建的是非 dueling 网络 | 直接报错并提示用 `d3qn` / `dueling: true` |
| DQN 就绪门槛不可达时静默零更新 | `capacity < learning_starts`（省内存的配置）整轮不学习 | 配置校验直接拒绝 |
| `target_update >= total_steps` | 目标网络整轮不同步，全部自举来自初始网络 | 配置校验拒绝 |
| `--set` 缺少 `=` | 用户少打一个字符，得到 dacite 的 `TypeError` | 覆盖解析阶段报 `KEY=VALUE` |
| YAML 字符串布尔被 `bool("false")` 反转 | 生成的配置/环境变量插值常是字符串 | dacite `type_hooks` 严格解析 |
| 输出头仍用 `nn.Linear` | 头层是 `addmm` 热点（15–18% 的单步开销） | 头层也换 `FastLinear` |
| CPU 采集用 torch 拷贝 | 紧跟引擎并行段时 76 KiB 拷贝 2.07 ms（numpy 0.094 ms） | CPU 路径改 numpy 赋值；`runner.step` 2.89 ms → **0.15 ms**，且与线程数无关 |
| T>1 窗口跨回合携带旧帧 | GRU/Transformer 在新回合前 `frames-1` 步看到上一局的画面 | 回合结束后整窗重置为新首帧（与 `reset()` 一致） |
| 固定 pinned staging 的非阻塞 H2D 被立即复用 | CUDA + `pin_memory`（默认）：下一步覆盖 staging 时上一份传输可能还没读 | 改为同步拷贝（76 KiB ≈ 10 µs），并在注释里说明原因 |
| 玩具环境 `close()` 后仍可用 | 与 `SnakeVecEnv` 的语义不一致 | 加 closed 守卫 |
| PPO 采集用 eval、更新用 train，dropout 让 ratio 失真 | `dropout > 0` 时 ratio 不是同一个函数的比值（实测 Δlogprob 0.0044） | PPO 配置直接拒绝 `dropout > 0`，并提示 DQN 支持 |
| `ActionOutput.entropy` 采集时计算但没人用 | 每个环境步多算一次熵 | 删除该字段 |

**否决（属于过度设计 / 收益不成立）**

- **sum tree 用 `np.bincount` 代替 `np.add.at`**：实测更慢（116 µs vs 48 µs/次，
  因为每层都要分配整层大小的数组）。
- **把 PER 的优先级回写攒到每个向量步批量刷新**：只对"回放在 CPU、网络在 GPU"的
  组合有意义，而该组合已经通过阻塞拷贝保证正确；为它引入队列和刷新点不划算。
- **`target_kl` 的每 minibatch 同步改成每 N 个**：默认关闭，开启时这是刻意的即时止损；
  为了省一次同步去改变停止语义不值得，改为在文档中说明该路径会有一次同步。
- **把每 minibatch 的花式索引改为预分配**：256 条 `adv[idx]` 约 25 µs，对比 40 ms 的
  minibatch 可忽略。
- **GAE 的 Python 时间循环向量化**：每次 iterate 一次，约 3 ms，对比数秒的更新可忽略。

