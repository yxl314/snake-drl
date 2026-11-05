from __future__ import annotations

import torch

from snake_drl.config import EnvConfig
from snake_drl.envs import make_env
from snake_drl.runner import VectorRunner


def _straight_until_done(runner, steps=20):
    for i in range(steps):
        action = torch.zeros(runner.num_envs, dtype=torch.long)
        step = runner.step(action)
        if bool(step.terminated.any()):
            return i + 1, step
    raise AssertionError("environment did not terminate")


def test_runner_window_and_terminal_semantics():
    for frames in (1, 3):
        env = make_env(
            EnvConfig(num_envs=1, width=8, height=8, max_episode_steps=0, num_threads=1), seed=3
        )
        runner = VectorRunner(env, frames, "cpu", pin_memory=False)
        obs = runner.reset()
        assert obs.shape == (1, frames, 3, 8, 8)
        assert obs.dtype == torch.uint8

        _, step = _straight_until_done(runner)
        final = torch.from_numpy(env._env.final_observations()[0].copy())
        # the successor reported to the learner is the terminal frame...
        assert torch.equal(step.next_obs[0, -1], final)
        # ...while the environment window already belongs to the new episode
        assert not torch.equal(runner.obs[0, -1], final)
        if frames > 1:
            assert torch.equal(step.next_obs[0, :-1], step.obs[0, 1:])
            assert torch.equal(runner.obs[0, :-1], step.obs[0, 1:])
        runner.close()


def test_runner_obs_is_not_mutated_by_step():
    env = make_env(EnvConfig(num_envs=2, width=8, height=8, max_episode_steps=0, num_threads=1))
    runner = VectorRunner(env, 2, "cpu", pin_memory=False)
    obs = runner.reset()
    before = obs.clone()
    step = runner.step(torch.zeros(2, dtype=torch.long))
    assert torch.equal(obs, before)  # the returned window was not recycled
    assert torch.equal(step.obs, before)
    runner.close()
    env.close()


def test_runner_shapes_for_vector_obs():
    env = make_env(EnvConfig(type="corridor", num_envs=3, seed=0))
    runner = VectorRunner(env, 4, "cpu", pin_memory=False)
    obs = runner.reset()
    assert obs.shape == (3, 4, 6)
    assert obs.dtype == torch.float32
    step = runner.step(torch.zeros(3, dtype=torch.long))
    assert step.next_obs.shape == (3, 4, 6)
    assert step.reward.dtype == torch.float32
    runner.close()
