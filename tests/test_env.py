from __future__ import annotations

import numpy as np
import pytest
from snake_drl.config import EnvConfig
from snake_drl.envs import make_env


def test_snake_specs_and_step():
    env = make_env(EnvConfig(num_envs=4, width=9, height=7, num_threads=1), seed=1)
    assert env.obs_spec.shape == (3, 7, 9)
    assert env.action_spec.n == 3  # relative actions
    obs = env.reset()
    assert obs.shape == (4, 3, 7, 9) and obs.dtype == np.uint8
    result = env.step(np.zeros(4, dtype=np.int8))
    assert result.obs.shape == (4, 3, 7, 9)
    assert result.reward.shape == (4,)
    assert result.terminated.shape == (4,)
    assert result.truncated.shape == (4,)
    assert float(result.reward[0]) in (0.0, -1.0, 1.0)
    env.close()


def test_snake_absolute_actions():
    env = make_env(EnvConfig(num_envs=2, width=8, height=8, action_mode="absolute", num_threads=1))
    assert env.action_spec.n == 4
    env.reset()
    env.step(np.array([1, 3], dtype=np.int8))
    env.close()


def test_death_gives_penalty_and_final_observation():
    env = make_env(
        EnvConfig(num_envs=1, width=8, height=8, max_episode_steps=0, num_threads=1), seed=3
    )
    env.reset()
    for _ in range(20):
        result = env.step(np.zeros(1, dtype=np.int8))  # straight into the right wall
        if result.terminated[0]:
            assert float(result.reward[0]) == pytest.approx(-1.0)
            assert result.final_obs is not None
            assert not np.array_equal(result.final_obs[0], result.obs[0])
            break
    else:
        pytest.fail("snake never terminated when running straight into a wall")
    env.close()


def test_truncation_keeps_bootstrapping_obs():
    env = make_env(
        EnvConfig(num_envs=1, width=8, height=8, max_episode_steps=3, num_threads=1), seed=3
    )
    env.reset()
    seen_truncation = False
    for _ in range(3):
        result = env.step(np.zeros(1, dtype=np.int8))
        if result.truncated[0]:
            seen_truncation = True
            assert result.final_obs is not None
            assert not result.terminated[0]
    assert seen_truncation
    env.close()


def test_toy_environments_semantics():
    bandit = make_env(EnvConfig(type="bandit", num_envs=5, seed=0))
    obs = bandit.reset()
    assert obs.shape == (5, 3)
    result = bandit.step(obs.argmax(axis=1))
    assert np.allclose(result.reward, 1.0)
    assert result.terminated.all() and not result.truncated.any()

    corridor = make_env(EnvConfig(type="corridor", num_envs=4, seed=0))
    corridor.reset()
    done = False
    for _ in range(10):
        result = corridor.step(np.ones(4, dtype=np.int64))
        if result.terminated.all():
            done = True
            break
    assert done
    assert np.allclose(result.reward, 1.0)


def test_closed_snake_env_raises():
    env = make_env(EnvConfig(num_envs=2, width=8, height=8, num_threads=1))
    env.reset()
    env.close()
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_unknown_environment_type_is_a_config_error():
    with pytest.raises(ValueError, match="unknown environment type"):
        make_env(EnvConfig(type="nope", num_envs=1))
