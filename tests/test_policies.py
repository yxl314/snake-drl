from __future__ import annotations

import pytest
import torch

from snake_drl.config import NetworkConfig, EncoderConfig
from snake_drl.nn import build_policy
from snake_drl.spaces import ActionSpec, ObsSpec

IMAGE = ObsSpec((3, 8, 8))
ACTIONS = ActionSpec(3)


def _network(kind: str = "cnn") -> NetworkConfig:
    return NetworkConfig(
        encoder=EncoderConfig(type=kind, frames=1, feature_dim=32, hidden_dims=[32],
                              cnn_channels=[8, 8])
    )


def test_actor_critic_shapes_and_logprob_consistency():
    policy = build_policy(_network(), IMAGE, ACTIONS, family="actor_critic", device="cpu")
    obs = torch.randint(0, 2, (6, 1, 3, 8, 8), dtype=torch.uint8)
    logits, value = policy(obs)
    assert logits.shape == (6, 3) and value.shape == (6,)
    out = policy.act(obs)
    assert out.action.shape == (6,) and out.logprob.shape == (6,)
    logp, entropy, value2 = policy.evaluate_actions(obs, out.action)
    assert torch.allclose(logp, out.logprob, atol=1e-5)
    assert torch.allclose(value, value2, atol=1e-5)
    assert (entropy >= 0).all()
    assert torch.equal(policy.greedy(obs), logits.argmax(dim=1))


def test_q_policy_and_dueling_decomposition():
    q_policy = build_policy(_network(), IMAGE, ACTIONS, family="q", device="cpu")
    obs = torch.randint(0, 2, (4, 1, 3, 8, 8), dtype=torch.uint8)
    q = q_policy(obs)
    assert q.shape == (4, 3)
    assert torch.equal(q_policy.act(obs, epsilon=0.0), q.argmax(dim=1))
    assert torch.equal(q_policy.greedy(obs), q.argmax(dim=1))

    dueling = build_policy(_network(), IMAGE, ACTIONS, family="q", device="cpu", dueling=True)
    q2 = dueling(obs)
    assert q2.shape == (4, 3)
    # dueling head: subtracting the mean advantage keeps V identifiable
    feats = dueling.encoder(dueling.prepare(obs))
    value = dueling.head.value(feats)
    advantage = dueling.head.advantage(feats)
    assert torch.allclose(value, (q2 - advantage + advantage.mean(dim=1, keepdim=True)),
                          atol=1e-5)


def test_head_family_mismatch_is_rejected():
    with pytest.raises(ValueError, match="only valid"):
        build_policy(NetworkConfig(head="q"), IMAGE, ACTIONS, family="actor_critic")
    with pytest.raises(ValueError, match="only valid"):
        build_policy(NetworkConfig(head="actor_critic"), IMAGE, ACTIONS, family="q")


def test_observation_scale_is_applied_without_mutating_input():
    spec = ObsSpec((3, 8, 8), scale=1 / 255.0)
    policy = build_policy(_network(), spec, ACTIONS, family="q", device="cpu")
    obs = torch.full((1, 1, 3, 8, 8), 255, dtype=torch.uint8)
    before = obs.clone()
    feats = policy.prepare(obs)
    assert torch.allclose(feats, torch.ones_like(feats))
    assert torch.equal(obs, before)
