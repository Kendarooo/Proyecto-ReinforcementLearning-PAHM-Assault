import copy
from pathlib import Path

import numpy as np
import pytest

from gym_wrapper.learned_pahm_ode import _load_wrapper_config
from gym_wrapper.rl_policy import (
    apply_rl_control_step,
    get_demo_model_path,
    load_rl_policy,
    predict_rl_action,
)


def _demo_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(_load_wrapper_config())
    config["demo"] = {
        "control_mode": "rl",
        "rl_model_type": "robust",
        "models": {
            "naive": str(tmp_path / "pahm_ppo_naive.zip"),
            "robust": str(tmp_path / "pahm_ppo_robust.zip"),
        },
        "deterministic_policy": True,
        "render": True,
    }
    return config


class MockPolicy:
    def __init__(self, action):
        self.action = np.array(action, dtype=np.float32)
        self.received_obs = None
        self.learn_called = False

    def predict(self, obs, deterministic=True):
        self.received_obs = np.array(obs, dtype=np.float32)
        self.deterministic = deterministic
        return self.action, None

    def learn(self, *args, **kwargs):
        self.learn_called = True
        raise AssertionError("demo must not train policies")


class DummyActionSpace:
    def __init__(self):
        self.low = np.array([0.0], dtype=np.float32)
        self.high = np.array([1.0], dtype=np.float32)
        self.shape = (1,)
        self.dtype = np.float32

    def contains(self, action):
        action = np.asarray(action, dtype=np.float32)
        return action.shape == self.shape and np.all(action >= self.low) and np.all(action <= self.high)


class DummyEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
        self.actions = []
        self.step_info = {
            "theta_ref": 0.25,
            "tracking_error": -0.1,
            "wind_torque": 0.3,
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        obs = np.array([0.1, 0.0, 0.25], dtype=np.float32)
        return obs, 1.0, False, False, dict(self.step_info)


def test_demo_config_selects_robust_model_path(tmp_path):
    config = _demo_config(tmp_path)

    path = get_demo_model_path(config, model_type="robust")

    assert path == Path(config["demo"]["models"]["robust"])


def test_demo_config_selects_naive_model_path(tmp_path):
    config = _demo_config(tmp_path)

    path = get_demo_model_path(config, model_type="naive")

    assert path == Path(config["demo"]["models"]["naive"])


def test_load_rl_policy_reports_missing_model(tmp_path):
    missing_path = tmp_path / "missing.zip"

    with pytest.raises(FileNotFoundError, match="RL policy model not found"):
        load_rl_policy(missing_path)


def test_predict_rl_action_passes_observation_and_clips_to_action_space():
    policy = MockPolicy(action=[1.5])
    action_space = DummyActionSpace()
    obs = np.array([0.2, 0.0, 0.5], dtype=np.float32)

    action = predict_rl_action(policy, obs, action_space, deterministic=True)

    np.testing.assert_allclose(policy.received_obs, obs)
    assert policy.deterministic is True
    assert action_space.contains(action)
    np.testing.assert_allclose(action, np.array([1.0], dtype=np.float32))


def test_apply_rl_control_step_uses_policy_action_and_keeps_debug_info():
    env = DummyEnv()
    policy = MockPolicy(action=[0.4])
    obs = np.array([0.2, 0.0, 0.5], dtype=np.float32)

    next_obs, reward, terminated, truncated, info = apply_rl_control_step(
        env,
        policy,
        obs,
        deterministic=True,
        model_path="artifacts/stage3/models/pahm_ppo_robust.zip",
    )

    np.testing.assert_allclose(env.actions[-1], np.array([0.4], dtype=np.float32))
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info["control_mode"] == "RL"
    assert info["rl_model_path"] == "artifacts/stage3/models/pahm_ppo_robust.zip"
    assert np.isclose(info["rl_action"], 0.4)
    assert "theta_ref" in info
    assert "tracking_error" in info
    assert "wind_torque" in info
    assert next_obs.shape == (3,)
    assert policy.learn_called is False
