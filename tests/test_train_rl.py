import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gym_wrapper.learned_pahm_ode import _load_wrapper_config

from train_rl import build_agent, load_config, make_training_env, train_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pahm_model" / "pahm_fast_v2_best.pth"


def _write_config(tmp_path: Path, *, mode: str = "naive", total_timesteps: int = 10) -> Path:
    config = copy.deepcopy(_load_wrapper_config())
    config["rl_training"] = {
        "enabled": True,
        "mode": mode,
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "n_steps": 8,
        "batch_size": 4,
        "device": "cpu",
        "render": False,
        "model_path": str(MODEL_PATH),
        "reset_angle_deg": 720,
        "model_output_dir": str(tmp_path / "models"),
        "model_name": f"pahm_ppo_{mode}",
        "checkpoint_freq": 0,
        "log_dir": str(tmp_path / "logs"),
    }
    path = tmp_path / f"{mode}_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_rl_training_config_loads_from_external_file(tmp_path):
    config_path = _write_config(tmp_path, mode="naive")

    config = load_config(config_path)

    assert config["rl_training"]["mode"] == "naive"
    assert config["rl_training"]["algorithm"] == "PPO"
    assert config["rl_training"]["render"] is False


def test_make_training_env_is_headless_and_resets_with_randomize(tmp_path):
    config = load_config(_write_config(tmp_path, mode="naive"))
    env = make_training_env(config)
    try:
        obs, info = env.reset(options={"randomize": True})

        assert env.render_mode is None
        assert obs.shape == (3,)
        assert np.isfinite(obs).all()
        assert "theta_ref" in info
        assert info["wind_automatic"] is False
    finally:
        env.close()


def test_naive_mode_disables_wind(tmp_path):
    config = load_config(_write_config(tmp_path, mode="naive"))
    env = make_training_env(config)
    try:
        _, info = env.reset(options={"randomize": True})

        assert env.enable_wind is False
        assert info["wind_automatic"] is False
    finally:
        env.close()


def test_robust_mode_enables_wind(tmp_path):
    config = load_config(_write_config(tmp_path, mode="robust"))
    env = make_training_env(config)
    try:
        _, info = env.reset(options={"randomize": True})

        assert env.enable_wind is True
        assert info["wind_automatic"] is True
    finally:
        env.close()


def test_training_env_step_does_not_call_render(tmp_path, monkeypatch):
    config = load_config(_write_config(tmp_path, mode="naive"))

    def fail_render(self):
        raise AssertionError("render must not be called during headless training")

    monkeypatch.setattr("gym_wrapper.learned_pahm_ode.LearnedPAHMODE.render", fail_render)
    env = make_training_env(config)
    try:
        env.reset(options={"randomize": True})
        env.step(np.array([0.0]))
    finally:
        env.close()


def test_build_agent_from_config_when_sb3_is_available(tmp_path):
    pytest.importorskip("stable_baselines3")
    config = load_config(_write_config(tmp_path, mode="naive"))
    env = make_training_env(config)
    try:
        agent = build_agent(config["rl_training"]["algorithm"], env, config)

        assert agent.__class__.__name__ == "PPO"
    finally:
        env.close()


def test_short_training_saves_model_when_sb3_is_available(tmp_path):
    pytest.importorskip("stable_baselines3")
    config_path = _write_config(tmp_path, mode="naive", total_timesteps=10)

    model_path = Path(train_from_config(config_path))

    assert model_path.exists()
    assert model_path.name == "pahm_ppo_naive.zip"
