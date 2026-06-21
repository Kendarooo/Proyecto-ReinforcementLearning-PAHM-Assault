import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gym_wrapper.learned_pahm_ode import _load_wrapper_config

from train_rl import (
    build_agent,
    load_config,
    make_training_env,
    train_all_modes,
    train_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pahm_model" / "pahm_fast_v2_best.pth"


def _write_config(tmp_path: Path, *, mode: str = "naive", total_timesteps: int = 10) -> Path:
    config = copy.deepcopy(_load_wrapper_config())
    config["wind"]["stage2_sampler_checkpoint"] = str(
        PROJECT_ROOT / "artifacts" / "stage2" / "gmm_wind_model.pkl"
    )
    config["rl_training"] = {
        "enabled": True,
        "mode": mode,
        "algorithm": "PPO",
        "seed": 42,
        "total_timesteps": total_timesteps,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "n_steps": 8,
        "batch_size": 4,
        "device": "cpu",
        "render": False,
        "model_path": str(MODEL_PATH),
        "reset_angle_deg": 720,
        "max_episode_steps": 25,
        "model_output_dir": str(tmp_path / "models"),
        "model_name": f"pahm_ppo_{mode}",
        "checkpoint_freq": 0,
        "log_dir": str(tmp_path / "logs"),
    }
    config["experiments"] = {
        "modes": ["naive", "robust"],
        "naive": {
            "wind_enabled": False,
            "model_name": "pahm_ppo_naive",
        },
        "robust": {
            "wind_enabled": True,
            "wind_source": "stage2_sampler",
            "model_name": "pahm_ppo_robust",
        },
    }
    config["wandb"] = {
        "enabled": False,
        "project": "pahm-rl-stage3",
        "entity": None,
        "mode": "disabled",
        "tags": ["stage3", "rl", "pahm"],
    }
    path = tmp_path / f"{mode}_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class FakeAgent:
    def __init__(self):
        self.learn_calls = []
        self.saved_paths = []

    def learn(self, total_timesteps: int, callback=None):
        self.learn_calls.append((total_timesteps, callback))
        return self

    def save(self, path: str):
        model_path = Path(path).with_suffix(".zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake model", encoding="utf-8")
        self.saved_paths.append(model_path)


class FakeTrainingEnv:
    def __init__(self):
        self.reset_calls = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append({"seed": seed, "options": options})
        return np.zeros(4, dtype=np.float32), {}

    def close(self):
        self.closed = True


def test_rl_training_config_loads_from_external_file(tmp_path):
    config_path = _write_config(tmp_path, mode="naive")

    config = load_config(config_path)

    assert config["rl_training"]["mode"] == "naive"
    assert config["rl_training"]["algorithm"] == "PPO"
    assert config["rl_training"]["seed"] == 42
    assert config["rl_training"]["render"] is False


def test_config_defines_naive_and_robust_experiment_modes(tmp_path):
    config = load_config(_write_config(tmp_path, mode="naive"))

    assert config["experiments"]["modes"] == ["naive", "robust"]
    assert config["experiments"]["naive"]["wind_enabled"] is False
    assert config["experiments"]["robust"]["wind_enabled"] is True
    assert config["experiments"]["robust"]["wind_source"] == "stage2_sampler"


def test_make_training_env_is_headless_and_resets_with_randomize(tmp_path):
    config = load_config(_write_config(tmp_path, mode="naive"))
    env = make_training_env(config)
    try:
        obs, info = env.reset(options={"randomize": True})

        assert env.unwrapped.render_mode is None
        assert obs.shape == (4,)
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

        assert env.unwrapped.enable_wind is False
        assert info["wind_automatic"] is False
    finally:
        env.close()


def test_robust_mode_enables_wind(tmp_path):
    config = load_config(_write_config(tmp_path, mode="robust"))
    env = make_training_env(config)
    try:
        _, info = env.reset(options={"randomize": True})

        assert env.unwrapped.enable_wind is True
        assert info["wind_automatic"] is True
    finally:
        env.close()


def test_train_from_config_accepts_mode_override_and_saves_named_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("train_rl.build_agent", lambda *args, **kwargs: FakeAgent())
    config_path = _write_config(tmp_path, mode="naive", total_timesteps=1)

    model_path = Path(train_from_config(config_path, mode="robust"))

    assert model_path.exists()
    assert model_path.name == "pahm_ppo_robust.zip"


def test_train_all_modes_saves_one_model_per_configured_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("train_rl.build_agent", lambda *args, **kwargs: FakeAgent())
    config_path = _write_config(tmp_path, mode="naive", total_timesteps=1)

    model_paths = train_all_modes(config_path)

    assert set(model_paths) == {"naive", "robust"}
    assert Path(model_paths["naive"]).name == "pahm_ppo_naive.zip"
    assert Path(model_paths["robust"]).name == "pahm_ppo_robust.zip"
    assert Path(model_paths["naive"]).exists()
    assert Path(model_paths["robust"]).exists()


def test_build_agent_passes_configured_seed_to_sb3(tmp_path, monkeypatch):
    captured_kwargs = {}

    class FakePPO:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_module = SimpleNamespace(PPO=FakePPO, A2C=FakePPO, SAC=FakePPO)
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_module)

    config = load_config(_write_config(tmp_path, mode="naive"))
    env = object()

    build_agent(config["rl_training"]["algorithm"], env, config)

    assert captured_kwargs["seed"] == 42
    assert captured_kwargs["env"] is env


def test_train_from_config_resets_env_with_configured_seed(tmp_path, monkeypatch):
    fake_env = FakeTrainingEnv()
    monkeypatch.setattr("train_rl.make_training_env", lambda config: fake_env)
    monkeypatch.setattr("train_rl.build_agent", lambda *args, **kwargs: FakeAgent())
    config_path = _write_config(tmp_path, mode="naive", total_timesteps=1)

    train_from_config(config_path)

    assert fake_env.reset_calls[0]["seed"] == 42
    assert fake_env.reset_calls[0]["options"] == {"randomize": True}
    assert fake_env.closed is True


def test_wandb_mock_receives_saved_model_path(tmp_path, monkeypatch):
    logged_payloads = []
    logged_artifacts = []

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path):
            self.files.append(str(path))

    class FakeWandbRun:
        _allowed_attrs = {"logged_payloads"}

        def __setattr__(self, name, value):
            if name not in self._allowed_attrs:
                raise Exception(f"Attribute {name} is not supported on Run object.")
            super().__setattr__(name, value)

        def __init__(self):
            self.logged_payloads = logged_payloads

        def log(self, payload):
            self.logged_payloads.append(payload)

        def log_artifact(self, artifact):
            logged_artifacts.append(artifact)

        def finish(self):
            self.logged_payloads.append({"finished": True})

    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeWandbRun()

    fake_wandb = SimpleNamespace(init=fake_init, Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr("train_rl.build_agent", lambda *args, **kwargs: FakeAgent())

    config_path = _write_config(tmp_path, mode="naive", total_timesteps=1)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["wandb"]["enabled"] = True
    raw_config["wandb"]["mode"] = "disabled"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    model_path = train_from_config(config_path, mode="naive")

    assert Path(model_path).exists()
    assert init_kwargs["mode"] == "disabled"
    assert init_kwargs["config"]["training"]["algorithm"] == "PPO"
    assert any(payload.get("model/path") == model_path for payload in logged_payloads)
    assert any(artifact.type == "model" for artifact in logged_artifacts)
    assert any(artifact.type == "config" for artifact in logged_artifacts)
    assert {"finished": True} in logged_payloads


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


def test_training_env_has_time_limit_for_comparable_episode_returns(tmp_path):
    config = load_config(_write_config(tmp_path, mode="naive"))
    env = make_training_env(config)
    try:
        assert env._max_episode_steps == 25
        env.reset(options={"initial_state": [0.0, 0.0], "theta_ref": 0.0})

        truncated = False
        for _ in range(25):
            _, _, _, truncated, _ = env.step(np.array([0.0]))

        assert truncated is True
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


def test_short_robust_training_saves_model_when_sb3_is_available(tmp_path):
    pytest.importorskip("stable_baselines3")
    config_path = _write_config(tmp_path, mode="robust", total_timesteps=10)

    model_path = Path(train_from_config(config_path, mode="robust"))

    assert model_path.exists()
    assert model_path.name == "pahm_ppo_robust.zip"
