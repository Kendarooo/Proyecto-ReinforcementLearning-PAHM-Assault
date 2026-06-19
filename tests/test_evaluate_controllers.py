import json
from pathlib import Path

import numpy as np

from evaluate_controllers import (
    compare_controllers,
    compute_overshoot,
    compute_settling_time,
    compute_tracking_mae,
    compute_tracking_mse,
    evaluate_controller,
    get_evaluation_model_paths,
)


class MockPolicy:
    def __init__(self, action=0.2):
        self.action = np.array([action], dtype=np.float32)
        self.predict_calls = 0
        self.learn_called = False

    def predict(self, obs, deterministic=True):
        self.predict_calls += 1
        self.last_obs = np.asarray(obs, dtype=np.float32)
        return self.action, None

    def learn(self, *args, **kwargs):
        self.learn_called = True
        raise AssertionError("evaluation must not train policies")


class DummyActionSpace:
    low = np.array([0.0], dtype=np.float32)
    high = np.array([1.0], dtype=np.float32)
    shape = (1,)
    dtype = np.float32


class MockEvaluationEnv:
    action_space = DummyActionSpace()
    render_mode = None
    dt = 0.1

    def __init__(self, trajectory=None):
        self.trajectory = trajectory or [0.0, 0.4, 0.7, 1.05, 1.0, 1.0]
        self.index = 0
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        assert options and options.get("randomize") is True
        self.index = 0
        obs = np.array([self.trajectory[0], 0.0, 1.0], dtype=np.float32)
        return obs, {"theta_ref": 1.0, "wind_torque": 0.0}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.index += 1
        theta = self.trajectory[min(self.index, len(self.trajectory) - 1)]
        obs = np.array([theta, 0.0, 1.0], dtype=np.float32)
        info = {
            "theta_ref": 1.0,
            "tracking_error": theta - 1.0,
            "wind_torque": 0.3,
        }
        terminated = self.index >= len(self.trajectory) - 1
        return obs, 1.0, terminated, False, info

    def close(self):
        self.closed = True


def _config(tmp_path: Path) -> dict:
    naive_path = tmp_path / "pahm_ppo_naive.zip"
    robust_path = tmp_path / "pahm_ppo_robust.zip"
    naive_path.write_text("mock naive", encoding="utf-8")
    robust_path.write_text("mock robust", encoding="utf-8")
    return {
        "_config_dir": str(tmp_path),
        "rl_training": {
            "algorithm": "PPO",
            "model_path": "pahm_model/pahm_fast_v2_best.pth",
            "reset_angle_deg": 720,
        },
        "evaluation": {
            "num_episodes": 1,
            "max_steps": 5,
            "render": False,
            "controllers": ["naive", "robust"],
            "models": {
                "naive": str(naive_path),
                "robust": str(robust_path),
            },
            "output_dir": str(tmp_path / "evaluation"),
            "metrics_path": str(tmp_path / "evaluation" / "controller_metrics.json"),
            "settling_tolerance": 0.05,
            "settling_window": 2,
            "unseen_wind_patterns": ["gust", "turbulent"],
        },
    }


def test_evaluation_model_paths_are_loaded_from_config(tmp_path):
    config = _config(tmp_path)

    paths = get_evaluation_model_paths(config)

    assert paths["naive"] == Path(config["evaluation"]["models"]["naive"])
    assert paths["robust"] == Path(config["evaluation"]["models"]["robust"])


def test_tracking_error_metrics_on_synthetic_signal():
    theta = np.array([0.0, 0.5, 1.0, 1.5])
    theta_ref = np.array([1.0, 1.0, 1.0, 1.0])

    assert compute_tracking_mae(theta, theta_ref) == 0.5
    assert compute_tracking_mse(theta, theta_ref) == 0.375


def test_overshoot_handles_positive_and_negative_setpoints():
    assert compute_overshoot([0.0, 1.1, 1.2], [1.0, 1.0, 1.0]) == 0.2
    assert compute_overshoot([0.0, -1.1, -1.25], [-1.0, -1.0, -1.0]) == 0.25
    assert compute_overshoot([0.0, 0.8, 0.95], [1.0, 1.0, 1.0]) == 0.0


def test_settling_time_requires_sustained_tolerance_window():
    theta = np.array([0.0, 0.9, 1.02, 1.01, 1.0])
    theta_ref = np.ones_like(theta)

    assert compute_settling_time(theta, theta_ref, dt=0.1, tolerance=0.05, window=3) == 0.2
    assert compute_settling_time(theta, theta_ref, dt=0.1, tolerance=0.005, window=2) is None


def test_evaluate_controller_runs_headless_with_mock_policy():
    env = MockEvaluationEnv()
    policy = MockPolicy(action=0.4)
    config = {"evaluation": {"num_episodes": 1, "max_steps": 5, "settling_window": 2}}

    result = evaluate_controller(policy, env, config, controller_name="robust")

    assert result["controller"] == "robust"
    assert result["episodes"][0]["metrics"]["mae_tracking_error"] >= 0.0
    assert result["episodes"][0]["metrics"]["mse_tracking_error"] >= 0.0
    assert result["episodes"][0]["metrics"]["max_overshoot"] >= 0.0
    assert result["episodes"][0]["metrics"]["episode_return"] == 5.0
    assert result["episodes"][0]["trajectory"]["wind_torque"][-1] == 0.3
    assert policy.predict_calls == 5
    assert policy.learn_called is False
    assert env.render_mode is None


def test_compare_controllers_writes_results_for_naive_and_robust(tmp_path):
    config_path = tmp_path / "stage3_eval.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    policies = {
        "naive": MockPolicy(action=0.1),
        "robust": MockPolicy(action=0.4),
    }

    result = compare_controllers(
        config_path,
        policy_loader=lambda path, algorithm: policies[Path(path).stem.replace("pahm_ppo_", "")],
        env_factory=lambda config, controller, episode_index: MockEvaluationEnv(),
    )

    assert set(result["controllers"]) == {"naive", "robust"}
    assert set(result["summary"]) == {"naive", "robust"}
    metrics_path = Path(result["artifacts"]["metrics_path"])
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(saved["summary"]) == {"naive", "robust"}
    assert policies["naive"].learn_called is False
    assert policies["robust"].learn_called is False


def test_compare_controllers_logs_evaluation_to_wandb_when_enabled(tmp_path):
    config = _config(tmp_path)
    config["wandb"] = {
        "enabled": True,
        "project": "pahm-rl-stage3-test",
        "mode": "disabled",
        "tags": ["stage3", "eval"],
        "log_models": False,
        "log_evaluation": True,
    }
    config_path = tmp_path / "stage3_eval_wandb.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    logged_payloads = []
    logged_artifacts = []

    class FakeRun:
        def log(self, payload, step=None):
            logged_payloads.append(payload)

        def log_artifact(self, artifact):
            logged_artifacts.append(artifact)

        def finish(self):
            logged_payloads.append({"finished": True})

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path):
            self.files.append(str(path))

    class FakeWandb:
        Artifact = FakeArtifact

        def __init__(self):
            self.run = FakeRun()
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

    fake_wandb = FakeWandb()
    policies = {
        "naive": MockPolicy(action=0.1),
        "robust": MockPolicy(action=0.4),
    }

    compare_controllers(
        config_path,
        policy_loader=lambda path, algorithm: policies[Path(path).stem.replace("pahm_ppo_", "")],
        env_factory=lambda config, controller, episode_index: MockEvaluationEnv(),
        wandb_module=fake_wandb,
    )

    assert fake_wandb.init_kwargs["mode"] == "disabled"
    assert any("eval/naive/mae_tracking_error_mean" in payload for payload in logged_payloads)
    assert any(artifact.type == "metrics" for artifact in logged_artifacts)
    assert {"finished": True} in logged_payloads
