import copy
import json

from gym_wrapper.learned_pahm_ode import _load_wrapper_config

from pahm_stage3.wandb_logger import (
    NullWandbRun,
    build_evaluation_metrics_payload,
    build_wandb_config_payload,
    init_wandb_run,
    log_file_artifact,
    log_model_artifact,
)


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path):
        self.files.append(str(path))


class FakeRun:
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeWandbModule:
    Artifact = FakeArtifact

    def __init__(self):
        self.init_kwargs = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def _config():
    config = copy.deepcopy(_load_wrapper_config())
    config["rl_training"]["mode"] = "robust"
    config["rl_training"]["total_timesteps"] = 123
    config["rl_training"]["learning_rate"] = 0.001
    config["rl_training"]["gamma"] = 0.95
    config["rl_training"]["n_steps"] = 16
    config["rl_training"]["batch_size"] = 8
    config["rl_training"]["seed"] = 7
    config["wandb"] = {
        "enabled": True,
        "project": "pahm-rl-stage3-test",
        "entity": None,
        "mode": "disabled",
        "tags": ["stage3", "test"],
        "log_models": True,
        "log_evaluation": True,
    }
    return config


def test_wandb_config_payload_groups_hyperparameters_and_environment():
    payload = build_wandb_config_payload(_config())

    assert payload["training"]["algorithm"] == "PPO"
    assert payload["training"]["learning_rate"] == 0.001
    assert payload["training"]["gamma"] == 0.95
    assert payload["training"]["n_steps"] == 16
    assert payload["training"]["batch_size"] == 8
    assert payload["training"]["total_timesteps"] == 123
    assert payload["training"]["seed"] == 7
    assert "theta_ref" in payload["control"]
    assert "tracking_error_weight" in payload["reward"]
    assert "enabled" in payload["wind"]
    assert payload["environment"]["reset_randomize"] is True


def test_init_wandb_run_returns_null_run_when_disabled():
    config = _config()
    config["wandb"]["enabled"] = False
    fake_wandb = FakeWandbModule()

    run = init_wandb_run(config, run_name="disabled-run", wandb_module=fake_wandb)

    assert isinstance(run, NullWandbRun)
    assert fake_wandb.init_kwargs is None
    run.log({"train/reward": 1.0})
    run.finish()


def test_init_wandb_run_uses_disabled_mode_without_network():
    fake_wandb = FakeWandbModule()

    run = init_wandb_run(_config(), run_name="robust-run", wandb_module=fake_wandb)

    assert run is fake_wandb.run
    assert fake_wandb.init_kwargs["project"] == "pahm-rl-stage3-test"
    assert fake_wandb.init_kwargs["mode"] == "disabled"
    assert fake_wandb.init_kwargs["name"] == "robust-run"
    assert "training" in fake_wandb.init_kwargs["config"]


def test_model_and_metric_artifacts_are_registered_when_enabled(tmp_path):
    fake_wandb = FakeWandbModule()
    run = fake_wandb.run
    model_path = tmp_path / "pahm_ppo_robust.zip"
    metrics_path = tmp_path / "controller_metrics.json"
    model_path.write_text("model", encoding="utf-8")
    metrics_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    log_model_artifact(run, model_path, "pahm_ppo_robust", _config(), wandb_module=fake_wandb)
    log_file_artifact(
        run,
        metrics_path,
        "stage3-controller-metrics",
        artifact_type="metrics",
        enabled=True,
        wandb_module=fake_wandb,
    )

    assert [artifact.name for artifact in run.artifacts] == [
        "pahm_ppo_robust",
        "stage3-controller-metrics",
    ]
    assert run.artifacts[0].type == "model"
    assert run.artifacts[1].type == "metrics"


def test_evaluation_metrics_payload_flattens_controller_comparison():
    comparison = {
        "summary": {
            "naive": {
                "mae_tracking_error": {"mean": 0.3, "std": 0.1},
                "episode_return": {"mean": 10.0, "std": 2.0},
            },
            "robust": {
                "mae_tracking_error": {"mean": 0.2, "std": 0.05},
                "episode_return": {"mean": 15.0, "std": 1.0},
            },
        }
    }

    payload = build_evaluation_metrics_payload(comparison)

    assert payload["eval/naive/mae_tracking_error_mean"] == 0.3
    assert payload["eval/robust/mae_tracking_error_mean"] == 0.2
    assert payload["eval/comparison/mae_tracking_error_delta_robust_minus_naive"] == -0.1
    assert payload["eval/comparison/episode_return_delta_robust_minus_naive"] == 5.0
