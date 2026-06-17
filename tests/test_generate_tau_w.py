# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pahm_stage2.estimator_interface import DummyWindEstimator
from pahm_stage2.generate_tau_w import run_pipeline


def _write_config(tmp_path: Path, n_trajectories: int = 3) -> Path:
    input_dir = tmp_path / "synthetic_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_trajectories):
        length = 5 + index
        trajectory = np.column_stack(
            [
                np.linspace(0.0, 1.0, length),
                np.linspace(1.0, 0.0, length),
                np.linspace(-0.1, 0.1, length),
                np.full(length, 0.3 + 0.1 * index),
            ]
        ).astype(np.float32)
        np.save(input_dir / f"traj_{index}.npy", trajectory)

    config = {
        "seed": 123,
        "data": {
            "source": "synthetic",
            "synthetic_input_dir": str(input_dir),
            "real_input_dir": str(tmp_path / "real_input"),
            "file_pattern": "*.npy",
        },
        "estimator": {
            "type": "dummy",
            "checkpoint_path": str(tmp_path / "estimador_wind.pth"),
            "device": "cpu",
        },
        "outputs": {
            "tau_w_dir": str(tmp_path / "tau_w_signals"),
            "file_prefix": "tau_w",
            "file_format": "npy",
            "manifest_path": str(tmp_path / "tau_w_manifest.json"),
        },
        "unsupervised": {
            "n_components_range": [2, 6],
            "covariance_type": "full",
            "features": ["mean", "std", "max_abs", "skewness", "smoothness", "energy"],
            "checkpoint_path": str(tmp_path / "gmm_wind_model.pkl"),
        },
        "validation": {
            "output_dir": str(tmp_path / "validation"),
            "synthetic_samples_per_pattern": 12,
            "signal_length": 96,
        },
        "wandb": {
            "project": "pahm-stage2-test",
            "entity": None,
            "mode": "disabled",
        },
    }
    config_path = tmp_path / "stage2_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.run = _FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_dummy_wind_estimator_predict_returns_zeros():
    trajectory = np.ones((7, 4), dtype=np.float32)

    tau_w = DummyWindEstimator().predict(trajectory)

    assert tau_w.shape == (7,)
    np.testing.assert_array_equal(tau_w, np.zeros(7))


def test_pipeline_generates_one_tau_file_per_trajectory_and_manifest(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr("pahm_stage2.generate_tau_w.wandb", fake_wandb)
    config_path = _write_config(tmp_path, n_trajectories=3)

    manifest = run_pipeline(str(config_path))

    output_paths = [Path(item["path"]) for item in manifest["signals"]]
    assert len(output_paths) == 3
    assert all(path.exists() for path in output_paths)
    for item, path in zip(manifest["signals"], output_paths):
        tau_w = np.load(path)
        assert tau_w.shape == (item["n_samples"],)

    manifest_path = tmp_path / "tau_w_manifest.json"
    assert manifest_path.exists()
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["n_trajectories"] == 3
    assert saved_manifest["data_source"] == "synthetic"
    assert saved_manifest["estimator_type"] == "dummy"
    assert len(saved_manifest["signals"]) == 3


def test_pipeline_logs_basic_wandb_metrics(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr("pahm_stage2.generate_tau_w.wandb", fake_wandb)
    config_path = _write_config(tmp_path, n_trajectories=2)

    run_pipeline(str(config_path))

    assert fake_wandb.init_kwargs["project"] == "pahm-stage2-test"
    assert fake_wandb.init_kwargs["mode"] == "disabled"
    assert fake_wandb.run.logged[-1] == {
        "n_trajectories": 2,
        "estimator_type": "dummy",
        "data_source": "synthetic",
    }
    assert fake_wandb.run.finished


def test_pipeline_runs_from_terminal_with_config(tmp_path):
    config_path = _write_config(tmp_path, n_trajectories=2)
    env = {**os.environ, "WANDB_MODE": "disabled"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pahm_stage2.generate_tau_w",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "tau_w_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["n_trajectories"] == 2
