# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import json
from pathlib import Path

import numpy as np
import pytest

from pahm_stage2.train_unsupervised import train_from_manifest
from pahm_stage2.unsupervised_model import GMMWindModel


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


def _write_manifest_config(tmp_path: Path, with_manifest: bool = True) -> Path:
    tau_dir = tmp_path / "tau_w_signals"
    tau_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "tau_w_manifest.json"
    if with_manifest:
        rng = np.random.default_rng(17)
        signals = []
        for index in range(12):
            center = 0.0 if index < 6 else 0.5
            signal = rng.normal(center, 0.02, size=80).astype(np.float32)
            path = tau_dir / f"tau_w_{index}.npy"
            np.save(path, signal)
            signals.append(
                {
                    "trajectory_id": f"traj_{index}",
                    "path": str(path),
                    "n_samples": int(signal.shape[0]),
                }
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "n_trajectories": len(signals),
                    "data_source": "real",
                    "estimator_type": "stage1",
                    "signals": signals,
                }
            ),
            encoding="utf-8",
        )

    config = {
        "seed": 42,
        "data": {
            "source": "real",
            "synthetic_input_dir": str(tmp_path / "synthetic_input"),
            "real_input_dir": str(tmp_path / "real_input"),
            "file_pattern": "*.npy",
        },
        "estimator": {
            "type": "dummy",
            "checkpoint_path": str(tmp_path / "stage1" / "estimador_wind.pth"),
            "device": "cpu",
        },
        "outputs": {
            "tau_w_dir": str(tau_dir),
            "file_prefix": "tau_w",
            "file_format": "npy",
            "manifest_path": str(manifest_path),
        },
        "unsupervised": {
            "n_components_range": [1, 3],
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


def test_train_from_manifest_saves_checkpoint_and_logs_metrics(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr("pahm_stage2.train_unsupervised.wandb", fake_wandb)
    config_path = _write_manifest_config(tmp_path, with_manifest=True)

    result = train_from_manifest(str(config_path))

    assert result.checkpoint_path.exists()
    assert len(result.trajectory_ids) == 12
    assert result.fit_result.selected_n_components in result.fit_result.bic_scores
    loaded = GMMWindModel.load(str(result.checkpoint_path))
    assert loaded.selected_n_components == result.fit_result.selected_n_components
    assert fake_wandb.init_kwargs["job_type"] == "train_unsupervised"
    assert fake_wandb.init_kwargs["mode"] == "disabled"
    assert fake_wandb.run.logged[-1]["unsupervised/n_trajectories"] == 12
    assert fake_wandb.run.finished


def test_train_from_manifest_requires_manifest(tmp_path):
    config_path = _write_manifest_config(tmp_path, with_manifest=False)

    with pytest.raises(FileNotFoundError, match="Run generate_tau_w.py"):
        train_from_manifest(str(config_path))
