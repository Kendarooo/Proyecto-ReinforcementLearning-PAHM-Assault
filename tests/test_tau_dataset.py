# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import json
from pathlib import Path

import numpy as np
import pytest

from pahm_stage2.config import load_config
from pahm_stage2.tau_dataset import TauTrajectoryDataset


def _write_config(tmp_path: Path, source: str) -> Path:
    config = {
        "seed": 42,
        "data": {
            "source": source,
            "synthetic_input_dir": str(tmp_path / "synthetic"),
            "real_input_dir": str(tmp_path / "real"),
            "file_pattern": "*.npy",
        },
        "estimator": {
            "type": "dummy",
            "checkpoint_path": str(tmp_path / "estimador_wind.pth"),
            "device": "cpu",
        },
        "outputs": {
            "tau_w_dir": str(tmp_path / "tau_w"),
            "file_prefix": "tau_w",
            "file_format": "npy",
            "manifest_path": str(tmp_path / "manifest.json"),
        },
        "wandb": {
            "project": "pahm-stage2-test",
            "entity": None,
        },
    }
    config_path = tmp_path / f"{source}_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _write_trajectory(directory: Path, name: str, length: int) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = np.column_stack(
        [
            np.linspace(0.0, 1.0, length),
            np.linspace(1.0, 0.0, length),
            np.linspace(-0.5, 0.5, length),
            np.full(length, 0.25),
        ]
    ).astype(np.float32)
    np.save(directory / name, trajectory)
    return trajectory


def test_dataset_loads_all_trajectories_from_synthetic_source(tmp_path):
    _write_trajectory(tmp_path / "synthetic", "traj_a.npy", 8)
    _write_trajectory(tmp_path / "synthetic", "traj_b.npy", 11)
    _write_trajectory(tmp_path / "real", "real_only.npy", 5)

    config = load_config(str(_write_config(tmp_path, source="synthetic")))
    dataset = TauTrajectoryDataset(config)

    assert len(dataset) == 2
    assert {trajectory_id for trajectory_id, _ in dataset} == {"traj_a", "traj_b"}


def test_dataset_uses_real_input_dir_when_source_is_real(tmp_path):
    _write_trajectory(tmp_path / "synthetic", "synthetic_only.npy", 8)
    _write_trajectory(tmp_path / "real", "real_a.npy", 5)
    _write_trajectory(tmp_path / "real", "real_b.npy", 7)

    config = load_config(str(_write_config(tmp_path, source="real")))
    dataset = TauTrajectoryDataset(config)

    assert len(dataset) == 2
    assert {trajectory_id for trajectory_id, _ in dataset} == {"real_a", "real_b"}


def test_dataset_has_no_hardcoded_loader_paths(tmp_path):
    custom_synthetic = tmp_path / "custom" / "nested" / "synthetic_input"
    _write_trajectory(custom_synthetic, "custom_traj.npy", 6)

    config_path = _write_config(tmp_path, source="synthetic")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["synthetic_input_dir"] = str(custom_synthetic)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    dataset = TauTrajectoryDataset(load_config(str(config_path)))

    assert len(dataset) == 1
    assert dataset[0][0] == "custom_traj"


def test_dataset_requires_trajectory_shape_t_by_4(tmp_path):
    expected = _write_trajectory(tmp_path / "synthetic", "valid.npy", 9)
    config = load_config(str(_write_config(tmp_path, source="synthetic")))

    _, trajectory = TauTrajectoryDataset(config)[0]

    assert trajectory.shape == (9, 4)
    np.testing.assert_allclose(trajectory, expected)


def test_dataset_rejects_invalid_trajectory_shape(tmp_path):
    invalid_dir = tmp_path / "synthetic"
    invalid_dir.mkdir(parents=True)
    np.save(invalid_dir / "bad.npy", np.zeros((5, 3), dtype=np.float32))

    config = load_config(str(_write_config(tmp_path, source="synthetic")))

    with pytest.raises(ValueError, match="shape"):
        TauTrajectoryDataset(config)
