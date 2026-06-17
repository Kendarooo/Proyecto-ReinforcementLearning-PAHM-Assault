# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""CLI pipeline to generate estimated PAHM wind signals tau_w(t)."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from pahm_stage2.artifacts import save_tau_signal, write_manifest
from pahm_stage2.config import load_config
from pahm_stage2.estimator_interface import build_estimator
from pahm_stage2.tau_dataset import TauTrajectoryDataset


class _NullRun:
    def log(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def finish(self) -> None:
        return None


class _NullWandb:
    def init(self, **kwargs: Any) -> _NullRun:
        return _NullRun()


try:
    import wandb
except ImportError:
    wandb = _NullWandb()
else:
    if not hasattr(wandb, "init"):
        wandb = _NullWandb()


def _validate_tau_signal(trajectory_id: str, trajectory: np.ndarray, tau_w: np.ndarray) -> None:
    if tau_w.ndim != 1 or tau_w.shape[0] != trajectory.shape[0]:
        raise ValueError(
            f"Estimator output for {trajectory_id} must have shape (T,), got {tau_w.shape}"
        )


def run_pipeline(config_path: str) -> dict:
    """Generate tau_w(t) files for every configured trajectory."""
    config = load_config(config_path)
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
        job_type="generate_tau_w",
        mode=config["wandb"].get("mode", "online"),
    )

    dataset = TauTrajectoryDataset(config)
    estimator = build_estimator(config)
    manifest = {
        "n_trajectories": 0,
        "data_source": config["data"]["source"],
        "estimator_type": config["estimator"]["type"],
        "signals": [],
    }

    try:
        for trajectory_id, trajectory in dataset:
            tau_w = estimator.predict(trajectory)
            tau_w = np.asarray(tau_w, dtype=np.float32)
            _validate_tau_signal(trajectory_id, trajectory, tau_w)
            output_path = save_tau_signal(
                tau_w=tau_w,
                trajectory_id=trajectory_id,
                outputs_config=config["outputs"],
            )
            manifest["signals"].append(
                {
                    "trajectory_id": trajectory_id,
                    "path": str(output_path),
                    "n_samples": int(tau_w.shape[0]),
                }
            )

        manifest["n_trajectories"] = len(manifest["signals"])
        write_manifest(manifest, config["outputs"]["manifest_path"])
        run.log(
            {
                "n_trajectories": manifest["n_trajectories"],
                "estimator_type": manifest["estimator_type"],
                "data_source": manifest["data_source"],
            }
        )
        return manifest
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_config.json")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
