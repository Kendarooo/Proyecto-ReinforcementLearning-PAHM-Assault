# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Train the production Stage 2 GMM from estimated real tau_w(t) signals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pahm_stage2.config import load_config
from pahm_stage2.feature_extractor import extract_feature_matrix_from_manifest
from pahm_stage2.unsupervised_model import GMMFitResult, GMMWindModel


@dataclass(frozen=True)
class TrainUnsupervisedResult:
    """Summary returned after fitting the production unsupervised model."""

    trajectory_ids: list[str]
    fit_result: GMMFitResult
    checkpoint_path: Path


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


def train_from_manifest(config_path: str) -> TrainUnsupervisedResult:
    """Fit the GMM using tau_w(t) files listed in the configured manifest."""
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run generate_tau_w.py before "
            "training the production unsupervised model."
        )

    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
        job_type="train_unsupervised",
        mode=config["wandb"].get("mode", "online"),
    )

    try:
        unsupervised_config = config["unsupervised"]
        trajectory_ids, feature_matrix = extract_feature_matrix_from_manifest(
            str(manifest_path),
            unsupervised_config["features"],
        )
        model = GMMWindModel(
            n_components_range=tuple(unsupervised_config["n_components_range"]),
            covariance_type=unsupervised_config.get("covariance_type", "full"),
            random_state=int(config.get("seed", 0)),
        )
        fit_result = model.fit(feature_matrix)
        checkpoint_path = model.save(unsupervised_config["checkpoint_path"])
        run.log(
            {
                "unsupervised/n_trajectories": len(trajectory_ids),
                "unsupervised/n_features": feature_matrix.shape[1],
                "unsupervised/n_components_selected": fit_result.selected_n_components,
                **{f"unsupervised/bic/n{k}": v for k, v in fit_result.bic_scores.items()},
            }
        )
        return TrainUnsupervisedResult(
            trajectory_ids=trajectory_ids,
            fit_result=fit_result,
            checkpoint_path=checkpoint_path,
        )
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_config.json")
    args = parser.parse_args()
    train_from_manifest(args.config)


if __name__ == "__main__":
    main()
