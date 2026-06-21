# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Synthetic-to-real validation pipeline for Stage 2 wind representations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from pahm_stage2.config import load_config
from pahm_stage2.feature_extractor import FeatureExtractor, extract_feature_matrix_from_manifest
from pahm_stage2.unsupervised_model import GMMFitResult, GMMWindModel
from pahm_stage2.validator import evaluate_clustering, explore_unlabeled


PATTERN_NAMES = ("calm", "gust", "bias", "turbulent")


@dataclass(frozen=True)
class ValidationArtifacts:
    """Paths written by the synthetic-to-real validation pipeline."""

    synthetic_metrics: Path
    bic_curve: Path
    synthetic_clusters: Path
    real_cluster_counts: Path | None = None
    real_cluster_distribution: Path | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Summary returned by the validation pipeline."""

    metrics: dict[str, float]
    fit_result: GMMFitResult
    artifacts: ValidationArtifacts
    real_cluster_counts: dict[str, int] | None


class _NullRun:
    def finish(self) -> None:
        return None


class _NullWandb:
    def init(self, **kwargs: Any) -> _NullRun:
        return _NullRun()

    def log(self, payload: dict[str, Any]) -> None:
        self.payload = payload


try:
    import wandb
except ImportError:
    wandb = _NullWandb()
else:
    if not hasattr(wandb, "init") or not hasattr(wandb, "log"):
        wandb = _NullWandb()


def generate_synthetic_wind_signals(
    samples_per_pattern: int,
    signal_length: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create labeled synthetic wind patterns for structural recovery tests."""
    if samples_per_pattern < 1:
        raise ValueError("samples_per_pattern must be >= 1")
    if signal_length < 32:
        raise ValueError("signal_length must be >= 32")

    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    for label, pattern_name in enumerate(PATTERN_NAMES):
        for _ in range(samples_per_pattern):
            if pattern_name == "calm":
                signal = rng.normal(0.0, 0.01, signal_length)
            elif pattern_name == "gust":
                signal = rng.normal(0.0, 0.01, signal_length)
                width = int(rng.integers(signal_length // 12, signal_length // 5))
                start = int(rng.integers(4, signal_length - width - 4))
                amplitude = float(rng.uniform(0.55, 0.95))
                signal[start : start + width] += np.hanning(width) * amplitude
            elif pattern_name == "bias":
                level = float(rng.uniform(0.35, 0.65))
                drift = float(rng.uniform(-0.05, 0.05)) * time
                signal = level + drift + rng.normal(0.0, 0.015, signal_length)
            else:
                frequency = float(rng.uniform(8.0, 16.0))
                oscillation = 0.12 * np.sin(2.0 * np.pi * frequency * time)
                signal = oscillation + rng.normal(0.0, 0.28, signal_length)

            signals.append(np.asarray(signal, dtype=np.float32))
            labels.append(label)

    return signals, np.asarray(labels, dtype=np.int64)


def run_validation(config_path: str) -> ValidationResult:
    """Run synthetic validation and optional real-data cluster exploration."""
    config = load_config(config_path)
    output_dir = Path(config["validation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            config=config,
            job_type="validate_synthetic_real",
            mode=config["wandb"].get("mode", "online"),
        )
    except Exception:
        fallback = _NullWandb()
        run = fallback.init()
        tracker = fallback
    else:
        tracker = wandb

    try:
        result = _run_synthetic_block(config, output_dir)
        real_counts, real_artifact = _run_real_block(config, result, output_dir)
        _log_wandb(tracker, result.metrics, result.fit_result)
        artifacts = ValidationArtifacts(
            synthetic_metrics=output_dir / "synthetic_metrics.json",
            bic_curve=output_dir / "bic_curve.png",
            synthetic_clusters=output_dir / "synthetic_clusters.png",
            real_cluster_counts=real_artifact[0],
            real_cluster_distribution=real_artifact[1],
        )
        return ValidationResult(
            metrics=result.metrics,
            fit_result=result.fit_result,
            artifacts=artifacts,
            real_cluster_counts=real_counts,
        )
    finally:
        run.finish()


@dataclass(frozen=True)
class _SyntheticBlockResult:
    metrics: dict[str, float]
    fit_result: GMMFitResult
    model: GMMWindModel


def _run_synthetic_block(config: dict[str, Any], output_dir: Path) -> _SyntheticBlockResult:
    validation_config = config["validation"]
    unsupervised_config = config["unsupervised"]
    signals, labels = generate_synthetic_wind_signals(
        samples_per_pattern=int(validation_config["synthetic_samples_per_pattern"]),
        signal_length=int(validation_config["signal_length"]),
        seed=int(config.get("seed", 0)),
    )
    features = FeatureExtractor(unsupervised_config["features"]).transform_many(signals)
    n_components_range = tuple(unsupervised_config["n_components_range"])
    model = GMMWindModel(
        n_components_range=n_components_range,
        covariance_type=unsupervised_config.get("covariance_type", "full"),
        random_state=int(config.get("seed", 0)),
    )
    fit_result = model.fit(features)
    predicted_labels = model.predict(features)
    metrics = evaluate_clustering(labels, predicted_labels)

    _write_json(
        output_dir / "synthetic_metrics.json",
        {
            **metrics,
            "n_components_selected": fit_result.selected_n_components,
            "bic_scores": fit_result.bic_scores,
            "pattern_names": PATTERN_NAMES,
        },
    )
    _plot_bic_curve(fit_result.bic_scores, output_dir / "bic_curve.png")
    _plot_clusters(
        features=features,
        labels=predicted_labels,
        output_path=output_dir / "synthetic_clusters.png",
        title="Synthetic wind clusters",
    )
    model.save(unsupervised_config["checkpoint_path"])
    return _SyntheticBlockResult(metrics=metrics, fit_result=fit_result, model=model)


def _run_real_block(
    config: dict[str, Any],
    synthetic_result: _SyntheticBlockResult,
    output_dir: Path,
) -> tuple[dict[str, int] | None, tuple[Path | None, Path | None]]:
    manifest_path = Path(config["outputs"]["manifest_path"])
    if not manifest_path.exists():
        print("Real tau_w signals not yet available - skipping real analysis.")
        return None, (None, None)

    _, real_features = extract_feature_matrix_from_manifest(
        str(manifest_path),
        config["unsupervised"]["features"],
    )
    predicted_labels = synthetic_result.model.predict(real_features)
    counts = explore_unlabeled(predicted_labels)
    counts_path = output_dir / "real_cluster_counts.json"
    distribution_path = output_dir / "real_cluster_distribution.png"
    _write_json(counts_path, counts)
    _plot_cluster_counts(counts, distribution_path)
    return counts, (counts_path, distribution_path)


def _log_wandb(
    tracker: Any,
    metrics: dict[str, float],
    fit_result: GMMFitResult,
) -> None:
    tracker.log(
        {
            "synthetic/ari": metrics["ari"],
            "synthetic/nmi": metrics["nmi"],
            "synthetic/n_components_selected": fit_result.selected_n_components,
            **{f"bic/n{k}": v for k, v in fit_result.bic_scores.items()},
        }
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _plot_bic_curve(bic_scores: dict[int, float], output_path: Path) -> None:
    n_components = sorted(bic_scores)
    values = [bic_scores[n] for n in n_components]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_components, values, marker="o")
    ax.set_xlabel("GMM components")
    ax.set_ylabel("BIC")
    ax.set_title("BIC model selection")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    projection = PCA(n_components=2, random_state=0).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=24)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_cluster_counts(counts: dict[str, int], output_path: Path) -> None:
    labels = sorted(counts, key=int)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Trajectories")
    ax.set_title("Real tau_w cluster distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_config.json")
    args = parser.parse_args()
    run_validation(args.config)


if __name__ == "__main__":
    main()
