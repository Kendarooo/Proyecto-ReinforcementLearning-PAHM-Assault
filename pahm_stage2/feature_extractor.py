# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

"""Feature extraction from variable-length PAHM wind torque signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_FEATURES = ("mean", "std", "max_abs", "skewness", "smoothness", "energy")


class FeatureExtractor:
    """Convert tau_w(t) signals into fixed-size feature vectors."""

    def __init__(self, feature_names: Iterable[str]):
        self.feature_names = list(feature_names)
        unsupported = sorted(set(self.feature_names) - set(SUPPORTED_FEATURES))
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"Unsupported feature(s): {joined}")

    def transform_signal(self, tau_w: np.ndarray) -> np.ndarray:
        """Extract configured features from one variable-length tau_w(t) signal."""
        signal = np.asarray(tau_w, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError("tau_w signal must contain at least one sample")

        values = {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "max_abs": float(np.max(np.abs(signal))),
            "skewness": self._skewness(signal),
            "smoothness": self._smoothness(signal),
            "energy": float(np.mean(np.square(signal))),
        }
        return np.array([values[name] for name in self.feature_names], dtype=np.float32)

    def transform_many(self, signals: Iterable[np.ndarray]) -> np.ndarray:
        """Extract one feature row per signal."""
        rows = [self.transform_signal(signal) for signal in signals]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    @staticmethod
    def _skewness(signal: np.ndarray) -> float:
        std = float(np.std(signal))
        if std < 1e-8:
            return 0.0
        centered = signal - float(np.mean(signal))
        return float(np.mean((centered / std) ** 3))

    @staticmethod
    def _smoothness(signal: np.ndarray) -> float:
        if signal.size < 2:
            return 0.0
        return float(np.mean(np.square(np.diff(signal))))


def extract_feature_matrix_from_manifest(
    manifest_path: str,
    feature_names: Iterable[str],
) -> tuple[list[str], np.ndarray]:
    """Load tau_w(t) files listed in a manifest and return fixed-size features."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    extractor = FeatureExtractor(feature_names)

    trajectory_ids = []
    signals = []
    for item in manifest.get("signals", []):
        trajectory_ids.append(item["trajectory_id"])
        signals.append(np.load(item["path"]))

    return trajectory_ids, extractor.transform_many(signals)

