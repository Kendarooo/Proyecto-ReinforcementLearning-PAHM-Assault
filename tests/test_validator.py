# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import numpy as np

from pahm_stage2.feature_extractor import FeatureExtractor
from pahm_stage2.unsupervised_model import GMMWindModel
from pahm_stage2.validator import evaluate_clustering


FEATURE_NAMES = ["mean", "std", "max_abs", "skewness", "smoothness", "energy"]


def _synthetic_wind_signals(seed: int = 13):
    rng = np.random.default_rng(seed)
    signals = []
    labels = []
    for label in range(4):
        for _ in range(16):
            length = int(rng.integers(80, 140))
            if label == 0:
                signal = rng.normal(0.0, 0.01, length)
            elif label == 1:
                signal = rng.normal(0.0, 0.01, length)
                start = int(rng.integers(10, length - 20))
                signal[start : start + 10] += np.hanning(10) * 0.8
            elif label == 2:
                signal = np.full(length, 0.5) + rng.normal(0.0, 0.01, length)
            else:
                signal = rng.normal(0.0, 0.3, length)
            signals.append(signal.astype(np.float32))
            labels.append(label)
    return signals, np.array(labels)


def test_ari_above_threshold_with_known_patterns():
    signals, labels = _synthetic_wind_signals()
    features = FeatureExtractor(FEATURE_NAMES).transform_many(signals)
    model = GMMWindModel((4, 4), random_state=42)
    model.fit(features)

    metrics = evaluate_clustering(labels, model.predict(features))

    assert metrics["ari"] > 0.8


def test_nmi_above_threshold_with_known_patterns():
    signals, labels = _synthetic_wind_signals()
    features = FeatureExtractor(FEATURE_NAMES).transform_many(signals)
    model = GMMWindModel((4, 4), random_state=42)
    model.fit(features)

    metrics = evaluate_clustering(labels, model.predict(features))

    assert metrics["nmi"] > 0.8

