# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import json

import numpy as np

from pahm_stage2.feature_extractor import (
    FeatureExtractor,
    extract_feature_matrix_from_manifest,
)


FEATURE_NAMES = ["mean", "std", "max_abs", "skewness", "smoothness", "energy"]


def test_output_is_fixed_size_regardless_of_trajectory_length():
    extractor = FeatureExtractor(FEATURE_NAMES)

    short_features = extractor.transform_signal(np.linspace(0.0, 1.0, 8))
    long_features = extractor.transform_signal(np.linspace(0.0, 1.0, 200))

    assert short_features.shape == (len(FEATURE_NAMES),)
    assert long_features.shape == (len(FEATURE_NAMES),)


def test_feature_names_match_config():
    extractor = FeatureExtractor(["mean", "energy", "std"])

    assert extractor.feature_names == ["mean", "energy", "std"]
    assert extractor.transform_signal(np.array([1.0, 2.0, 3.0])).shape == (3,)


def test_smoothness_is_zero_for_constant_signal():
    extractor = FeatureExtractor(FEATURE_NAMES)
    features = extractor.transform_signal(np.full(20, 0.5))

    smoothness_index = extractor.feature_names.index("smoothness")
    assert features[smoothness_index] == 0.0


def test_extract_feature_matrix_from_manifest(tmp_path):
    signal_a = tmp_path / "tau_w_a.npy"
    signal_b = tmp_path / "tau_w_b.npy"
    np.save(signal_a, np.zeros(10, dtype=np.float32))
    np.save(signal_b, np.ones(15, dtype=np.float32))
    manifest = {
        "signals": [
            {"trajectory_id": "a", "path": str(signal_a), "n_samples": 10},
            {"trajectory_id": "b", "path": str(signal_b), "n_samples": 15},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    trajectory_ids, feature_matrix = extract_feature_matrix_from_manifest(
        str(manifest_path),
        FEATURE_NAMES,
    )

    assert trajectory_ids == ["a", "b"]
    assert feature_matrix.shape == (2, len(FEATURE_NAMES))
