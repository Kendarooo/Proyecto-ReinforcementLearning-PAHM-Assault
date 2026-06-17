# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import json

import numpy as np

from pahm_stage2.validate_synthetic_real import (
    PATTERN_NAMES,
    generate_synthetic_wind_signals,
    run_validation,
)


def _write_config(tmp_path):
    config = {
        "seed": 42,
        "data": {
            "source": "synthetic",
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
            "tau_w_dir": str(tmp_path / "tau_w_signals"),
            "file_prefix": "tau_w",
            "file_format": "npy",
            "manifest_path": str(tmp_path / "missing_manifest.json"),
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


def test_generate_synthetic_wind_signals_returns_expected_counts():
    signals, labels = generate_synthetic_wind_signals(
        samples_per_pattern=3,
        signal_length=64,
        seed=7,
    )

    assert len(signals) == len(PATTERN_NAMES) * 3
    assert labels.shape == (len(signals),)
    assert all(signal.shape == (64,) for signal in signals)
    np.testing.assert_array_equal(np.unique(labels), np.arange(len(PATTERN_NAMES)))


def test_run_validation_writes_synthetic_artifacts_and_skips_missing_real_manifest(tmp_path):
    config_path = _write_config(tmp_path)

    result = run_validation(str(config_path))

    assert result.metrics["ari"] > 0.8
    assert result.metrics["nmi"] > 0.8
    assert result.fit_result.selected_n_components in result.fit_result.bic_scores
    assert result.real_cluster_counts is None
    assert result.artifacts.synthetic_metrics.exists()
    assert result.artifacts.bic_curve.exists()
    assert result.artifacts.synthetic_clusters.exists()
    assert result.artifacts.real_cluster_counts is None
    assert result.artifacts.real_cluster_distribution is None

    payload = json.loads(result.artifacts.synthetic_metrics.read_text(encoding="utf-8"))
    assert payload["n_components_selected"] == result.fit_result.selected_n_components
    assert set(payload["bic_scores"]) == {"2", "3", "4", "5", "6"}
