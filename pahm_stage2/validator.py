# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos / Codex

"""Validation metrics for synthetic-to-real Stage 2 perturbation analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_clustering(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float]:
    """Compute ARI/NMI for synthetic data with known wind-pattern labels."""
    true = np.asarray(true_labels)
    predicted = np.asarray(predicted_labels)
    if true.shape[0] != predicted.shape[0]:
        raise ValueError("true_labels and predicted_labels must have the same length")
    return {
        "ari": float(adjusted_rand_score(true, predicted)),
        "nmi": float(normalized_mutual_info_score(true, predicted)),
    }


def explore_unlabeled(predicted_labels: np.ndarray) -> dict[str, int]:
    """Summarize cluster counts for real unlabeled perturbation signals."""
    labels = np.asarray(predicted_labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}

