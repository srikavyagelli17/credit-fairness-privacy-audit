"""Validation-set decision thresholds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import Config


def select_threshold(y_true: np.ndarray, proba: np.ndarray, config: Config) -> float:
    """Choose the decision threshold on a validation set.

    ``f1`` maximizes F1 for the default class. ``fixed`` returns the configured
    constant. Ties in F1 keep the first threshold returned by the precision-recall
    curve, which is deterministic.
    """
    method = str(config.threshold.method)
    fixed = float(config.threshold.fixed_value)
    if method == "fixed":
        return fixed
    if method != "f1":
        raise ValueError(f"Unknown threshold method {method}")
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return fixed
    eps = float(config.numerical.eps)
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], eps, None)
    return float(thresholds[int(np.nanargmax(f1))])


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Predict default when the probability is at least the threshold."""
    return (np.asarray(proba, dtype=float) >= float(threshold)).astype(int)
