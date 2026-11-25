"""Performance metrics and group-fairness gaps.

The favorable decision is approval, defined as a predicted non-default
(``y_hat = 0``). Demographic parity is the gap in approval rates.
Predictive metrics and equalized odds treat default (``y = 1``) as the
positive class: true positive rate is the share of actual defaults that are
flagged, and false positive rate is the share of non-defaults that are flagged.
Equalized-odds difference is the larger of the cross-group TPR gap and the
cross-group FPR gap. Groups that do not contain the relevant label are omitted
from that rate. When every group contains both classes, these gaps match
fairlearn's between-groups differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config
from src.utils import quantile_ci, slug


def performance_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Threshold-free and thresholded performance for the default class."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = np.asarray(pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def _codes(sensitive: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sensitive).astype(str)
    uniques, codes = np.unique(labels, return_inverse=True)
    return uniques, codes.astype(int)


def fairness_gaps(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray | pd.Series) -> dict[str, float]:
    """Approval demographic parity and default-class equalized odds."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    uniques, codes = _codes(sensitive)
    n_groups = len(uniques)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    approval = (y_pred == 0).astype(float)
    approval_sum = np.bincount(codes, weights=approval, minlength=n_groups).astype(float)
    present = counts > 0
    rates = np.full(n_groups, np.nan)
    rates[present] = approval_sum[present] / counts[present]
    if int(present.sum()) < 2:
        dp_difference = float("nan")
        dp_ratio = float("nan")
    else:
        dp_difference = float(np.nanmax(rates[present]) - np.nanmin(rates[present]))
        maximum = float(np.nanmax(rates[present]))
        minimum = float(np.nanmin(rates[present]))
        dp_ratio = float(minimum / maximum) if maximum > 0 else float("nan")

    positive = y_true == 1
    negative = y_true == 0
    positive_count = np.bincount(codes[positive], minlength=n_groups).astype(float) if positive.any() else np.zeros(n_groups)
    negative_count = np.bincount(codes[negative], minlength=n_groups).astype(float) if negative.any() else np.zeros(n_groups)
    true_positive = ((y_true == 1) & (y_pred == 1)).astype(float)
    false_positive = ((y_true == 0) & (y_pred == 1)).astype(float)
    tpr_sum = np.bincount(codes, weights=true_positive, minlength=n_groups).astype(float)
    fpr_sum = np.bincount(codes, weights=false_positive, minlength=n_groups).astype(float)
    tpr = np.full(n_groups, np.nan)
    fpr = np.full(n_groups, np.nan)
    tpr_ok = positive_count > 0
    fpr_ok = negative_count > 0
    tpr[tpr_ok] = tpr_sum[tpr_ok] / positive_count[tpr_ok]
    fpr[fpr_ok] = fpr_sum[fpr_ok] / negative_count[fpr_ok]
    tpr_gap = float(np.nanmax(tpr[tpr_ok]) - np.nanmin(tpr[tpr_ok])) if int(tpr_ok.sum()) >= 2 else float("nan")
    fpr_gap = float(np.nanmax(fpr[fpr_ok]) - np.nanmin(fpr[fpr_ok])) if int(fpr_ok.sum()) >= 2 else float("nan")
    if np.isnan(tpr_gap) and np.isnan(fpr_gap):
        eo = float("nan")
    else:
        eo = float(np.nanmax([tpr_gap, fpr_gap]))
    return {
        "dp_difference": dp_difference,
        "dp_ratio": dp_ratio,
        "eo_difference": eo,
        "approval_rates": {str(group): float(rates[index]) for index, group in enumerate(uniques)},
        "tpr": {str(group): float(tpr[index]) for index, group in enumerate(uniques)},
        "fpr": {str(group): float(fpr[index]) for index, group in enumerate(uniques)},
    }


def fairlearn_gaps(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray | pd.Series) -> dict[str, float]:
    """fairlearn's between-groups gaps, using approval for parity and default for equalized odds."""
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    approval = 1 - y_pred
    return {
        "dp_difference": float(demographic_parity_difference(y_true, approval, sensitive_features=sensitive)),
        "dp_ratio": float(demographic_parity_ratio(y_true, approval, sensitive_features=sensitive)),
        "eo_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)),
    }


def bootstrap_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.Series,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    """Point gaps plus percentile intervals from resampling the evaluation rows."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive_array = np.asarray(sensitive).astype(str)
    point = fairness_gaps(y_true, y_pred, sensitive_array)
    generator = np.random.default_rng(int(seed))
    collected = {key: [] for key in ("dp_difference", "dp_ratio", "eo_difference")}
    group_approval: dict[str, list[float]] = {group: [] for group in point["approval_rates"]}
    group_tpr: dict[str, list[float]] = {group: [] for group in point["tpr"]}
    group_fpr: dict[str, list[float]] = {group: [] for group in point["fpr"]}
    n_rows = len(y_true)
    for _ in range(int(n_bootstrap)):
        index = generator.integers(0, n_rows, n_rows)
        draw = fairness_gaps(y_true[index], y_pred[index], sensitive_array[index])
        for key in collected:
            collected[key].append(draw[key])
        for group in group_approval:
            group_approval[group].append(draw["approval_rates"].get(group, float("nan")))
            group_tpr[group].append(draw["tpr"].get(group, float("nan")))
            group_fpr[group].append(draw["fpr"].get(group, float("nan")))
    intervals = {}
    for key, values in collected.items():
        low, high = quantile_ci(values, ci_level)
        intervals[f"{key}_ci_low"] = low
        intervals[f"{key}_ci_high"] = high
    return {
        "point": point,
        "intervals": intervals,
        "group_approval": group_approval,
        "group_tpr": group_tpr,
        "group_fpr": group_fpr,
    }


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def group_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    sensitive: np.ndarray | pd.Series,
    bootstrap: dict[str, Any] | None,
    ci_level: float,
) -> pd.DataFrame:
    """Per-group rates. Interval columns are filled when bootstrap samples are supplied."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = np.asarray(pred).astype(int)
    sensitive_array = np.asarray(sensitive).astype(str)
    rows = []
    for group in list(dict.fromkeys(sensitive_array.tolist())):
        mask = sensitive_array == group
        yt = y_true[mask]
        yp = pred[mask]
        pr = proba[mask]
        positives = yt == 1
        negatives = yt == 0
        tpr = float(yp[positives].mean()) if positives.any() else float("nan")
        fpr = float(yp[negatives].mean()) if negatives.any() else float("nan")
        predicted_positive = yp == 1
        precision = float(yt[predicted_positive].mean()) if predicted_positive.any() else 0.0
        row: dict[str, Any] = {
            "group": group,
            "n": int(mask.sum()),
            "base_rate": float(yt.mean()) if mask.any() else float("nan"),
            "approval_rate": float((yp == 0).mean()) if mask.any() else float("nan"),
            "tpr": tpr,
            "fpr": fpr,
            "fnr": float(1.0 - tpr) if np.isfinite(tpr) else float("nan"),
            "precision": precision,
            "accuracy": float((yp == yt).mean()) if mask.any() else float("nan"),
            "auc": _safe_auc(yt, pr),
        }
        if bootstrap is not None:
            low, high = quantile_ci(bootstrap["group_approval"][group], ci_level)
            row["approval_ci_low"] = low
            row["approval_ci_high"] = high
            low, high = quantile_ci(bootstrap["group_tpr"][group], ci_level)
            row["tpr_ci_low"] = low
            row["tpr_ci_high"] = high
            low, high = quantile_ci(bootstrap["group_fpr"][group], ci_level)
            row["fpr_ci_low"] = low
            row["fpr_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def prefixed_gaps(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray | pd.Series, prefix: str) -> dict[str, float]:
    """Fairness gaps with a column prefix such as ``sex`` or ``age``."""
    gaps = fairness_gaps(y_true, y_pred, sensitive)
    return {
        f"dp_difference_{prefix}": gaps["dp_difference"],
        f"dp_ratio_{prefix}": gaps["dp_ratio"],
        f"eo_difference_{prefix}": gaps["eo_difference"],
    }


def per_group_metric_columns(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    frame: pd.DataFrame,
    attributes: list[str],
) -> dict[str, float]:
    """Wide per-group AUC, accuracy, and approval columns."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = np.asarray(pred).astype(int)
    columns: dict[str, float] = {}
    for attribute in attributes:
        values = frame[attribute].astype(str).to_numpy()
        for group in sorted(set(values.tolist())):
            mask = values == group
            key = slug(group)
            yt = y_true[mask]
            yp = pred[mask]
            columns[f"n__{attribute}__{key}"] = float(mask.sum())
            columns[f"approval__{attribute}__{key}"] = float((yp == 0).mean()) if mask.any() else float("nan")
            columns[f"accuracy__{attribute}__{key}"] = float((yp == yt).mean()) if mask.any() else float("nan")
            columns[f"auc__{attribute}__{key}"] = _safe_auc(yt, proba[mask]) if mask.any() else float("nan")
    return columns


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    frame: pd.DataFrame,
    config: Config,
    include_reweigh_attributes: bool = False,
) -> dict[str, float]:
    """Performance, sex gaps, age-group gaps, and per-group AUC and accuracy."""
    del config
    result = performance_metrics(y_true, proba, pred)
    result.update(prefixed_gaps(y_true, pred, frame["sex_label"], "sex"))
    result.update(prefixed_gaps(y_true, pred, frame["age_group"], "age"))
    if include_reweigh_attributes:
        result.update(prefixed_gaps(y_true, pred, frame["age_young_label"], "age_young"))
        result.update(prefixed_gaps(y_true, pred, frame["sex_age_young"], "intersection"))
    result.update(per_group_metric_columns(y_true, proba, pred, frame, ["sex_label", "age_group"]))
    return result
