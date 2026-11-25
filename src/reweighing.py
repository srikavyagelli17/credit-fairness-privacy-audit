"""Kamiran and Calders (2012) reweighing weights and the mitigation experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def reweighing_weights(y: np.ndarray | pd.Series, sensitive: np.ndarray | pd.Series) -> np.ndarray:
    """Return ``w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y)`` for each training row.

    Probabilities are the training-set frequencies. The weights sum to the number
    of rows, and the weighted positive rate is the same in every group that
    contains both labels.
    """
    labels = np.asarray(y).astype(int)
    groups = np.asarray(sensitive).astype(str)
    if len(labels) != len(groups):
        raise ValueError("y and sensitive must have the same length")
    n = len(labels)
    if n == 0:
        return np.array([], dtype=float)
    frame = pd.DataFrame({"y": labels, "a": groups})
    counts_a = frame.groupby("a", sort=False).size()
    counts_y = frame.groupby("y", sort=False).size()
    counts_ay = frame.groupby(["a", "y"], sort=False).size()
    weights = np.empty(n, dtype=float)
    for index, (label, group) in enumerate(zip(labels, groups)):
        joint = int(counts_ay.loc[(group, int(label))])
        if joint == 0:
            raise ValueError("Observed (group, label) cell has count 0")
        weights[index] = (float(counts_a.loc[group]) * float(counts_y.loc[int(label)])) / (float(n) * float(joint))
    return weights


def weighted_positive_rate(y: np.ndarray | pd.Series, sensitive: np.ndarray | pd.Series, weights: np.ndarray) -> dict[str, float]:
    """Weighted ``P(Y=1 | A=a)`` for each group."""
    labels = np.asarray(y).astype(int)
    groups = np.asarray(sensitive).astype(str)
    weights = np.asarray(weights, dtype=float)
    rates: dict[str, float] = {}
    for group in pd.unique(groups):
        mask = groups == group
        denom = float(weights[mask].sum())
        if denom == 0:
            rates[str(group)] = float("nan")
        else:
            rates[str(group)] = float(weights[mask & (labels == 1)].sum() / denom)
    return rates


def resample_by_weights(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a multiset of size n with probability proportional to ``weights``.

    diffprivlib's logistic regression ignores ``sample_weight``, so the privacy
    sweep uses this resampling for both the private model and its non-private
    control. Phase 4 passes the weights to ``fit`` directly.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value")
    probabilities = weights / weights.sum()
    generator = np.random.default_rng(int(seed))
    index = generator.choice(len(target), size=len(target), replace=True, p=probabilities)
    return np.asarray(features)[index], np.asarray(target)[index]


_EO_BY_ATTRIBUTE = {
    "sex": "eo_difference_sex",
    "age_young": "eo_difference_age_young",
    "intersection": "eo_difference_intersection",
}
_DP_BY_ATTRIBUTE = {
    "sex": "dp_difference_sex",
    "age_young": "dp_difference_age_young",
    "intersection": "dp_difference_intersection",
}


def _score_model(model: Any, prepared: Any, config: Any) -> dict[str, float]:
    from src.metrics import evaluate_predictions
    from src.models import positive_proba
    from src.thresholds import apply_threshold, select_threshold

    val_proba = positive_proba(model, prepared.X_val)
    threshold = select_threshold(prepared.y_val, val_proba, config)
    test_proba = positive_proba(model, prepared.X_test)
    test_pred = apply_threshold(test_proba, threshold)
    metrics = evaluate_predictions(
        prepared.y_test,
        test_proba,
        test_pred,
        prepared.test_frame,
        config,
        include_reweigh_attributes=True,
    )
    metrics["threshold"] = float(threshold)
    return metrics


def run_reweighing(config: Any) -> None:
    """Compare baseline and reweighed models over the configured seeds."""
    from src.data import analysis_frame, make_split
    from src.features import prepare_matrices
    from src.models import bundle_path, fit_estimator, make_estimator, save_bundle
    from src.plots import plot_tradeoff
    from src.utils import get_logger, save_table

    logger = get_logger()
    frame = analysis_frame(config)
    attributes = {str(name): str(column) for name, column in config.reweighing.attributes.as_dict().items()}
    model_keys = [str(config.models.logistic_key), str(config.models.tree_key)]
    rows: list[dict[str, Any]] = []
    primary = int(config.seeds[0])
    for seed in [int(value) for value in config.seeds]:
        split = make_split(frame, seed, config)
        for feature_set in config.features.reweigh_sets:
            prepared = prepare_matrices(split, str(feature_set), config)
            for model_key in model_keys:
                baseline = fit_estimator(make_estimator(config, model_key, seed), prepared.X_train, prepared.y_train)
                baseline_metrics = _score_model(baseline, prepared, config)
                if seed == primary:
                    performance = pd.read_csv(config.results_root / "tables" / "baseline_performance.csv")
                    match = performance[
                        (performance["model"] == model_key)
                        & (performance["feature_set"] == feature_set)
                        & (performance["seed"] == seed)
                    ]
                    if len(match) == 1:
                        gap = abs(float(match.iloc[0]["roc_auc"]) - baseline_metrics["roc_auc"])
                        if gap > 1e-5:
                            raise AssertionError(f"Reweigh baseline AUC drifted by {gap} for {model_key} {feature_set}")
                for attribute, column in attributes.items():
                    rows.append(
                        {
                            "model": model_key,
                            "feature_set": feature_set,
                            "seed": seed,
                            "mitigation": "baseline",
                            "protected_attribute": attribute,
                            **baseline_metrics,
                        }
                    )
                    weights = reweighing_weights(prepared.y_train, prepared.train_frame[column])
                    mitigated = fit_estimator(
                        make_estimator(config, model_key, seed),
                        prepared.X_train,
                        prepared.y_train,
                        sample_weight=weights,
                    )
                    mitigated_metrics = _score_model(mitigated, prepared, config)
                    rows.append(
                        {
                            "model": model_key,
                            "feature_set": feature_set,
                            "seed": seed,
                            "mitigation": "reweighed",
                            "protected_attribute": attribute,
                            **mitigated_metrics,
                        }
                    )
                    if seed == primary:
                        save_bundle(
                            bundle_path(
                                config.results_root / "models",
                                model_key,
                                str(feature_set),
                                "reweighed",
                                seed,
                                attribute,
                            ),
                            {
                                "model": mitigated,
                                "preprocessor": prepared.preprocessor,
                                "threshold": mitigated_metrics["threshold"],
                                "feature_names": prepared.feature_names,
                                "feature_columns": prepared.feature_columns,
                                "feature_set": feature_set,
                                "model_key": model_key,
                                "seed": seed,
                                "mitigation": "reweighed",
                                "attribute": attribute,
                            },
                        )
                logger.info("Reweighing seed %s %s done", seed, feature_set)
    comparison = pd.DataFrame(rows)
    save_table(comparison, config.results_root / "tables" / "reweighing_comparison.csv")
    id_cols = ["model", "feature_set", "mitigation", "protected_attribute"]
    value_cols = [col for col in comparison.columns if col not in id_cols + ["seed"]]
    long = comparison.melt(id_vars=id_cols + ["seed"], value_vars=value_cols, var_name="metric", value_name="value")

    def _sample_std(series: pd.Series) -> float:
        if len(series) < 2:
            return float("nan")
        return float(series.std(ddof=1))

    summary = long.groupby(id_cols + ["metric"], as_index=False).agg(mean=("value", "mean"), std=("value", _sample_std))
    save_table(summary, config.results_root / "tables" / "reweighing_summary.csv")
    plot_rows = []
    for keys, part in summary.groupby(id_cols):
        key = dict(zip(id_cols, keys if isinstance(keys, tuple) else (keys,)))
        auc = part[part["metric"] == "roc_auc"].iloc[0]
        eo_name = _EO_BY_ATTRIBUTE[key["protected_attribute"]]
        dp_name = _DP_BY_ATTRIBUTE[key["protected_attribute"]]
        eo = part[part["metric"] == eo_name].iloc[0]
        dp = part[part["metric"] == dp_name].iloc[0]
        f1 = part[part["metric"] == "f1"].iloc[0]
        plot_rows.append(
            {
                **key,
                "roc_auc_mean": float(auc["mean"]),
                "roc_auc_std": float(auc["std"]),
                "eo_mean": float(eo["mean"]),
                "eo_std": float(eo["std"]),
                "dp_mean": float(dp["mean"]),
                "dp_std": float(dp["std"]),
                "f1_mean": float(f1["mean"]),
                "f1_std": float(f1["std"]),
            }
        )
    tradeoff = pd.DataFrame(plot_rows)
    save_table(tradeoff, config.results_root / "tables" / "reweighing_tradeoff.csv")
    plot_tradeoff(tradeoff, config.results_root / "figures" / "fairness_accuracy_tradeoff.png", config)
