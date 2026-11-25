"""Differentially private logistic regression across a grid of epsilon values."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.config import Config
from src.explain import _sample_index, _shap_values, explanation_stability
from src.features import aggregate_columns, categorical_columns
from src.reweighing import resample_by_weights, reweighing_weights
from src.utils import get_logger

logger = get_logger()

RECOMMENDATION_RULE = "smallest_epsilon_within_auc_tolerance_and_eo_no_worse"


def clip_matrix(features: np.ndarray, feature_clip: float, data_norm: float, atol: float) -> tuple[np.ndarray, dict[str, float]]:
    """Clip coordinates, then clip each row's L2 norm to ``data_norm``."""
    array = np.asarray(features, dtype=float)
    coord_hit = np.any(np.abs(array) > feature_clip, axis=1)
    clipped = np.clip(array, -feature_clip, feature_clip)
    norms = np.linalg.norm(clipped, axis=1)
    l2_hit = norms > data_norm
    scale = np.ones_like(norms)
    scale[l2_hit] = data_norm / norms[l2_hit]
    clipped = clipped * scale[:, None]
    max_after = float(np.linalg.norm(clipped, axis=1).max()) if len(clipped) else 0.0
    if max_after > data_norm + atol:
        raise AssertionError(f"Row norm {max_after} exceeds data_norm {data_norm}")
    stats = {
        "frac_coord_clipped": float(np.mean(coord_hit)) if len(array) else 0.0,
        "frac_l2_clipped": float(np.mean(l2_hit)) if len(array) else 0.0,
        "max_norm_before_l2": float(norms.max()) if len(norms) else 0.0,
        "max_norm_after": max_after,
    }
    return clipped, stats


def select_recommended(
    summary: pd.DataFrame,
    *,
    auc_tolerance: float,
    eo_slack: float,
    auc_column: str = "roc_auc_mean",
    eo_column: str = "eo_difference_sex_mean",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply the configured epsilon rule and return the chosen row plus all candidates.

    Within each feature set and reweighing flag, the reference is that cell's
    non-private model. A finite epsilon qualifies when its mean ROC-AUC is at
    least the reference AUC minus ``auc_tolerance`` and its mean sex equalized-odds
    difference is no larger than the reference plus ``eo_slack``. The winner is
    the smallest qualifying epsilon, then the higher AUC, then the lower EO.
    """
    if summary.empty:
        raise ValueError("Cannot recommend a model from an empty sweep")
    candidates: list[dict[str, Any]] = []
    for (feature_set, reweighed), part in summary.groupby(["feature_set", "reweighed"], sort=True):
        reference = part[~np.isfinite(part["epsilon"].to_numpy(dtype=float))]
        if reference.empty:
            continue
        ref_auc = float(reference.iloc[0][auc_column])
        ref_eo = float(reference.iloc[0][eo_column])
        for _, row in part.iterrows():
            epsilon = float(row["epsilon"])
            auc = float(row[auc_column])
            eo = float(row[eo_column])
            finite = bool(np.isfinite(epsilon))
            meets_auc = bool(finite and auc >= ref_auc - auc_tolerance)
            meets_eo = bool(finite and eo <= ref_eo + eo_slack)
            candidates.append(
                {
                    "epsilon": epsilon,
                    "feature_set": feature_set,
                    "reweighed": bool(reweighed),
                    "roc_auc_mean": auc,
                    "eo_difference_sex_mean": eo,
                    "reference_auc": ref_auc,
                    "reference_eo": ref_eo,
                    "meets_auc": meets_auc,
                    "meets_eo": meets_eo,
                    "meets_rule": bool(meets_auc and meets_eo),
                }
            )
    table = pd.DataFrame(candidates)
    qualifying = table[table["meets_rule"]].copy()
    if not qualifying.empty:
        qualifying = qualifying.sort_values(
            ["epsilon", "roc_auc_mean", "eo_difference_sex_mean"],
            ascending=[True, False, True],
        )
        chosen = qualifying.iloc[0].to_dict()
        chosen["reason"] = "smallest epsilon whose mean ROC-AUC is within tolerance of its paired non-private model and whose mean sex equalized-odds difference is no worse"
        chosen["fallback"] = False
        return chosen, table
    feasible = table[table["meets_eo"]].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            ["roc_auc_mean", "epsilon", "eo_difference_sex_mean"],
            ascending=[False, True, True],
        )
        chosen = feasible.iloc[0].to_dict()
        chosen["reason"] = "no epsilon met both rules; selected the highest-AUC setting whose sex equalized-odds difference is no worse than its paired non-private model"
        chosen["fallback"] = True
        return chosen, table
    private = table[~np.isfinite(table["epsilon"].to_numpy(dtype=float))].copy()
    private = private.sort_values("roc_auc_mean", ascending=False)
    chosen = private.iloc[0].to_dict()
    chosen["reason"] = "no private setting kept sex equalized odds at or below its paired non-private model; selected the non-private setting with the highest ROC-AUC"
    chosen["fallback"] = True
    return chosen, table


def _eps_label(epsilon: float) -> str:
    if not np.isfinite(epsilon):
        return "non-private"
    return f"{float(epsilon):g}"


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["epsilon", "feature_set", "reweighed", "model"]
    value_cols = [col for col in frame.columns if col not in id_cols + ["seed", "reweigh_attribute"]]
    grouped = frame.groupby(id_cols, dropna=False)
    means = grouped[value_cols].mean().add_suffix("_mean")
    stds = grouped[value_cols].std(ddof=1).add_suffix("_std")
    summary = means.join(stds).reset_index()
    return summary


def _fit_opacus(features: np.ndarray, target: np.ndarray, epsilon: float, seed: int, config: Config) -> Any:
    """One-layer DP-SGD logistic regression. Used only when ``use_opacus`` is true."""
    try:
        import torch
        from opacus import PrivacyEngine
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("use_opacus is true, but torch and opacus are not installed") from exc

    spec = config.privacy.opacus
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(spec.learning_rate))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    batch_size = min(int(spec.batch_size), len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    engine = PrivacyEngine()
    model, optimizer, loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=int(spec.epochs),
        target_epsilon=float(epsilon),
        target_delta=float(spec.target_delta),
        max_grad_norm=float(spec.max_grad_norm),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(int(spec.epochs)):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x).squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
    weight = model.weight.detach().cpu().numpy().reshape(1, -1)
    bias = model.bias.detach().cpu().numpy().ravel()

    class _Wrapper:
        def __init__(self) -> None:
            self.coef_ = weight
            self.intercept_ = bias
            self.classes_ = np.array([0, 1])

        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            logits = np.asarray(matrix, dtype=float) @ self.coef_.ravel() + float(self.intercept_[0])
            positive = 1.0 / (1.0 + np.exp(-logits))
            return np.column_stack([1.0 - positive, positive])

    return _Wrapper()


def run_privacy(config: Config) -> None:
    """Sweep epsilon and write the privacy, fairness, and SHAP-stability tables."""
    from src.data import analysis_frame, make_split
    from src.features import prepare_matrices
    from src.metrics import evaluate_predictions
    from src.models import fit_estimator, make_dp_logistic, make_logistic, positive_proba
    from src.plots import plot_epsilon_curves, plot_rank_heatmap
    from src.thresholds import apply_threshold, select_threshold

    frame = analysis_frame(config)
    epsilons = [float(value) for value in config.privacy.epsilons]
    if bool(config.privacy.include_non_private):
        epsilons = epsilons + [float("inf")]
    attribute_name = str(config.privacy.reweigh_attribute)
    attribute_column = str(config.reweighing.attributes.as_dict()[attribute_name])
    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    notes: list[str] = [str(config.privacy.scaler_limitation), ""]
    feature_clip = float(config.privacy.feature_clip)
    data_norm = float(config.privacy.data_norm)
    atol = float(config.numerical.clip_atol)

    for feature_set in config.features.dp_sets:
        for seed in [int(value) for value in config.seeds]:
            split = make_split(frame, seed, config)
            prepared = prepare_matrices(split, str(feature_set), config)
            x_train, train_stats = clip_matrix(prepared.X_train, feature_clip, data_norm, atol)
            x_val, _ = clip_matrix(prepared.X_val, feature_clip, data_norm, atol)
            x_test, _ = clip_matrix(prepared.X_test, feature_clip, data_norm, atol)
            weights = reweighing_weights(prepared.y_train, prepared.train_frame[attribute_column])
            for reweighed in (False, True):
                train_x, train_y = x_train, prepared.y_train
                if reweighed:
                    train_x, train_y = resample_by_weights(
                        x_train,
                        prepared.y_train,
                        weights,
                        seed + int(config.privacy.resample_seed_offset),
                    )
                background_index = _sample_index(
                    len(train_x),
                    int(config.privacy.shap_background_size),
                    seed + int(config.explain.background_seed_offset),
                )
                eval_index = _sample_index(
                    len(x_test),
                    int(config.privacy.shap_eval_size),
                    seed + int(config.explain.eval_seed_offset),
                )
                reference_values: np.ndarray | None = None
                reference_names: list[str] | None = None
                categorical = categorical_columns(config, prepared.feature_columns)
                ordered_epsilons = sorted(
                    epsilons,
                    key=lambda value: (np.isfinite(value), float(value) if np.isfinite(value) else -1.0),
                )
                for epsilon in ordered_epsilons:
                    if np.isfinite(epsilon):
                        model = make_dp_logistic(config, epsilon, seed)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(train_x, train_y)
                        leak_warnings = [str(item.message) for item in caught if item.category.__name__ == "PrivacyLeakWarning"]
                        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
                    else:
                        model = fit_estimator(make_logistic(config, seed), train_x, train_y)
                        leak_warnings = []
                        converged = True
                    val_proba = positive_proba(model, x_val)
                    threshold = select_threshold(prepared.y_val, val_proba, config)
                    test_proba = positive_proba(model, x_test)
                    test_pred = apply_threshold(test_proba, threshold)
                    metrics = evaluate_predictions(
                        prepared.y_test,
                        test_proba,
                        test_pred,
                        prepared.test_frame,
                        config,
                        include_reweigh_attributes=False,
                    )
                    values, _base = _shap_values(model, train_x[background_index], x_test[eval_index], "lr")
                    aggregated, names = aggregate_columns(values, prepared.feature_names, categorical)
                    if reference_values is None:
                        reference_values = aggregated
                        reference_names = names
                    stability = explanation_stability(
                        aggregated,
                        names,
                        reference_values,
                        reference_names if reference_names is not None else names,
                        k=int(config.explain.top_k),
                    )
                    if not np.isfinite(epsilon):
                        stability = {"spearman": 1.0, "jaccard_topk": 1.0, "mean_cosine": 1.0}
                    magnitude = np.mean(np.abs(aggregated), axis=0)
                    order = np.argsort(-magnitude)
                    for rank, index in enumerate(order, start=1):
                        rank_rows.append(
                            {
                                "epsilon": epsilon,
                                "epsilon_label": _eps_label(epsilon),
                                "feature_set": feature_set,
                                "reweighed": reweighed,
                                "seed": seed,
                                "feature": names[index],
                                "mean_abs_shap": float(magnitude[index]),
                                "rank": rank,
                            }
                        )
                    row = {
                        "epsilon": epsilon,
                        "feature_set": feature_set,
                        "reweighed": reweighed,
                        "reweigh_attribute": attribute_name if reweighed else "none",
                        "model": str(config.models.logistic_key),
                        "seed": seed,
                        "threshold": threshold,
                        "converged": float(converged),
                        "n_privacy_leak_warnings": float(len(leak_warnings)),
                        **train_stats,
                        **metrics,
                        "shap_spearman": stability["spearman"],
                        "shap_jaccard": stability["jaccard_topk"],
                        "shap_cosine": stability["mean_cosine"],
                    }
                    rows.append(row)
                    logger.info(
                        "DP eps=%s seed=%s features=%s reweigh=%s AUC=%.4f EO_sex=%.4f",
                        _eps_label(epsilon),
                        seed,
                        feature_set,
                        reweighed,
                        metrics["roc_auc"],
                        metrics["eo_difference_sex"],
                    )
                    if bool(config.privacy.use_opacus) and np.isfinite(epsilon):
                        opacus_model = _fit_opacus(train_x, train_y, epsilon, seed, config)
                        opacus_proba = positive_proba(opacus_model, x_test)
                        opacus_threshold = select_threshold(
                            prepared.y_val,
                            positive_proba(opacus_model, x_val),
                            config,
                        )
                        opacus_metrics = evaluate_predictions(
                            prepared.y_test,
                            opacus_proba,
                            apply_threshold(opacus_proba, opacus_threshold),
                            prepared.test_frame,
                            config,
                        )
                        opacus_row = dict(row)
                        opacus_row["model"] = "opacus_lr"
                        opacus_row.update(opacus_metrics)
                        rows.append(opacus_row)

    sweep = pd.DataFrame(rows)
    from src.utils import save_table

    tables = config.results_root / "tables"
    figures = config.results_root / "figures"
    save_table(sweep, tables / "dp_sweep.csv")
    model_name = str(config.models.logistic_key)
    summary_source = sweep[sweep["model"] == model_name].copy()
    summary = _summarize(summary_source)
    save_table(summary, tables / "dp_sweep_summary.csv")
    chosen, candidates = select_recommended(
        summary,
        auc_tolerance=float(config.recommendation.auc_tolerance),
        eo_slack=float(config.recommendation.eo_slack),
    )
    save_table(candidates, tables / "recommendation_candidates.csv")
    save_table(pd.DataFrame([chosen]), tables / "recommended_config.csv")
    logger.info("Recommended epsilon=%s (%s)", _eps_label(float(chosen["epsilon"])), chosen["reason"])

    ranks = pd.DataFrame(rank_rows)
    save_table(ranks, tables / "dp_shap_ranks.csv")
    curve_feature = str(config.reporting.dp_curve_feature_set)
    curve_reweigh = bool(config.reporting.dp_curve_reweighed)
    curve = summary[(summary["feature_set"] == curve_feature) & (summary["reweighed"] == curve_reweigh)].copy()
    curve["series"] = curve_feature + (" reweighed" if curve_reweigh else " no reweighing")
    # Also show the other feature set without reweighing so the figure is not a single line.
    extra = summary[(summary["reweighed"] == False)].copy()  # noqa: E712
    extra["series"] = extra["feature_set"].astype(str) + np.where(extra["reweighed"], " reweighed", " no reweighing")
    plot_source = extra.drop_duplicates(subset=["epsilon", "series"])
    plot_epsilon_curves(
        plot_source,
        figures / "dp_auc_vs_epsilon.png",
        config,
        value_col="roc_auc_mean",
        std_col="roc_auc_std",
        series_col="series",
        title="ROC-AUC against the privacy budget",
        ylabel="ROC-AUC (mean ± 1 std over seeds)",
    )
    fairness_long = []
    for metric, label in (
        ("dp_difference_sex", "DP difference, sex"),
        ("eo_difference_sex", "EO difference, sex"),
        ("dp_difference_age", "DP difference, age"),
        ("eo_difference_age", "EO difference, age"),
    ):
        piece = plot_source.copy()
        piece["series"] = label
        piece["value_mean"] = piece[f"{metric}_mean"]
        piece["value_std"] = piece[f"{metric}_std"]
        fairness_long.append(piece)
    # The four metrics above are for the displayed feature sets. Plot sex gaps for the configured curve only
    # so the figure stays readable, and a second figure is not required.
    sex_only = curve.copy()
    sex_long = []
    for metric, label in (("dp_difference_sex", "Demographic parity difference"), ("eo_difference_sex", "Equalized odds difference")):
        piece = sex_only.copy()
        piece["series"] = label
        piece["value_mean"] = piece[f"{metric}_mean"]
        piece["value_std"] = piece[f"{metric}_std"]
        sex_long.append(piece)
    plot_epsilon_curves(
        pd.concat(sex_long, ignore_index=True),
        figures / "dp_fairness_vs_epsilon.png",
        config,
        value_col="value_mean",
        std_col="value_std",
        series_col="series",
        title=f"Sex fairness gaps against ε ({curve['series'].iloc[0] if len(curve) else curve_feature})",
        ylabel="Gap (mean ± 1 std over seeds)",
    )
    group_long = []
    for column in summary.columns:
        if column.startswith("auc__") and column.endswith("_mean"):
            group_long.append(column)
    group_frame_rows = []
    for column in group_long:
        if not any(token in column for token in ("sex_label", "age_group")):
            continue
        piece = curve.copy()
        piece["series"] = column.replace("auc__", "").replace("_mean", "")
        piece["value_mean"] = piece[column]
        std_col = column.replace("_mean", "_std")
        piece["value_std"] = piece[std_col]
        group_frame_rows.append(piece)
    if group_frame_rows:
        plot_epsilon_curves(
            pd.concat(group_frame_rows, ignore_index=True),
            figures / "dp_group_auc_vs_epsilon.png",
            config,
            value_col="value_mean",
            std_col="value_std",
            series_col="series",
            title=f"Per-group ROC-AUC against ε ({curve_feature})",
            ylabel="Group ROC-AUC (mean ± 1 std over seeds)",
        )
    stability_long = []
    for metric, label in (
        ("shap_spearman", "Spearman rank correlation"),
        ("shap_jaccard", "Top-k Jaccard overlap"),
    ):
        piece = curve.copy()
        piece["series"] = label
        piece["value_mean"] = piece[f"{metric}_mean"]
        piece["value_std"] = piece[f"{metric}_std"]
        stability_long.append(piece)
    plot_epsilon_curves(
        pd.concat(stability_long, ignore_index=True),
        figures / "dp_shap_stability_vs_epsilon.png",
        config,
        value_col="value_mean",
        std_col="value_std",
        series_col="series",
        title=f"SHAP stability against the non-private model ({curve_feature})",
        ylabel="Stability (mean ± 1 std over seeds)",
    )
    heat = ranks[(ranks["feature_set"] == curve_feature) & (ranks["reweighed"] == curve_reweigh)]
    heat_mean = heat.groupby(["epsilon", "epsilon_label", "feature"], as_index=False)["mean_abs_shap"].mean()
    heat_rows = []
    for (epsilon, epsilon_label), part in heat_mean.groupby(["epsilon", "epsilon_label"]):
        ordered = part.sort_values("mean_abs_shap", ascending=False)
        for rank, (_, item) in enumerate(ordered.iterrows(), start=1):
            heat_rows.append(
                {
                    "epsilon": epsilon,
                    "epsilon_label": epsilon_label,
                    "feature": item["feature"],
                    "rank": rank,
                }
            )
    plot_rank_heatmap(
        pd.DataFrame(heat_rows),
        figures / "dp_shap_rank_heatmap.png",
        config,
        f"SHAP feature rank by privacy budget ({curve_feature})",
    )
    notes.append(
        f"Mean fraction of training rows clipped on a coordinate: {float(summary_source['frac_coord_clipped'].mean()):.4f}."
    )
    notes.append(
        f"Mean fraction of training rows clipped on L2 norm (bound {data_norm:g}): {float(summary_source['frac_l2_clipped'].mean()):.4f}."
    )
    notes.append(
        "Reweighing inside this sweep resamples the training set. Phase 4 applies the weights directly. "
        "SHAP stability compares each private model with the non-private model trained on the same clipped matrix, "
        "feature set, reweighing arm, and seed. The non-private model is fit first and is the reference."
    )
    log_path = config.results_root / "logs" / "privacy_notes.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
