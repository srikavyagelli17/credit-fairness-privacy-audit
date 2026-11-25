"""Fairness audit and proxy-discrimination checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import Config, feature_columns
from src.data import analysis_frame, make_split
from src.features import aggregate_columns, categorical_columns, prepare_matrices
from src.metrics import bootstrap_gaps, fairlearn_gaps, fairness_gaps, group_table
from src.models import load_bundle, positive_proba
from src.models import bundle_path
from src.plots import plot_grouped_rates, plot_proxy_bars, plot_tpr_fpr
from src.thresholds import apply_threshold
from src.utils import get_logger, save_table

logger = get_logger()


def _ordered_groups(labels: pd.Series, config: Config, attribute: str) -> list[str]:
    order_map = config.fairness.group_order.as_dict()
    preferred = [str(group) for group in order_map.get(attribute, [])]
    present = list(dict.fromkeys(labels.astype(str).tolist()))
    if preferred:
        return [group for group in preferred if group in present] + [group for group in present if group not in preferred]
    return sorted(present)


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left, right)
    if table.size == 0 or min(table.shape) < 2:
        return float("nan")
    chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    n_obs = float(table.to_numpy().sum())
    denom = n_obs * (min(table.shape) - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def _association_rows(train: pd.DataFrame, columns: list[str], target: pd.Series, task: str, feature_set: str, config: Config) -> list[dict[str, Any]]:
    categorical = set(categorical_columns(config, columns))
    rows = []
    binary = target.astype(int)
    for column in columns:
        if column in categorical:
            value = _cramers_v(train[column], binary)
            kind = "cramers_v"
        else:
            value = float(stats.pointbiserialr(binary, pd.to_numeric(train[column], errors="coerce"))[0])
            kind = "point_biserial"
        rows.append(
            {
                "task": task,
                "feature_set": feature_set,
                "target": task,
                "record_type": kind,
                "name": column,
                "value": value,
                "abs_value": abs(value) if np.isfinite(value) else float("nan"),
                "positive_class": "",
            }
        )
    return rows


def _proxy_model(
    split: Any,
    feature_set: str,
    target_train: np.ndarray,
    target_test: np.ndarray,
    positive_name: str,
    task: str,
    config: Config,
) -> tuple[list[dict[str, Any]], float]:
    prepared = prepare_matrices(split, feature_set, config)
    model = LogisticRegression(C=float(config.models.logistic.C), solver="lbfgs", max_iter=int(config.models.logistic.max_iter))
    model.fit(prepared.X_train, target_train)
    proba = positive_proba(model, prepared.X_test)
    auc = float(roc_auc_score(target_test, proba))
    categorical = categorical_columns(config, prepared.feature_columns)
    aggregated, names = aggregate_columns(np.abs(model.coef_.ravel()), prepared.feature_names, categorical)
    rows = [
        {
            "task": task,
            "feature_set": feature_set,
            "target": task,
            "record_type": "auc",
            "name": "model_auc",
            "value": auc,
            "abs_value": auc,
            "positive_class": positive_name,
        }
    ]
    order = np.argsort(-aggregated)
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "task": task,
                "feature_set": feature_set,
                "target": task,
                "record_type": "coefficient_abs",
                "name": names[index],
                "value": float(aggregated[index]),
                "abs_value": float(aggregated[index]),
                "positive_class": positive_name,
            }
        )
    rows.extend(
        _association_rows(split.train, feature_columns(config, feature_set), pd.Series(target_train), task, feature_set, config)
    )
    return rows, auc


def run_audit(config: Config) -> None:
    """Write fairness tables, proxy analysis, and the audit figures for the primary seed."""
    frame = analysis_frame(config)
    seed = int(config.seeds[0])
    split = make_split(frame, seed, config)
    atol = float(config.numerical.fairlearn_atol)
    ci_level = float(config.fairness.ci_level)
    n_bootstrap = int(config.fairness.n_bootstrap)
    boot_seed = seed + int(config.fairness.bootstrap_seed_offset)
    four_fifths = float(config.fairness.four_fifths)
    model_keys = [str(config.models.logistic_key), str(config.models.tree_key)]
    feature_sets = [str(name) for name in config.features.baseline_sets]
    attributes = [str(name) for name in config.fairness.attributes]
    fairness_rows: list[dict[str, Any]] = []
    group_frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    tables = config.results_root / "tables"
    figures = config.results_root / "figures"

    for feature_set in feature_sets:
        for model_key in model_keys:
            path = bundle_path(config.results_root / "models", model_key, feature_set, "baseline", seed)
            bundle = load_bundle(path)
            columns = list(bundle["feature_columns"])
            transformed = np.asarray(bundle["preprocessor"].transform(split.test[columns]), dtype=float)
            proba = positive_proba(bundle["model"], transformed)
            pred = apply_threshold(proba, float(bundle["threshold"]))
            for attribute in attributes:
                sensitive = split.test[attribute]
                point = fairness_gaps(split.test["default"].to_numpy(), pred, sensitive)
                try:
                    reference = fairlearn_gaps(split.test["default"].to_numpy(), pred, sensitive)
                except ValueError as exc:
                    logger.warning("fairlearn gap failed for %s/%s/%s: %s", model_key, feature_set, attribute, exc)
                    reference = None
                if reference is not None:
                    for key in ("dp_difference", "eo_difference"):
                        if np.isfinite(point[key]) and abs(point[key] - reference[key]) > atol:
                            raise AssertionError(
                                f"fairlearn {key} {reference[key]} disagrees with the project definition {point[key]} "
                                f"for {model_key} {feature_set} {attribute}"
                            )
                boot = bootstrap_gaps(
                    split.test["default"].to_numpy(),
                    pred,
                    sensitive,
                    n_bootstrap,
                    boot_seed,
                    ci_level,
                )
                for metric in ("dp_difference", "dp_ratio", "eo_difference"):
                    fairness_rows.append(
                        {
                            "model": model_key,
                            "feature_set": feature_set,
                            "attribute": attribute,
                            "metric": metric,
                            "value": point[metric],
                            "ci_low": boot["intervals"][f"{metric}_ci_low"],
                            "ci_high": boot["intervals"][f"{metric}_ci_high"],
                        }
                    )
                ratio = point["dp_ratio"]
                fairness_rows.append(
                    {
                        "model": model_key,
                        "feature_set": feature_set,
                        "attribute": attribute,
                        "metric": "four_fifths_pass",
                        "value": float(ratio >= four_fifths) if np.isfinite(ratio) else float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                    }
                )
                if reference is not None:
                    for metric in ("dp_difference", "eo_difference"):
                        fairness_rows.append(
                            {
                                "model": model_key,
                                "feature_set": feature_set,
                                "attribute": attribute,
                                "metric": f"fairlearn_{metric}",
                                "value": reference[metric],
                                "ci_low": float("nan"),
                                "ci_high": float("nan"),
                            }
                        )
                table = group_table(
                    split.test["default"].to_numpy(),
                    proba,
                    pred,
                    sensitive,
                    boot,
                    ci_level,
                )
                order = _ordered_groups(split.test[attribute], config, attribute)
                table["_order"] = table["group"].map({group: index for index, group in enumerate(order)})
                table = table.sort_values("_order").drop(columns="_order")
                group_frames[(model_key, feature_set, attribute)] = table
                save_table(table, tables / f"group_metrics_{model_key}_{feature_set}_{attribute}.csv")
                logger.info(
                    "Audit %s/%s/%s DP %.4f EO %.4f",
                    model_key,
                    feature_set,
                    attribute,
                    point["dp_difference"],
                    point["eo_difference"],
                )

    save_table(pd.DataFrame(fairness_rows), tables / "fairness_baseline.csv")

    proxy_rows: list[dict[str, Any]] = []
    sex_rows, sex_auc = _proxy_model(
        split,
        "no_sex",
        (split.train["sex_label"] == "male").astype(int).to_numpy(),
        (split.test["sex_label"] == "male").astype(int).to_numpy(),
        "male",
        "sex_from_no_sex",
        config,
    )
    age_rows, age_auc = _proxy_model(
        split,
        "no_sex_no_age",
        split.train["age_young"].to_numpy(dtype=int),
        split.test["age_young"].to_numpy(dtype=int),
        "young",
        "age_young_from_no_sex_no_age",
        config,
    )
    proxy_rows.extend(sex_rows)
    proxy_rows.extend(age_rows)
    proxy = pd.DataFrame(proxy_rows)
    save_table(proxy, tables / "proxy_analysis.csv")
    logger.info("Proxy AUC sex=%.4f age_young=%.4f", sex_auc, age_auc)

    def _approval_frame(attribute: str, specs: list[tuple[str, str, str]]) -> pd.DataFrame:
        pieces = []
        for model_key, feature_set, label in specs:
            table = group_frames[(model_key, feature_set, attribute)].copy()
            table["series"] = label
            pieces.append(table)
        return pd.concat(pieces, ignore_index=True)

    lr = str(config.models.logistic_key)
    tree = str(config.models.tree_key)
    plot_grouped_rates(
        _approval_frame(
            "sex_label",
            [
                (lr, "full", "LR full"),
                (lr, "no_sex", "LR no sex"),
                (tree, "full", "Tree full"),
                (tree, "no_sex", "Tree no sex"),
            ],
        ),
        figures / "approval_rate_by_sex.png",
        config,
        value_col="approval_rate",
        low_col="approval_ci_low",
        high_col="approval_ci_high",
        title="Approval rate by sex with 95% bootstrap intervals",
        ylabel="Approval rate (predicted non-default)",
        xlabel="Sex",
    )
    plot_grouped_rates(
        _approval_frame(
            "age_group",
            [
                (lr, "full", "LR full"),
                (lr, "no_sex", "LR no sex"),
                (lr, "no_sex_no_age", "LR no sex or age"),
            ],
        ),
        figures / "approval_rate_by_age.png",
        config,
        value_col="approval_rate",
        low_col="approval_ci_low",
        high_col="approval_ci_high",
        title="Approval rate by age group with 95% bootstrap intervals",
        ylabel="Approval rate (predicted non-default)",
        xlabel="Age group",
    )
    plot_grouped_rates(
        _approval_frame("sex_age", [(lr, "full", "LR full"), (lr, "no_sex", "LR no sex")]),
        figures / "approval_rate_by_sex_age.png",
        config,
        value_col="approval_rate",
        low_col="approval_ci_low",
        high_col="approval_ci_high",
        title="Approval rate by sex and age group",
        ylabel="Approval rate (predicted non-default)",
        xlabel="Sex and age group",
    )
    plot_tpr_fpr(
        group_frames[(lr, "full", "sex_label")],
        figures / "tpr_fpr_by_sex.png",
        config,
        "Default-class true and false positive rates by sex (logistic regression, full features)",
        "Sex",
    )
    plot_tpr_fpr(
        group_frames[(lr, "full", "age_group")],
        figures / "tpr_fpr_by_age.png",
        config,
        "Default-class true and false positive rates by age (logistic regression, full features)",
        "Age group",
    )
    for task, filename, title, xlabel in (
        (
            "sex_from_no_sex",
            "proxy_features_sex.png",
            "Features associated with sex in the no-sex feature set",
            "Absolute point-biserial correlation or Cramér's V",
        ),
        (
            "age_young_from_no_sex_no_age",
            "proxy_features_age.png",
            "Features associated with age under 30, sex and age removed",
            "Absolute point-biserial correlation or Cramér's V",
        ),
    ):
        part = proxy[(proxy["task"] == task) & (proxy["record_type"].isin(["point_biserial", "cramers_v"]))]
        part = part.sort_values("abs_value", ascending=False).head(10)
        plot_proxy_bars(part, figures / filename, config, title, xlabel)
