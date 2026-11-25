"""Load result tables and build the numbers used in generated documents.

Every scalar inserted into the README, model card, or governance mapping is
registered with the CSV it came from. The writer re-reads that CSV before
rendering so a mismatch fails the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, docs_dir
from src.utils import dataframe_to_markdown, json_safe, slug


class Registry:
    """Provenance for every reported scalar."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def one(
        self,
        key: str,
        frame: pd.DataFrame,
        filters: dict[str, Any],
        column: str,
        source: str,
        *,
        headline: bool = False,
    ) -> Any:
        """Return the single cell matching ``filters`` and record its source."""
        mask = pd.Series(True, index=frame.index)
        for field, expected in filters.items():
            series = frame[field]
            if isinstance(expected, (bool, np.bool_)):
                mask &= series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"}).astype(bool) == bool(expected)
            elif expected == "inf" or (isinstance(expected, float) and np.isinf(expected) and expected > 0):
                mask &= ~np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)) & (
                    pd.to_numeric(series, errors="coerce").to_numpy(dtype=float) > 0
                )
            elif expected == "-inf" or (isinstance(expected, float) and np.isinf(expected) and expected < 0):
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                mask &= np.isinf(numeric) & (numeric < 0)
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                mask &= np.isfinite(numeric) & np.isclose(numeric, float(expected), rtol=0.0, atol=1e-8)
            else:
                mask &= series.astype(str) == str(expected)
        hit = frame.loc[mask, column]
        if len(hit) != 1:
            raise KeyError(f"{key}: expected 1 row for {filters} column {column} in {source}, found {len(hit)}")
        value = hit.iloc[0]
        if isinstance(value, str) and value.strip().lower() in {"inf", "-inf"}:
            value = float(value)
        elif isinstance(value, (np.floating, float)):
            value = float(value)
        elif isinstance(value, (np.integer,)):
            value = int(value)
        self.items.append(
            {
                "key": key,
                "value": value,
                "source": source,
                "filters": filters,
                "column": column,
                "headline": headline,
            }
        )
        return value


def _read(config: Config, name: str) -> pd.DataFrame:
    path = config.results_root / "tables" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing results table {path}")
    frame = pd.read_csv(path)
    if "epsilon" in frame.columns:
        frame["epsilon"] = pd.to_numeric(frame["epsilon"], errors="coerce")
    return frame


def fmt(value: Any, digits: int = 4) -> str:
    """Format a finite float with a fixed number of digits."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    """Format mean ± sample standard deviation."""
    if std is None or (isinstance(std, (float, np.floating)) and not np.isfinite(std)):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"


def fmt_rate(value: Any) -> str:
    """Format a rate as a proportion and a percent, both from the same value."""
    number = float(value)
    return f"{number:.4f} ({number * 100:.2f}%)"


def _ci_phrase(name: str, value: float, low: float, high: float) -> str:
    if np.isfinite(low) and np.isfinite(high) and low <= 0 <= high:
        zero = "includes 0"
    elif np.isfinite(low) and np.isfinite(high):
        zero = "excludes 0"
    else:
        zero = "has no finite interval"
    return f"{name} is {fmt(value)} (95% bootstrap CI {fmt(low)} to {fmt(high)}; the interval {zero})"


def _direction(before: float, after: float) -> str:
    if after < before - 1e-4:
        return "lower"
    if after > before + 1e-4:
        return "higher"
    return "effectively unchanged"


def build_context(config: Config) -> dict[str, Any]:
    """Assemble template variables and a provenance manifest from ``results/tables``."""
    digits = int(config.reporting.round_digits)
    reg = Registry()
    primary = str(config.reporting.primary_model)
    tree = str(config.reporting.comparison_model)
    full = str(config.reporting.primary_feature_set)
    no_sex = str(config.reporting.unawareness_feature_set)
    no_age = str(config.reporting.age_removed_feature_set)
    focus = str(config.reporting.reweigh_focus_attribute)

    summary = _read(config, "dataset_summary.csv")
    n_rows = reg.one("n_rows", summary, {"group_type": "overall", "group": "all"}, "n", "tables/dataset_summary.csv", headline=True)
    default_rate = reg.one(
        "default_rate", summary, {"group_type": "overall", "group": "all"}, "default_rate", "tables/dataset_summary.csv", headline=True
    )
    sex_bits = []
    for group in config.fairness.group_order.sex_label:
        count = reg.one(f"n_{group}", summary, {"group_type": "sex", "group": group}, "n", "tables/dataset_summary.csv", headline=True)
        rate = reg.one(
            f"rate_{group}", summary, {"group_type": "sex", "group": group}, "default_rate", "tables/dataset_summary.csv", headline=True
        )
        sex_bits.append(f"{group} n={int(count)}, default rate {fmt_rate(rate)}")
    age_bits = []
    for group in config.fairness.group_order.age_group:
        count = reg.one(
            f"n_age_{slug(group)}", summary, {"group_type": "age_group", "group": group}, "n", "tables/dataset_summary.csv", headline=True
        )
        rate = reg.one(
            f"rate_age_{slug(group)}",
            summary,
            {"group_type": "age_group", "group": group},
            "default_rate",
            "tables/dataset_summary.csv",
            headline=True,
        )
        age_bits.append(f"{group} n={int(count)}, default rate {fmt_rate(rate)}")

    performance = _read(config, "baseline_performance.csv")

    def perf(model: str, feature_set: str, column: str, headline: bool = False) -> Any:
        return reg.one(
            f"perf_{model}_{feature_set}_{column}",
            performance,
            {"model": model, "feature_set": feature_set},
            column,
            "tables/baseline_performance.csv",
            headline=headline,
        )

    lr_auc = perf(primary, full, "roc_auc", True)
    lr_f1 = perf(primary, full, "f1", True)
    lr_acc = perf(primary, full, "accuracy", True)
    lr_recall = perf(primary, full, "recall", True)
    lr_precision = perf(primary, full, "precision", True)
    lr_brier = perf(primary, full, "brier", True)
    lr_threshold = perf(primary, full, "threshold", True)
    lr_nosex_auc = perf(primary, no_sex, "roc_auc", True)
    lr_nosex_f1 = perf(primary, no_sex, "f1", True)
    tree_auc = perf(tree, full, "roc_auc", True)
    tree_f1 = perf(tree, full, "f1", True)

    fairness = _read(config, "fairness_baseline.csv")

    def gap(model: str, feature_set: str, attribute: str, metric: str, column: str = "value", headline: bool = False) -> Any:
        return reg.one(
            f"fair_{model}_{feature_set}_{attribute}_{metric}_{column}",
            fairness,
            {"model": model, "feature_set": feature_set, "attribute": attribute, "metric": metric},
            column,
            "tables/fairness_baseline.csv",
            headline=headline,
        )

    sex_dp = gap(primary, full, "sex_label", "dp_difference", headline=True)
    sex_dp_lo = gap(primary, full, "sex_label", "dp_difference", "ci_low", True)
    sex_dp_hi = gap(primary, full, "sex_label", "dp_difference", "ci_high", True)
    sex_eo = gap(primary, full, "sex_label", "eo_difference", headline=True)
    sex_eo_lo = gap(primary, full, "sex_label", "eo_difference", "ci_low", True)
    sex_eo_hi = gap(primary, full, "sex_label", "eo_difference", "ci_high", True)
    sex_ratio = gap(primary, full, "sex_label", "dp_ratio", headline=True)
    sex_fifths = gap(primary, full, "sex_label", "four_fifths_pass", headline=True)
    nosex_dp = gap(primary, no_sex, "sex_label", "dp_difference", headline=True)
    nosex_dp_lo = gap(primary, no_sex, "sex_label", "dp_difference", "ci_low", True)
    nosex_dp_hi = gap(primary, no_sex, "sex_label", "dp_difference", "ci_high", True)
    nosex_eo = gap(primary, no_sex, "sex_label", "eo_difference", headline=True)
    nosex_eo_lo = gap(primary, no_sex, "sex_label", "eo_difference", "ci_low", True)
    nosex_eo_hi = gap(primary, no_sex, "sex_label", "eo_difference", "ci_high", True)
    nosex_ratio = gap(primary, no_sex, "sex_label", "dp_ratio", headline=True)
    age_dp = gap(primary, full, "age_group", "dp_difference", headline=True)
    age_dp_lo = gap(primary, full, "age_group", "dp_difference", "ci_low", True)
    age_dp_hi = gap(primary, full, "age_group", "dp_difference", "ci_high", True)
    age_eo = gap(primary, full, "age_group", "eo_difference", headline=True)
    age_eo_lo = gap(primary, full, "age_group", "eo_difference", "ci_low", True)
    age_eo_hi = gap(primary, full, "age_group", "eo_difference", "ci_high", True)
    age_removed_dp = gap(primary, no_age, "age_group", "dp_difference", headline=True)
    age_removed_eo = gap(primary, no_age, "age_group", "eo_difference", headline=True)
    age_removed_eo_lo = gap(primary, no_age, "age_group", "eo_difference", "ci_low")
    age_removed_eo_hi = gap(primary, no_age, "age_group", "eo_difference", "ci_high")

    groups = _read(config, f"group_metrics_{primary}_{full}_sex_label.csv")
    approval = {}
    for group in config.fairness.group_order.sex_label:
        approval[group] = reg.one(
            f"approval_{group}",
            groups,
            {"group": group},
            "approval_rate",
            f"tables/group_metrics_{primary}_{full}_sex_label.csv",
            headline=True,
        )
    age_groups = _read(config, f"group_metrics_{primary}_{full}_age_group.csv")
    age_approval = []
    for group in config.fairness.group_order.age_group:
        value = reg.one(
            f"approval_age_{slug(group)}",
            age_groups,
            {"group": group},
            "approval_rate",
            f"tables/group_metrics_{primary}_{full}_age_group.csv",
            headline=True,
        )
        age_approval.append(f"{group} {fmt_rate(value)}")

    proxy = _read(config, "proxy_analysis.csv")
    sex_proxy = reg.one(
        "sex_proxy_auc",
        proxy,
        {"task": "sex_from_no_sex", "record_type": "auc", "name": "model_auc"},
        "value",
        "tables/proxy_analysis.csv",
        headline=True,
    )
    age_proxy = reg.one(
        "age_proxy_auc",
        proxy,
        {"task": "age_young_from_no_sex_no_age", "record_type": "auc", "name": "model_auc"},
        "value",
        "tables/proxy_analysis.csv",
        headline=True,
    )

    def association_list(task: str) -> str:
        part = proxy[(proxy["task"] == task) & (proxy["record_type"].isin(["point_biserial", "cramers_v"]))]
        part = part.sort_values("abs_value", ascending=False).head(5)
        bits = [f"{row['name']} ({row['record_type']} {float(row['value']):.4f})" for _, row in part.iterrows()]
        return "; ".join(bits)

    tradeoff = _read(config, "reweighing_tradeoff.csv")

    def trade(model: str, feature_set: str, attribute: str, mitigation: str, column: str, headline: bool = False) -> Any:
        return reg.one(
            f"rw_{model}_{feature_set}_{attribute}_{mitigation}_{column}",
            tradeoff,
            {"model": model, "feature_set": feature_set, "protected_attribute": attribute, "mitigation": mitigation},
            column,
            "tables/reweighing_tradeoff.csv",
            headline=headline,
        )

    base_auc = trade(primary, full, focus, "baseline", "roc_auc_mean", True)
    base_auc_std = trade(primary, full, focus, "baseline", "roc_auc_std", True)
    base_eo = trade(primary, full, focus, "baseline", "eo_mean", True)
    base_eo_std = trade(primary, full, focus, "baseline", "eo_std", True)
    base_dp = trade(primary, full, focus, "baseline", "dp_mean", True)
    base_dp_std = trade(primary, full, focus, "baseline", "dp_std", True)
    rw_auc = trade(primary, full, focus, "reweighed", "roc_auc_mean", True)
    rw_auc_std = trade(primary, full, focus, "reweighed", "roc_auc_std", True)
    rw_eo = trade(primary, full, focus, "reweighed", "eo_mean", True)
    rw_eo_std = trade(primary, full, focus, "reweighed", "eo_std", True)
    rw_dp = trade(primary, full, focus, "reweighed", "dp_mean", True)
    rw_dp_std = trade(primary, full, focus, "reweighed", "dp_std", True)
    nosex_base_auc = trade(primary, no_sex, focus, "baseline", "roc_auc_mean", True)
    nosex_base_auc_std = trade(primary, no_sex, focus, "baseline", "roc_auc_std")
    nosex_base_eo = trade(primary, no_sex, focus, "baseline", "eo_mean", True)
    nosex_base_eo_std = trade(primary, no_sex, focus, "baseline", "eo_std")
    nosex_rw_auc = trade(primary, no_sex, focus, "reweighed", "roc_auc_mean", True)
    nosex_rw_auc_std = trade(primary, no_sex, focus, "reweighed", "roc_auc_std")
    nosex_rw_eo = trade(primary, no_sex, focus, "reweighed", "eo_mean", True)
    nosex_rw_eo_std = trade(primary, no_sex, focus, "reweighed", "eo_std")

    dp = _read(config, "dp_sweep_summary.csv")
    curve_feature = str(config.reporting.dp_curve_feature_set)
    curve_reweigh = bool(config.reporting.dp_curve_reweighed)

    def dp_cell(epsilon: float, column: str, headline: bool = False) -> Any:
        filters: dict[str, Any] = {
            "epsilon": epsilon,
            "feature_set": curve_feature,
            "reweighed": curve_reweigh,
            "model": primary,
        }
        label = "inf" if not np.isfinite(epsilon) else f"{epsilon:g}"
        return reg.one(
            f"dp_{label}_{column}",
            dp,
            filters,
            column,
            "tables/dp_sweep_summary.csv",
            headline=headline,
        )

    finite_eps = sorted(float(value) for value in config.privacy.epsilons)
    smallest = finite_eps[0]
    largest = finite_eps[-1]
    dp_private_auc = dp_cell(float("inf"), "roc_auc_mean", True)
    dp_private_auc_std = dp_cell(float("inf"), "roc_auc_std", True)
    dp_private_eo = dp_cell(float("inf"), "eo_difference_sex_mean", True)
    dp_private_eo_std = dp_cell(float("inf"), "eo_difference_sex_std")
    dp_small_auc = dp_cell(smallest, "roc_auc_mean", True)
    dp_small_auc_std = dp_cell(smallest, "roc_auc_std")
    dp_small_eo = dp_cell(smallest, "eo_difference_sex_mean", True)
    dp_small_eo_std = dp_cell(smallest, "eo_difference_sex_std")
    dp_small_dp = dp_cell(smallest, "dp_difference_sex_mean", True)
    dp_large_auc = dp_cell(largest, "roc_auc_mean", True)
    dp_large_auc_std = dp_cell(largest, "roc_auc_std")
    dp_large_eo = dp_cell(largest, "eo_difference_sex_mean", True)
    dp_small_spear = dp_cell(smallest, "shap_spearman_mean", True)
    dp_small_jacc = dp_cell(smallest, "shap_jaccard_mean", True)
    dp_large_spear = dp_cell(largest, "shap_spearman_mean", True)
    dp_large_jacc = dp_cell(largest, "shap_jaccard_mean", True)

    sex_drop_bits = []
    for group in config.fairness.group_order.sex_label:
        column = f"auc__sex_label__{slug(group)}_mean"
        base = dp_cell(float("inf"), column, True)
        low = dp_cell(smallest, column, True)
        sex_drop_bits.append(f"{group} {fmt(base - low)} (from {fmt(base)} to {fmt(low)})")
    smaller_sex = min(
        ((group, reg.one(f"n_{group}", summary, {"group_type": "sex", "group": group}, "n", "tables/dataset_summary.csv")) for group in config.fairness.group_order.sex_label),
        key=lambda item: item[1],
    )[0]

    recommended = _read(config, "recommended_config.csv")
    rec_eps = float(recommended.iloc[0]["epsilon"])
    rec_features = str(recommended.iloc[0]["feature_set"])
    rec_reweigh = str(recommended.iloc[0]["reweighed"]).strip().lower() in {"1", "true", "yes"}
    rec_auc = float(recommended.iloc[0]["roc_auc_mean"])
    rec_eo = float(recommended.iloc[0]["eo_difference_sex_mean"])
    rec_reason = str(recommended.iloc[0]["reason"])
    rec_reason = rec_reason[:1].upper() + rec_reason[1:]
    reg.items.append(
        {
            "key": "recommended_auc",
            "value": rec_auc,
            "source": "tables/recommended_config.csv",
            "filters": {"row": 0},
            "column": "roc_auc_mean",
            "headline": True,
        }
    )
    reg.items.append(
        {
            "key": "recommended_eo",
            "value": rec_eo,
            "source": "tables/recommended_config.csv",
            "filters": {"row": 0},
            "column": "eo_difference_sex_mean",
            "headline": True,
        }
    )

    stability = _read(config, "shap_stability.csv")
    stab_reweigh = reg.one(
        "shap_reweigh_spearman",
        stability,
        {"comparison": "baseline_vs_reweigh_sex"},
        "spearman",
        "tables/shap_stability.csv",
        headline=True,
    )
    stab_proxy = reg.one(
        "shap_full_nosex_spearman",
        stability,
        {"comparison": "full_vs_no_sex"},
        "spearman",
        "tables/shap_stability.csv",
        headline=True,
    )
    global_full = _read(config, f"shap_global_{primary}_{full}_baseline.csv")
    top_feature = str(global_full.iloc[0]["feature"])
    top_value = float(global_full.iloc[0]["mean_abs_shap"])
    sex_rank_rows = global_full[global_full["feature"] == "SEX"]
    sex_rank = int(sex_rank_rows.iloc[0]["rank"]) if len(sex_rank_rows) else None
    global_nosex = _read(config, f"shap_global_{primary}_{no_sex}_baseline.csv")
    top_nosex = str(global_nosex.iloc[0]["feature"])
    local = _read(config, "local_explanations.csv")
    local_bits = []
    for case_type, part in local.groupby("case_type", sort=False):
        text = str(part.iloc[0]["adverse_action_text"])
        local_bits.append(f"{case_type} (row {int(part.iloc[0]['row_id'])}): {text}")

    weak = float(config.reporting.proxy_auc_weak)
    moderate = float(config.reporting.proxy_auc_moderate)

    def proxy_words(auc: float) -> str:
        if auc < weak:
            return "close to 0.5, so the measured proxy signal is weak"
        if auc < moderate:
            return "above the weak-proxy cutoff in config, so the attribute is partly recoverable"
        return "at or above the moderate-proxy cutoff in config, so the attribute is recoverable from the other inputs"

    fifths_word = "meets" if float(sex_fifths) >= 1 else "misses"
    rq1 = (
        f"On the modeling frame of {int(n_rows)} rows, the default rate is {fmt_rate(default_rate)}. "
        f"By sex, {'; '.join(sex_bits)}. By age, {'; '.join(age_bits)}. "
        f"The primary logistic regression on all features has test ROC-AUC {fmt(lr_auc)}, F1 {fmt(lr_f1)}, "
        f"accuracy {fmt(lr_acc)}, precision {fmt(lr_precision)}, recall {fmt(lr_recall)}, "
        f"Brier score {fmt(lr_brier)}, and a validation-chosen default threshold of {fmt(lr_threshold)}. "
        f"Removing sex changes ROC-AUC to {fmt(lr_nosex_auc)} and F1 to {fmt(lr_nosex_f1)}. "
        f"The tree model on all features has ROC-AUC {fmt(tree_auc)} and F1 {fmt(tree_f1)}. "
        f"For sex, on the full logistic model, {_ci_phrase('the demographic parity difference', sex_dp, sex_dp_lo, sex_dp_hi)} "
        f"and {_ci_phrase('the equalized odds difference', sex_eo, sex_eo_lo, sex_eo_hi)}. "
        f"The approval-rate ratio is {fmt(sex_ratio)}, which {fifths_word} the four-fifths threshold of {float(config.fairness.four_fifths):.2f}. "
        f"Approval rates are {', '.join(f'{group} {fmt_rate(approval[str(group)])}' for group in config.fairness.group_order.sex_label)}. "
        f"After sex is removed, {_ci_phrase('the demographic parity difference', nosex_dp, nosex_dp_lo, nosex_dp_hi)} "
        f"and {_ci_phrase('the equalized odds difference', nosex_eo, nosex_eo_lo, nosex_eo_hi)}; the approval ratio is {fmt(nosex_ratio)}. "
        f"A logistic model predicting sex from the no-sex features has test ROC-AUC {fmt(sex_proxy)}, which is {proxy_words(float(sex_proxy))}. "
        f"The strongest training-set associations with sex are {association_list('sex_from_no_sex')}. "
        f"For age group, on the full logistic model, {_ci_phrase('the demographic parity difference', age_dp, age_dp_lo, age_dp_hi)} "
        f"and {_ci_phrase('the equalized odds difference', age_eo, age_eo_lo, age_eo_hi)}. "
        f"Approval rates by age are {', '.join(age_approval)}. "
        f"With sex and age removed, the age-group demographic parity difference is {fmt(age_removed_dp)} and the equalized odds difference is {fmt(age_removed_eo)} "
        f"(95% CI {fmt(age_removed_eo_lo)} to {fmt(age_removed_eo_hi)}). "
        f"A model predicting age under 30 from the remaining features has test ROC-AUC {fmt(age_proxy)}, which is {proxy_words(float(age_proxy))}. "
        f"The strongest associations with being under 30 are {association_list('age_young_from_no_sex_no_age')}."
    )
    rq2 = (
        f"Reweighing is estimated over {len(list(config.seeds))} seeds. "
        f"For logistic regression on all features, reweighing on {focus} moves ROC-AUC from {fmt_pm(base_auc, base_auc_std)} "
        f"to {fmt_pm(rw_auc, rw_auc_std)} ({_direction(float(base_auc), float(rw_auc))}) and moves equalized odds difference from "
        f"{fmt_pm(base_eo, base_eo_std)} to {fmt_pm(rw_eo, rw_eo_std)} ({_direction(float(base_eo), float(rw_eo))}). "
        f"Demographic parity difference moves from {fmt_pm(base_dp, base_dp_std)} to {fmt_pm(rw_dp, rw_dp_std)} "
        f"({_direction(float(base_dp), float(rw_dp))}). "
        f"On the no-sex feature set, ROC-AUC moves from {fmt_pm(nosex_base_auc, nosex_base_auc_std)} to {fmt_pm(nosex_rw_auc, nosex_rw_auc_std)} "
        f"and equalized odds difference moves from {fmt_pm(nosex_base_eo, nosex_base_eo_std)} to {fmt_pm(nosex_rw_eo, nosex_rw_eo_std)}."
    )
    eps_text = ", ".join(f"{value:g}" for value in finite_eps)
    rq3 = (
        f"The plotted privacy curve is logistic regression on {curve_feature}, reweighing={curve_reweigh}, "
        f"with finite epsilon values {eps_text}. "
        f"The paired non-private model has ROC-AUC {fmt_pm(dp_private_auc, dp_private_auc_std)} and sex equalized odds difference "
        f"{fmt_pm(dp_private_eo, dp_private_eo_std)}. "
        f"At epsilon {smallest:g}, ROC-AUC is {fmt_pm(dp_small_auc, dp_small_auc_std)}, sex equalized odds difference is "
        f"{fmt_pm(dp_small_eo, dp_small_eo_std)}, and sex demographic parity difference is {fmt(dp_small_dp)}. "
        f"At epsilon {largest:g}, ROC-AUC is {fmt_pm(dp_large_auc, dp_large_auc_std)} and sex equalized odds difference is {fmt(dp_large_eo)}. "
        f"Relative to the non-private model, the change in group ROC-AUC at epsilon {smallest:g} is {'; '.join(sex_drop_bits)}. "
        f"The smaller sex group in the modeling frame is {smaller_sex}. "
        f"SHAP Spearman correlation with the non-private model is {fmt(dp_small_spear)} at epsilon {smallest:g} "
        f"(top-k Jaccard {fmt(dp_small_jacc)}) and {fmt(dp_large_spear)} at epsilon {largest:g} (top-k Jaccard {fmt(dp_large_jacc)}). "
        f"The recommendation rule selected epsilon {_eps_text(rec_eps)} on {rec_features} with reweighing={rec_reweigh}: "
        f"ROC-AUC {fmt(rec_auc)}, sex equalized odds difference {fmt(rec_eo)}. {rec_reason}."
    )
    sample = config.runtime.sample if "runtime" in config.as_dict() else None
    run_note = (
        f"Seeds: {list(config.seeds)}. Bootstrap resamples: {int(config.fairness.n_bootstrap)}. "
        f"Fast mode: {bool(config.runtime.fast) if 'runtime' in config.as_dict() else False}. "
        + (f"This run uses a stratified sample of {int(sample)} rows. " if sample else "This run uses the full cleaned dataset. ")
    )
    shap_sentence = (
        f"On the full logistic model the largest mean absolute SHAP feature is {top_feature} ({top_value:.4f}). "
        + (f"SEX has rank {sex_rank}. " if sex_rank is not None else "")
        + f"On the no-sex model the largest is {top_nosex}. "
        f"Spearman correlation of mean absolute SHAP ranks is {fmt(stab_reweigh)} between the baseline and the sex-reweighed model, "
        f"and {fmt(stab_proxy)} between the full and no-sex models on their shared features."
    )

    risks = _risk_rows(
        config,
        sex_phrase=_ci_phrase("Sex demographic parity difference", sex_dp, sex_dp_lo, sex_dp_hi)
        + "; "
        + _ci_phrase("sex equalized odds difference", sex_eo, sex_eo_lo, sex_eo_hi)
        + f"; approval ratio {fmt(sex_ratio)}",
        age_phrase=_ci_phrase("Age-group demographic parity difference", age_dp, age_dp_lo, age_dp_hi)
        + "; "
        + _ci_phrase("age-group equalized odds difference", age_eo, age_eo_lo, age_eo_hi),
        proxy_phrase=(
            f"No-sex sex demographic parity difference {fmt(nosex_dp)} and equalized odds difference {fmt(nosex_eo)}, "
            f"compared with {fmt(sex_dp)} and {fmt(sex_eo)} when sex is an input. "
            f"Sex-from-features ROC-AUC {fmt(sex_proxy)}. Age-under-30 ROC-AUC without sex and age {fmt(age_proxy)}."
        ),
        shap_sentence=shap_sentence,
        local_count=local["case_type"].nunique(),
        rq2=rq2,
        rq3=rq3,
        rec_text=f"epsilon {_eps_text(rec_eps)}, feature set {rec_features}, reweighing={rec_reweigh}",
    )
    baseline_view = performance[
        ["model", "feature_set", "roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy", "brier", "threshold"]
    ]
    fair_view = fairness[fairness["metric"].isin(["dp_difference", "dp_ratio", "eo_difference", "four_fifths_pass"])][
        ["model", "feature_set", "attribute", "metric", "value", "ci_low", "ci_high"]
    ]
    dp_view_cols = [
        "epsilon",
        "feature_set",
        "reweighed",
        "roc_auc_mean",
        "roc_auc_std",
        "eo_difference_sex_mean",
        "eo_difference_sex_std",
        "dp_difference_sex_mean",
        "shap_spearman_mean",
        "shap_jaccard_mean",
    ]
    context = {
        "run_note": run_note,
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "shap_sentence": shap_sentence,
        "local_explanations": " ".join(local_bits),
        "risks": risks,
        "dataset_table": dataframe_to_markdown(summary, digits),
        "baseline_table": dataframe_to_markdown(baseline_view, digits),
        "fairness_table": dataframe_to_markdown(fair_view, digits),
        "reweigh_table": dataframe_to_markdown(tradeoff, digits),
        "dp_table": dataframe_to_markdown(dp[dp_view_cols], digits),
        "recommended_text": f"epsilon {_eps_text(rec_eps)} on feature set {rec_features}, reweighing={rec_reweigh}. {rec_reason} Mean ROC-AUC {fmt(rec_auc)}; mean sex equalized odds difference {fmt(rec_eo)}.",
        "config_hash": config.canonical_hash(),
        "scaler_limitation": str(config.privacy.scaler_limitation),
        "figure_prefix": "results/figures" if docs_dir(config) == config.root / "docs" else "figures",
        "n_local": int(local["case_type"].nunique()),
        "monitoring_eo": fmt(float(config.monitoring.eo_margin)),
        "monitoring_auc": fmt(float(config.monitoring.auc_margin)),
        "manifest": reg.items,
    }
    return context


def _eps_text(epsilon: float) -> str:
    if not np.isfinite(epsilon):
        return "non-private"
    return f"{float(epsilon):g}"


def _risk_rows(config: Config, **kwargs: Any) -> list[dict[str, str]]:
    """Governance rows. Legal wording is fixed; evidence strings are measured."""
    del config
    return [
        {
            "risk": "Sex disparity in approval and error rates",
            "evidence": kwargs["sex_phrase"],
            "law": "Art. 9 risk management system; Art. 10(2)(f) and (g) bias examination and mitigation; Annex III, point 5(b) (creditworthiness is high-risk, fraud detection excepted).",
            "nist": "Map (sex is a protected attribute in this lending context); Measure (group rates, demographic parity, equalized odds, bootstrap intervals).",
            "control": "MetricFrame-style group table, approval-based demographic parity, default-class equalized odds, validation F1 threshold, and Kamiran-Calders reweighing on sex.",
            "residual": kwargs["rq2"],
        },
        {
            "risk": "Age disparity",
            "evidence": kwargs["age_phrase"],
            "law": "Art. 9 and Art. 10(2)(f)–(g). Art. 9(9) calls for attention to people under 18 and other vulnerable groups; this dataset's youngest clients are adults, and age is still a fairness factor in the audit.",
            "nist": "Map (age groups and the binary under-30 attribute); Measure (the same gap metrics on age_group).",
            "control": "Age-group metrics, a no-sex-no-age feature set, and reweighing on age_young and on the sex × age_young intersection.",
            "residual": "Age gaps after removing age, and the reweighing trade-off, are the measured values in the fairness and reweighing tables rather than a claim that the gap is closed.",
        },
        {
            "risk": "Proxy discrimination after dropping sex or age",
            "evidence": kwargs["proxy_phrase"],
            "law": "Art. 10(2)(f) covers biases that can lead to discrimination even when a prohibited attribute is not an input. Art. 4a of the consolidated text (27 July 2026) is the special-category processing rule for bias detection and correction; that rule was Art. 10(5) in the original Regulation (EU) 2024/1689. This study uses the dataset's sex field only to audit, which is not itself a production processing template.",
            "nist": "Map (proxy risk, the Apple Card question of fairness through unawareness); Measure (proxy AUC, correlations, and the full versus no-sex comparison).",
            "control": "Paired full, no-sex, and no-sex-no-age models; a sex classifier and an age classifier on the reduced feature sets; association tables.",
            "residual": "Whatever sex or age gap remains in the reduced models is the residual proxy disparity. The proxy AUC states how recoverable the attribute is.",
        },
        {
            "risk": "No individual explanation of a credit decision",
            "evidence": kwargs["shap_sentence"] + f" Local SHAP reason lists were produced for {kwargs['local_count']} deterministically chosen test clients.",
            "law": "Art. 86 right to explanation of individual decision-making for Annex III systems, except point 2; Art. 13 transparency and information to deployers; Art. 14 human oversight, including the ability to interpret outputs.",
            "nist": "Measure (SHAP); Manage (adverse-action style reason lists for a reviewer).",
            "control": "LinearExplainer for logistic regression, TreeExplainer for the tree, aggregated one-hot SHAP, waterfall plots, and a plain-language top-feature list.",
            "residual": "These are research explanations of a default model, not a legal adverse-action notice and not an explanation of a credit-limit decision. Rank correlation between the full and no-sex models is the measured statement of how influence moves.",
        },
        {
            "risk": "Training-data privacy exposure",
            "evidence": "Non-private logistic regression and the tree are ordinary empirical-risk minimizers. The privacy sweep records epsilon, clipping fractions, and the utility of diffprivlib objective perturbation. " + kwargs["rq3"],
            "law": "Art. 15 accuracy, robustness, and cybersecurity, including resilience to confidentiality attacks; Art. 10 data governance.",
            "nist": "Measure (privacy-utility curve); Manage (epsilon selection under a written rule).",
            "control": "Fixed data_norm, coordinate clipping, L2 row clipping, a fresh accountant per fit, and an epsilon grid that includes the non-private model on the same clipped matrix.",
            "residual": "The StandardScaler is fit on the private training split, so the stated epsilon does not cover preprocessing. Resampling for reweighing is also outside the accountant. Both limits are recorded in the privacy notes and the model card.",
        },
        {
            "risk": "Differential privacy changes group error rates",
            "evidence": kwargs["rq3"],
            "law": "Art. 15's accuracy requirement and Art. 10's bias controls have to be re-measured after a privacy control. Art. 9 requires the residual risk of each control to be judged explicitly.",
            "nist": "Measure (per-group AUC and fairness gaps across epsilon); Manage (do not treat a privacy setting as acceptable solely because overall AUC remains high).",
            "control": "The sweep records accuracy, ROC-AUC, sex and age gaps, and per-group AUC and accuracy at every epsilon and seed, with and without reweighing.",
            "residual": "The group-AUC changes in the RQ3 paragraph are the measured residual. They are not assumed to match the direction in Bagdasaryan et al.",
        },
        {
            "risk": "Explanation instability under differential privacy",
            "evidence": kwargs["rq3"],
            "law": "Art. 13 and Art. 86 assume a deployer can give a stable account of the main elements of a decision. An explanation that changes with the privacy noise is a transparency residual.",
            "nist": "Measure (Spearman correlation of mean absolute SHAP, top-k Jaccard overlap, and mean cosine similarity versus the non-private model).",
            "control": "explanation_stability is computed for every private model against the paired non-private model on a fixed evaluation sample.",
            "residual": "The Spearman and Jaccard values at the smallest and largest epsilon are the residual instability for this grid.",
        },
        {
            "risk": "Weak documentation and oversight",
            "evidence": f"Generated model card, this mapping, run metadata, and config hash are produced by the same command. Recommended setting: {kwargs['rec_text']}.",
            "law": "Art. 11 and Annex IV technical documentation; Art. 12 record-keeping; Art. 14 human oversight. The threshold is a research decision rule, not an automated lending decision.",
            "nist": "Govern (accountability and documentation) cuts across Map, Measure, and Manage.",
            "control": "One configuration file, fixed seeds, multi-seed mean and standard deviation, a model card, and a monitoring margin of "
            + "the configured EO and AUC margins if a later window is scored with the same metrics.",
            "residual": "This is a course audit of a public 2005 dataset. It is not a conformity assessment, a deployed system, or a substitute for human review of any person's application.",
        },
    ]


def manifest_payload(context: dict[str, Any]) -> dict[str, Any]:
    """JSON-ready provenance list."""
    return {"numbers": json_safe(context["manifest"])}


def verify_manifest(config: Config, items: list[dict[str, Any]] | None = None) -> None:
    """Re-read every registered cell and fail if the rendered value drifted."""
    import json

    if items is None:
        payload = json.loads((config.results_root / "tables" / "reported_numbers.json").read_text(encoding="utf-8"))
        items = payload["numbers"]
    checker = Registry()
    for item in items:
        filters = item["filters"]
        if isinstance(filters, str):
            import json
            filters = json.loads(filters)
        frame = pd.read_csv(config.results_root / item["source"])
        if "epsilon" in frame.columns:
            frame["epsilon"] = pd.to_numeric(frame["epsilon"], errors="coerce")
        if filters == {"row": 0}:
            got = frame.iloc[0][item["column"]]
        else:
            got = checker.one(item["key"], frame, filters, item["column"], item["source"])
        expected = item["value"]
        if expected is None or (isinstance(expected, float) and not np.isfinite(expected)):
            if not (got is None or (isinstance(got, float) and not np.isfinite(float(got)))):
                raise AssertionError(f"{item['key']} expected non-finite, found {got}")
            continue
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not np.isclose(float(got), float(expected), rtol=0.0, atol=1e-8):
                raise AssertionError(f"{item['key']} manifest {expected} != table {got}")
        elif str(got) != str(expected):
            raise AssertionError(f"{item['key']} manifest {expected} != table {got}")
