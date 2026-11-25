"""Figures for the audit. Every saved axes has a title and axis labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config
from src.utils import ensure_runtime_dirs


def _apply_style(config: Config) -> None:
    ensure_runtime_dirs()
    sns.set_theme(style=str(config.plots.style))
    plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 11, "figure.autolayout": False})


def _finish(fig: plt.Figure, path: Path, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(config.plots.dpi), bbox_inches="tight")
    plt.close(fig)


def _require_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str, log_x: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if not ax.get_title() or not ax.get_xlabel() or not ax.get_ylabel():
        raise RuntimeError("A figure axis is missing a title or an axis label")


def plot_grouped_rates(
    frame: pd.DataFrame,
    path: Path,
    config: Config,
    *,
    value_col: str,
    low_col: str | None,
    high_col: str | None,
    title: str,
    ylabel: str,
    xlabel: str,
) -> None:
    """Grouped bars. ``series`` distinguishes models or feature sets; ``group`` is the protected group."""
    _apply_style(config)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    series = list(dict.fromkeys(frame["series"].tolist()))
    groups = list(dict.fromkeys(frame["group"].tolist()))
    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    for index, name in enumerate(series):
        part = frame[frame["series"] == name].set_index("group").reindex(groups)
        heights = part[value_col].to_numpy(dtype=float)
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=str(name))
        if low_col and high_col and low_col in part.columns:
            low = np.minimum(heights, part[low_col].to_numpy(dtype=float))
            high = np.maximum(heights, part[high_col].to_numpy(dtype=float))
            yerr = np.vstack([np.clip(heights - low, 0, None), np.clip(high - heights, 0, None)])
            ax.errorbar(x + offset, heights, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.set_xticks(x, [str(group) for group in groups])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    _require_labels(ax, title, xlabel, ylabel)
    _finish(fig, path, config)


def plot_tpr_fpr(frame: pd.DataFrame, path: Path, config: Config, title: str, xlabel: str) -> None:
    """Side-by-side TPR and FPR bars for one series."""
    _apply_style(config)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, metric, label in zip(axes, ("tpr", "fpr"), ("True positive rate (default)", "False positive rate (default)")):
        groups = frame["group"].tolist()
        heights = frame[metric].to_numpy(dtype=float)
        ax.bar(np.arange(len(groups)), heights, color="C0")
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if low_col in frame.columns:
            low = np.minimum(heights, frame[low_col].to_numpy(dtype=float))
            high = np.maximum(heights, frame[high_col].to_numpy(dtype=float))
            yerr = np.vstack([np.clip(heights - low, 0, None), np.clip(high - heights, 0, None)])
            ax.errorbar(np.arange(len(groups)), heights, yerr=yerr, fmt="none", ecolor="black", capsize=2)
        ax.set_xticks(np.arange(len(groups)), [str(group) for group in groups], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        _require_labels(ax, label, xlabel, label)
    fig.suptitle(title)
    _finish(fig, path, config)


def plot_proxy_bars(frame: pd.DataFrame, path: Path, config: Config, title: str, xlabel: str) -> None:
    """Horizontal bars of association strength."""
    _apply_style(config)
    ordered = frame.sort_values("abs_value", ascending=True)
    label_column = "feature" if "feature" in ordered.columns else "name"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered[label_column].astype(str), ordered["abs_value"].to_numpy(dtype=float), color="C0")
    _require_labels(ax, title, xlabel, "Feature")
    _finish(fig, path, config)


def plot_tradeoff(frame: pd.DataFrame, path: Path, config: Config) -> None:
    """Scatter of ROC-AUC against equalized-odds difference, with arrows from baseline to reweighing."""
    _apply_style(config)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    baseline = frame[frame["mitigation"] == "baseline"]
    mitigated = frame[frame["mitigation"] == "reweighed"]
    for _, row in mitigated.iterrows():
        match = baseline[
            (baseline["model"] == row["model"])
            & (baseline["feature_set"] == row["feature_set"])
            & (baseline["protected_attribute"] == row["protected_attribute"])
        ]
        if match.empty:
            continue
        start = match.iloc[0]
        ax.annotate(
            "",
            xy=(row["roc_auc_mean"], row["eo_mean"]),
            xytext=(start["roc_auc_mean"], start["eo_mean"]),
            arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.8},
        )
    for mitigation, marker in (("baseline", "o"), ("reweighed", "s")):
        part = frame[frame["mitigation"] == mitigation]
        ax.scatter(part["roc_auc_mean"], part["eo_mean"], marker=marker, s=40, label=mitigation)
    ax.legend()
    _require_labels(
        ax,
        "Fairness-accuracy trade-off after reweighing",
        "ROC-AUC on the test set (mean over seeds)",
        "Equalized odds difference on the reweighed attribute (mean over seeds)",
    )
    _finish(fig, path, config)


def plot_epsilon_curves(
    frame: pd.DataFrame,
    path: Path,
    config: Config,
    *,
    value_col: str,
    std_col: str | None,
    series_col: str,
    title: str,
    ylabel: str,
) -> None:
    """Log-scaled epsilon curves. Dashed horizontal lines are the paired non-private means."""
    _apply_style(config)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labeled_private = False
    for name, part in frame.groupby(series_col, sort=True):
        part = part.sort_values("epsilon")
        finite = part[np.isfinite(part["epsilon"].to_numpy(dtype=float))]
        if finite.empty:
            continue
        line = ax.plot(
            finite["epsilon"].to_numpy(dtype=float),
            finite[value_col].to_numpy(dtype=float),
            marker="o",
            label=str(name),
        )[0]
        if std_col and std_col in finite.columns:
            mean = finite[value_col].to_numpy(dtype=float)
            std = finite[std_col].to_numpy(dtype=float)
            ax.fill_between(
                finite["epsilon"].to_numpy(dtype=float),
                mean - std,
                mean + std,
                color=line.get_color(),
                alpha=0.18,
            )
        private = part[~np.isfinite(part["epsilon"].to_numpy(dtype=float))]
        if not private.empty:
            ax.axhline(
                float(private.iloc[0][value_col]),
                color=line.get_color(),
                linestyle="--",
                linewidth=0.9,
                label="paired non-private" if not labeled_private else None,
            )
            labeled_private = True
    ax.legend(fontsize=8)
    _require_labels(ax, title, "Privacy budget ε (log scale)", ylabel, log_x=True)
    _finish(fig, path, config)


def plot_rank_heatmap(frame: pd.DataFrame, path: Path, config: Config, title: str) -> None:
    """Heatmap of SHAP ranks. Rank 1 is the largest mean absolute SHAP value."""
    _apply_style(config)
    pivot = frame.pivot(index="feature", columns="epsilon_label", values="rank")
    ordered = []
    if "non-private" in pivot.columns:
        ordered.append("non-private")
    finite = frame[np.isfinite(frame["epsilon"].to_numpy(dtype=float))].sort_values("epsilon")
    for label in finite["epsilon_label"].tolist():
        if label not in ordered:
            ordered.append(label)
    pivot = pivot[[col for col in ordered if col in pivot.columns]]
    rank_column = "non-private" if "non-private" in pivot.columns else pivot.columns[0]
    pivot = pivot.loc[pivot[rank_column].sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", ax=ax, cbar_kws={"label": "Rank"})
    _require_labels(ax, title, "Privacy budget ε", "Feature")
    _finish(fig, path, config)


def plot_shap_summary(
    values: np.ndarray,
    data: np.ndarray,
    names: list[str],
    path: Path,
    config: Config,
    title: str,
    plot_type: str,
) -> None:
    """Bar or beeswarm summary. SHAP's own labels are kept and a title is added."""
    import shap

    _apply_style(config)
    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(
        values,
        data,
        feature_names=names,
        plot_type=plot_type,
        show=False,
        max_display=min(20, len(names)),
    )
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title(title)
    if not ax.get_xlabel():
        ax.set_xlabel("mean |SHAP value|" if plot_type == "bar" else "SHAP value")
    if not ax.get_ylabel():
        ax.set_ylabel("Feature")
    _finish(fig, path, config)


def plot_waterfall(explanation: object, path: Path, config: Config, title: str, xlabel: str) -> None:
    """Save one SHAP waterfall."""
    import shap

    _apply_style(config)
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation, max_display=int(config.explain.waterfall_max_display), show=False)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title(title)
    if not ax.get_xlabel():
        ax.set_xlabel(xlabel)
    if not ax.get_ylabel():
        ax.set_ylabel("Feature")
    _finish(fig, path, config)
