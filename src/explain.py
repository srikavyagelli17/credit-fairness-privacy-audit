"""Global and local SHAP explanations, plus explanation stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.features import aggregate_columns, categorical_columns
from src.utils import get_logger

logger = get_logger()


def explanation_stability(
    shap_a: np.ndarray,
    names_a: list[str],
    shap_b: np.ndarray,
    names_b: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Spearman rank correlation, top-k Jaccard overlap, and mean cosine similarity.

    Features are aligned by name. One-hot groups should already have been summed.
    """
    shared = [name for name in names_a if name in names_b]
    if len(shared) < 2:
        return {"spearman": float("nan"), "jaccard_topk": float("nan"), "mean_cosine": float("nan"), "n_shared": float(len(shared))}
    index_a = [names_a.index(name) for name in shared]
    index_b = [names_b.index(name) for name in shared]
    left = np.asarray(shap_a, dtype=float)[:, index_a]
    right = np.asarray(shap_b, dtype=float)[:, index_b]
    magnitude_a = np.mean(np.abs(left), axis=0)
    magnitude_b = np.mean(np.abs(right), axis=0)
    if np.allclose(magnitude_a, magnitude_a[0]) or np.allclose(magnitude_b, magnitude_b[0]):
        spearman = float("nan")
    else:
        spearman = float(stats.spearmanr(magnitude_a, magnitude_b).statistic)
    top = min(int(k), len(shared))
    set_a = set(np.argsort(-magnitude_a)[:top].tolist())
    set_b = set(np.argsort(-magnitude_b)[:top].tolist())
    jaccard = float(len(set_a & set_b) / len(set_a | set_b))
    # Cosine of the shared coordinates, row by row. Zero rows are excluded.
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    usable = (left_norm > 0) & (right_norm > 0)
    if not np.any(usable):
        mean_cosine = float("nan")
    else:
        dots = np.sum(left[usable] * right[usable], axis=1)
        mean_cosine = float(np.mean(dots / (left_norm[usable] * right_norm[usable])))
    return {
        "spearman": spearman,
        "jaccard_topk": jaccard,
        "mean_cosine": mean_cosine,
        "n_shared": float(len(shared)),
    }


def _sample_index(n_rows: int, size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    take = min(int(size), int(n_rows))
    return np.sort(generator.choice(n_rows, size=take, replace=False))


def _shap_values(model: Any, background: np.ndarray, query: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    import shap

    if kind == "lr":
        explainer = shap.LinearExplainer(model, background)
    elif kind == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(kind)
    values = explainer.shap_values(query)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    expected = np.asarray(explainer.expected_value, dtype=float).ravel()
    return values, float(expected[-1])


def _output_space(base: float, row_values: np.ndarray, proba: float) -> str:
    reconstructed = base + float(np.sum(row_values))
    proba = float(np.clip(proba, 1e-6, 1 - 1e-6))
    logit = float(np.log(proba / (1.0 - proba)))
    if abs(reconstructed - logit) <= abs(reconstructed - proba):
        return "log-odds of default"
    return "predicted default score"


def _readable_value(feature: str, value: Any, config: Config) -> str:
    if feature == "EDUCATION":
        labels = {int(k): str(v) for k, v in config.data.education_labels.as_dict().items()}
        return f"{int(value)} ({labels.get(int(value), 'code')})"
    if feature == "MARRIAGE":
        labels = {int(k): str(v) for k, v in config.data.marriage_labels.as_dict().items()}
        return f"{int(value)} ({labels.get(int(value), 'code')})"
    if feature == "SEX":
        labels = {int(k): str(v) for k, v in config.data.sex_labels.as_dict().items()}
        return f"{int(value)} ({labels.get(int(value), 'code')})"
    if str(feature).startswith("PAY_"):
        labels = {str(k): str(v) for k, v in config.data.pay_status_labels.as_dict().items()}
        code = str(int(value))
        suffix = labels.get(code)
        return f"{code} ({suffix})" if suffix else code
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _describe(feature: str, config: Config) -> str:
    descriptions = config.data.feature_descriptions.as_dict()
    return str(descriptions.get(feature, feature))


def _reason(feature: str, value: Any, shap_value: float, config: Config, space: str) -> str:
    direction = "increased" if shap_value > 0 else "decreased"
    return (
        f"{_describe(feature, config)} = {_readable_value(feature, value, config)} "
        f"{direction} the default score by {shap_value:+.3f} SHAP units ({space})."
    )


def _global_table(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    magnitude = np.mean(np.abs(values), axis=0)
    order = np.argsort(-magnitude)
    return pd.DataFrame(
        {
            "feature": [names[index] for index in order],
            "mean_abs_shap": magnitude[order],
            "rank": np.arange(1, len(names) + 1),
        }
    )


def _original_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    columns = []
    for name in names:
        if name in frame.columns:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.zeros(len(frame)))
    return np.column_stack(columns)


def _pick_clients(test: pd.DataFrame, proba: np.ndarray, pred: np.ndarray, threshold: float, embedding: np.ndarray) -> list[dict[str, Any]]:
    row_id = test["row_id"].to_numpy()
    sex = test["sex_label"].astype(str).to_numpy()
    chosen: list[int] = []

    def take(order: np.ndarray, mask: np.ndarray) -> int | None:
        for index in order:
            if mask[index] and int(index) not in chosen:
                return int(index)
        return None

    female_denied = (sex == "female") & (pred == 1)
    order = np.lexsort((row_id, -proba))
    denied = take(order, female_denied)
    if denied is None:
        denied = take(order, sex == "female")
        denied_type = "female_highest_risk"
    else:
        denied_type = "denied_female"
    if denied is None:
        raise RuntimeError("The test split has no female client")
    chosen.append(denied)

    male = sex == "male"
    approved_male = male & (pred == 0)
    neighbor_pool = approved_male if approved_male.any() else male
    neighbor_type = "male_neighbor_approved" if approved_male.any() else "male_neighbor_no_approval"
    distance = np.linalg.norm(embedding - embedding[denied], axis=1)
    order = np.lexsort((row_id, distance))
    neighbor = take(order, neighbor_pool)
    if neighbor is not None:
        chosen.append(neighbor)

    gap = np.abs(proba - threshold)
    borderline = take(np.lexsort((row_id, gap)), np.ones(len(test), dtype=bool))
    if borderline is not None:
        chosen.append(borderline)
    approval = take(np.lexsort((row_id, proba)), pred == 0)
    if approval is None:
        approval = take(np.lexsort((row_id, proba)), np.ones(len(test), dtype=bool))
        approval_type = "lowest_risk"
    else:
        approval_type = "clear_approval"
    if approval is not None:
        chosen.append(approval)
    kinds = []
    for index in chosen:
        if index == denied:
            kinds.append(denied_type)
        elif neighbor is not None and index == neighbor:
            kinds.append(neighbor_type)
        elif index == borderline:
            kinds.append("borderline")
        elif index == approval:
            kinds.append(approval_type)
    return [{"index": index, "case_type": kind} for index, kind in zip(chosen, kinds)]


def run_explain(config: Config) -> None:
    """Explain the primary-seed models and write SHAP tables and figures."""
    import shap

    from src.data import analysis_frame, make_split
    from src.models import bundle_path, load_bundle, positive_proba
    from src.plots import plot_shap_summary, plot_waterfall
    from src.thresholds import apply_threshold
    from src.utils import save_table

    del shap
    frame = analysis_frame(config)
    seed = int(config.seeds[0])
    split = make_split(frame, seed, config)
    specs = [
        (str(config.models.logistic_key), "full", "baseline", None, "lr"),
        (str(config.models.logistic_key), "full", "reweighed", "sex", "lr"),
        (str(config.models.logistic_key), "no_sex", "baseline", None, "lr"),
        (str(config.models.tree_key), "full", "baseline", None, "tree"),
    ]
    stored: dict[tuple, dict[str, Any]] = {}
    tables = config.results_root / "tables"
    figures = config.results_root / "figures"
    background_index = _sample_index(len(split.train), int(config.explain.background_size), seed + int(config.explain.background_seed_offset))
    eval_index = _sample_index(len(split.test), int(config.explain.eval_size), seed + int(config.explain.eval_seed_offset))

    for model_key, feature_set, mitigation, attribute, kind in specs:
        path = bundle_path(config.results_root / "models", model_key, feature_set, mitigation, seed, attribute)
        bundle = load_bundle(path)
        columns = list(bundle["feature_columns"])
        train_x = np.asarray(bundle["preprocessor"].transform(split.train[columns]), dtype=float)
        test_x = np.asarray(bundle["preprocessor"].transform(split.test[columns]), dtype=float)
        background = train_x[background_index]
        query = test_x[eval_index]
        values, base = _shap_values(bundle["model"], background, query, kind)
        categorical = categorical_columns(config, columns)
        aggregated, names = aggregate_columns(values, list(bundle["feature_names"]), categorical)
        label = f"{model_key}_{feature_set}_{mitigation}" + (f"_{attribute}" if attribute else "")
        global_table = _global_table(aggregated, names)
        global_table.insert(0, "model", label)
        save_table(global_table, tables / f"shap_global_{label}.csv")
        display = _original_matrix(split.test.iloc[eval_index], names)
        plot_shap_summary(
            aggregated,
            display,
            names,
            figures / f"shap_bar_{label}.png",
            config,
            f"Mean absolute SHAP by feature ({label})",
            "bar",
        )
        if mitigation == "baseline":
            plot_shap_summary(
                aggregated,
                display,
                names,
                figures / f"shap_beeswarm_{label}.png",
                config,
                f"SHAP summary ({label})",
                "dot",
            )
        proba = positive_proba(bundle["model"], query)
        space = _output_space(base, aggregated[0], float(proba[0]))
        stored[(model_key, feature_set, mitigation, attribute)] = {
            "values": aggregated,
            "names": names,
            "base": base,
            "space": space,
            "bundle": bundle,
            "test_x": test_x,
            "label": label,
        }
        logger.info("SHAP %s in %s", label, space)

    stability_rows = []
    pairs = [
        (
            (str(config.models.logistic_key), "full", "baseline", None),
            (str(config.models.logistic_key), "full", "reweighed", "sex"),
            "baseline_vs_reweigh_sex",
        ),
        (
            (str(config.models.logistic_key), "full", "baseline", None),
            (str(config.models.logistic_key), "no_sex", "baseline", None),
            "full_vs_no_sex",
        ),
    ]
    for left_key, right_key, name in pairs:
        left = stored[left_key]
        right = stored[right_key]
        stats_row = explanation_stability(left["values"], left["names"], right["values"], right["names"], k=int(config.explain.top_k))
        stats_row["comparison"] = name
        stability_rows.append(stats_row)
    save_table(pd.DataFrame(stability_rows), tables / "shap_stability.csv")

    full = stored[(str(config.models.logistic_key), "full", "baseline", None)]
    bundle = full["bundle"]
    columns = list(bundle["feature_columns"])
    test_x = full["test_x"]
    proba = positive_proba(bundle["model"], test_x)
    pred = apply_threshold(proba, float(bundle["threshold"]))
    neighbor_columns = [col for col in feature_columns_no_protected(config) if col in split.test.columns]
    from src.features import build_preprocessor

    neighbor_prep = build_preprocessor(neighbor_columns, config)
    neighbor_prep.fit(split.train[neighbor_columns])
    neighbor_space = np.asarray(neighbor_prep.transform(split.test[neighbor_columns]), dtype=float)
    clients = _pick_clients(split.test, proba, pred, float(bundle["threshold"]), neighbor_space)
    # SHAP for the selected clients, not only the eval subsample.
    client_index = np.array([client["index"] for client in clients])
    background = np.asarray(bundle["preprocessor"].transform(split.train[columns]), dtype=float)[
        _sample_index(len(split.train), int(config.explain.background_size), seed + int(config.explain.background_seed_offset))
    ]
    client_values, base = _shap_values(bundle["model"], background, test_x[client_index], "lr")
    categorical = categorical_columns(config, columns)
    client_agg, names = aggregate_columns(client_values, list(bundle["feature_names"]), categorical)
    import shap

    local_rows = []
    top_n = int(config.explain.local_top_features)
    for position, client in enumerate(clients):
        index = client["index"]
        row = split.test.iloc[index]
        values = client_agg[position]
        space = full["space"]
        order_up = np.argsort(-values)[:top_n]
        order_down = np.argsort(values)[:top_n]
        adverse = []
        for direction, order in (("increases_risk", order_up), ("decreases_risk", order_down)):
            for rank, feature_index in enumerate(order, start=1):
                feature = names[feature_index]
                raw_value = row[feature] if feature in row.index else float("nan")
                text = _reason(feature, raw_value, float(values[feature_index]), config, space)
                if direction == "increases_risk":
                    adverse.append(text)
                local_rows.append(
                    {
                        "case_type": client["case_type"],
                        "row_id": int(row["row_id"]),
                        "sex_label": row["sex_label"],
                        "age": int(row["AGE"]),
                        "age_group": row["age_group"],
                        "y_true": int(row["default"]),
                        "y_pred": int(pred[index]),
                        "probability": float(proba[index]),
                        "threshold": float(bundle["threshold"]),
                        "direction": direction,
                        "rank": rank,
                        "feature": feature,
                        "feature_value": _readable_value(feature, raw_value, config),
                        "shap_value": float(values[feature_index]),
                        "shap_output_space": space,
                        "reason": text,
                        "adverse_action_text": " ".join(adverse) if direction == "increases_risk" and rank == top_n else "",
                    }
                )
        explanation = shap.Explanation(
            values=values,
            base_values=base,
            data=_original_matrix(split.test.iloc[[index]], names).ravel(),
            feature_names=names,
        )
        plot_waterfall(
            explanation,
            figures / f"shap_waterfall_{client['case_type']}.png",
            config,
            f"{client['case_type'].replace('_', ' ')} (row {int(row['row_id'])})",
            f"SHAP value ({space})",
        )
    local = pd.DataFrame(local_rows)
    # Fill the adverse-action sentence on every row of a case once it is complete.
    for case_type, part in local.groupby("case_type"):
        text = part.loc[part["direction"] == "increases_risk", "reason"].head(top_n)
        joined = " ".join(text.tolist())
        local.loc[local["case_type"] == case_type, "adverse_action_text"] = joined
    save_table(local, tables / "local_explanations.csv")


def feature_columns_no_protected(config: Config) -> list[str]:
    """Model inputs used for the nearest-neighbor profile, excluding sex and age."""
    from src.config import feature_columns

    return feature_columns(config, str(config.explain.neighbor_feature_set))
