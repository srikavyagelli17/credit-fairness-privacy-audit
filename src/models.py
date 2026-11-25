"""Model factories for logistic regression, the tree model, and private logistic regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.compat import load_diffprivlib_logistic_regression
from src.config import Config


def make_logistic(config: Config, seed: int) -> LogisticRegression:
    """Unweighted L2 logistic regression. ``class_weight`` stays at the sklearn default."""
    spec = config.models.logistic
    return LogisticRegression(
        C=float(spec.C),
        solver=str(spec.solver),
        max_iter=int(spec.max_iter),
        tol=float(spec.tol),
        random_state=int(seed),
    )


def make_tree(config: Config, seed: int) -> XGBClassifier:
    """XGBoost classifier with the fixed hyperparameters in config."""
    spec = config.models.tree
    return XGBClassifier(
        n_estimators=int(spec.n_estimators),
        max_depth=int(spec.max_depth),
        learning_rate=float(spec.learning_rate),
        subsample=float(spec.subsample),
        colsample_bytree=float(spec.colsample_bytree),
        min_child_weight=float(spec.min_child_weight),
        reg_lambda=float(spec.reg_lambda),
        objective=str(spec.objective),
        eval_metric=str(spec.eval_metric),
        tree_method=str(spec.tree_method),
        n_jobs=int(spec.n_jobs),
        verbosity=int(spec.verbosity),
        random_state=int(seed),
    )


def make_dp_logistic(config: Config, epsilon: float, seed: int) -> Any:
    """Differentially private logistic regression with a fresh infinite accountant.

    ``data_norm`` is the configured bound. A new accountant is passed on every
    call so successive fits do not share a global privacy budget.
    """
    model_cls, _warning = load_diffprivlib_logistic_regression()
    from diffprivlib.accountant import BudgetAccountant
    spec = config.privacy
    return model_cls(
        epsilon=float(epsilon),
        data_norm=float(spec.data_norm),
        max_iter=int(spec.max_iter),
        tol=float(spec.tol),
        C=float(spec.C),
        n_jobs=int(spec.n_jobs),
        random_state=int(seed),
        accountant=BudgetAccountant(),
    )


def make_estimator(config: Config, model_key: str, seed: int) -> Any:
    """Build the named estimator. ``model_key`` is ``lr`` or ``tree``."""
    if model_key == str(config.models.logistic_key):
        return make_logistic(config, seed)
    if model_key == str(config.models.tree_key):
        return make_tree(config, seed)
    raise ValueError(f"Unknown model key {model_key}")


def fit_estimator(model: Any, features: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    """Fit an estimator, passing sample weights only when they are provided."""
    if sample_weight is None:
        model.fit(features, target)
    else:
        model.fit(features, target, sample_weight=sample_weight)
    return model


def bundle_path(directory: Path, model_key: str, feature_set: str, mitigation: str, seed: int, attribute: str | None = None) -> Path:
    """File name for a saved preprocessor, model, and threshold."""
    if mitigation == "baseline":
        stem = f"{model_key}_{feature_set}_baseline_seed{seed}"
    else:
        stem = f"{model_key}_{feature_set}_reweigh_{attribute}_seed{seed}"
    return directory / f"{stem}.joblib"


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    """Persist a fitted model bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a model bundle written by :func:`save_bundle`."""
    return joblib.load(path)


def run_baseline(config: Config) -> None:
    """Train the primary-seed logistic and tree models and write performance."""
    from src.data import analysis_frame, make_split
    from src.features import prepare_matrices
    from src.metrics import performance_metrics
    from src.thresholds import apply_threshold, select_threshold
    from src.utils import get_logger, save_table

    logger = get_logger()
    frame = analysis_frame(config)
    seed = int(config.seeds[0])
    split = make_split(frame, seed, config)
    rows: list[dict[str, Any]] = []
    model_keys = [str(config.models.logistic_key), str(config.models.tree_key)]
    for feature_set in config.features.baseline_sets:
        prepared = prepare_matrices(split, str(feature_set), config)
        for model_key in model_keys:
            model = fit_estimator(make_estimator(config, model_key, seed), prepared.X_train, prepared.y_train)
            val_proba = positive_proba(model, prepared.X_val)
            threshold = select_threshold(prepared.y_val, val_proba, config)
            val_pred = apply_threshold(val_proba, threshold)
            test_proba = positive_proba(model, prepared.X_test)
            test_pred = apply_threshold(test_proba, threshold)
            metrics = performance_metrics(prepared.y_test, test_proba, test_pred)
            rows.append(
                {
                    "model": model_key,
                    "feature_set": feature_set,
                    "seed": seed,
                    **metrics,
                    "threshold": threshold,
                    "validation_f1": float(f1_score(prepared.y_val, val_pred, pos_label=1, zero_division=0)),
                    "n_train": int(len(prepared.y_train)),
                    "n_validation": int(len(prepared.y_val)),
                    "n_test": int(len(prepared.y_test)),
                }
            )
            save_bundle(
                bundle_path(config.results_root / "models", model_key, str(feature_set), "baseline", seed),
                {
                    "model": model,
                    "preprocessor": prepared.preprocessor,
                    "threshold": threshold,
                    "feature_names": prepared.feature_names,
                    "feature_columns": prepared.feature_columns,
                    "feature_set": feature_set,
                    "model_key": model_key,
                    "seed": seed,
                    "mitigation": "baseline",
                    "attribute": None,
                },
            )
            logger.info(
                "Baseline %s/%s seed %s ROC-AUC %.4f F1 %.4f threshold %.4f",
                model_key,
                feature_set,
                seed,
                metrics["roc_auc"],
                metrics["f1"],
                threshold,
            )
    save_table(pd.DataFrame(rows), config.results_root / "tables" / "baseline_performance.csv")


def positive_proba(model: Any, features: np.ndarray) -> np.ndarray:
    """Predicted probability of the default class (label 1)."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim == 1:
            return probabilities
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return probabilities[:, -1]
    scores = np.asarray(model.decision_function(features), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-scores))
