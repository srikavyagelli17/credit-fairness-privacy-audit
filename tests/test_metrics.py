"""Known fairness gaps, the favorable-outcome ratio, and the privacy recommendation rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain import explanation_stability
from src.metrics import fairlearn_gaps, fairness_gaps
from src.privacy import select_recommended
from src.thresholds import apply_threshold, select_threshold
from src.config import load_config


def test_gaps_match_a_hand_built_example_and_fairlearn() -> None:
    # Group a approves 4 of 5 (0.80). Group b approves 2 of 5 (0.40).
    # Demographic parity uses those approval rates. The ratio of default predictions
    # is 0.20/0.60, which is a different number, so the favorable outcome is not
    # an unlabeled flip of the default class.
    y_true = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    groups = np.array(["a"] * 5 + ["b"] * 5)
    gaps = fairness_gaps(y_true, y_pred, groups)
    assert abs(gaps["dp_difference"] - 0.4) < 1e-12
    assert abs(gaps["dp_ratio"] - 0.5) < 1e-12
    default_selection_ratio = fairness_gaps(y_true, 1 - y_pred, groups)["dp_ratio"]
    assert abs(default_selection_ratio - (0.2 / 0.6)) < 1e-12
    assert abs(default_selection_ratio - gaps["dp_ratio"]) > 1e-6
    # TPR gap is 0.5 (0.5 versus 1.0). FPR gap is 1/3 (0 versus 1/3). EO is the larger gap.
    assert abs(gaps["eo_difference"] - 0.5) < 1e-12
    reference = fairlearn_gaps(y_true, y_pred, groups)
    assert abs(reference["dp_difference"] - gaps["dp_difference"]) < 1e-9
    assert abs(reference["dp_ratio"] - gaps["dp_ratio"]) < 1e-9
    assert abs(reference["eo_difference"] - gaps["eo_difference"]) < 1e-9


def test_threshold_separates_a_perfect_score() -> None:
    config = load_config()
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = select_threshold(y_true, proba, config)
    pred = apply_threshold(proba, threshold)
    assert list(pred) == [0, 0, 0, 1, 1, 1]


def test_explanation_stability_is_one_for_identical_values() -> None:
    values = np.array([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]], dtype=float)
    names = ["a", "b", "c"]
    result = explanation_stability(values, names, values.copy(), names, k=2)
    assert abs(result["spearman"] - 1.0) < 1e-12
    assert abs(result["jaccard_topk"] - 1.0) < 1e-12
    assert abs(result["mean_cosine"] - 1.0) < 1e-12


def test_recommendation_picks_the_smallest_epsilon_that_meets_both_rules() -> None:
    summary = pd.DataFrame(
        [
            {"epsilon": 0.1, "feature_set": "full", "reweighed": False, "roc_auc_mean": 0.70, "eo_difference_sex_mean": 0.05},
            {"epsilon": 1.0, "feature_set": "full", "reweighed": False, "roc_auc_mean": 0.80, "eo_difference_sex_mean": 0.04},
            {"epsilon": np.inf, "feature_set": "full", "reweighed": False, "roc_auc_mean": 0.81, "eo_difference_sex_mean": 0.04},
            {"epsilon": 0.5, "feature_set": "no_sex", "reweighed": False, "roc_auc_mean": 0.90, "eo_difference_sex_mean": 0.20},
            {"epsilon": np.inf, "feature_set": "no_sex", "reweighed": False, "roc_auc_mean": 0.91, "eo_difference_sex_mean": 0.04},
        ]
    )
    chosen, _candidates = select_recommended(summary, auc_tolerance=0.02, eo_slack=0.0)
    assert chosen["epsilon"] == 1.0
    assert chosen["feature_set"] == "full"
    assert chosen["fallback"] is False
