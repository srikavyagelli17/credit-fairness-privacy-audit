"""Kamiran-Calders weights restore equal weighted positive rates and sum to n."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.reweighing import reweighing_weights, weighted_positive_rate


def test_weights_sum_to_n_and_balance_groups() -> None:
    config = load_config()
    group = np.array(["a"] * 100 + ["b"] * 100)
    labels = np.array([1] * 10 + [0] * 90 + [1] * 80 + [0] * 20)
    weights = reweighing_weights(labels, group)
    assert abs(float(weights.sum()) - len(labels)) <= float(config.numerical.weight_atol)
    rates = weighted_positive_rate(labels, group, weights)
    assert abs(rates["a"] - rates["b"]) <= float(config.numerical.weight_atol)
    assert abs(rates["a"] - float(labels.mean())) <= float(config.numerical.weight_atol)
    # The unweighted rates are not equal, so the test is not vacuous.
    raw_a = float(labels[:100].mean())
    raw_b = float(labels[100:].mean())
    assert abs(raw_a - raw_b) > 0.5
