"""Recoding, split leakage, and the full UCI row count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import feature_columns, load_config
from src.data import clean_dataframe, load_clean_frame, make_split
from src.features import aggregate_columns, build_preprocessor, prepare_matrices


def _toy_frame(config) -> pd.DataFrame:
    n_rows = 40
    columns = list(config.features.base_columns)
    frame = pd.DataFrame({column: np.zeros(n_rows) for column in columns})
    frame["SEX"] = np.tile([1, 2], n_rows // 2)
    frame["EDUCATION"] = np.tile([0, 1, 2, 3, 5, 6, 1, 2], n_rows // 8)
    frame["MARRIAGE"] = np.tile([0, 1, 2, 3], n_rows // 4)
    frame["AGE"] = np.tile([21, 29, 30, 44, 45, 60, 25, 33], n_rows // 8)
    frame["default"] = np.tile([0, 1], n_rows // 2)
    frame["LIMIT_BAL"] = np.arange(n_rows) * 1000
    return frame


def test_recoding_maps() -> None:
    config = load_config()
    cleaned, _log = clean_dataframe(_toy_frame(config), config)
    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert set(cleaned.loc[cleaned["SEX"] == 1, "sex_label"].unique()) == {"male"}
    assert set(cleaned.loc[cleaned["SEX"] == 2, "sex_label"].unique()) == {"female"}
    ages = dict(zip(cleaned["AGE"], cleaned["age_group"]))
    assert ages[29] == "<30"
    assert ages[30] == "30-44"
    assert ages[44] == "30-44"
    assert ages[45] == "45+"
    assert int(cleaned.loc[cleaned["AGE"] == 29, "age_young"].iloc[0]) == 1
    assert int(cleaned.loc[cleaned["AGE"] == 30, "age_young"].iloc[0]) == 0
    assert int(cleaned.isna().sum().sum()) == 0
    assert "PAY_0" in cleaned.columns
    assert "PAY_1" not in cleaned.columns


def test_feature_sets_and_split_are_disjoint() -> None:
    config = load_config()
    full = feature_columns(config, "full")
    no_sex = feature_columns(config, "no_sex")
    no_age = feature_columns(config, "no_sex_no_age")
    assert "SEX" in full and "SEX" not in no_sex and "AGE" not in no_age
    cleaned, _log = clean_dataframe(_toy_frame(config), config)
    # Duplicate the toy so every stratum has enough rows for two splits.
    cleaned = pd.concat([cleaned, cleaned], ignore_index=True)
    cleaned["row_id"] = np.arange(len(cleaned))
    split = make_split(cleaned, seed=int(config.seed), config=config)
    ids = [set(part["row_id"]) for part in (split.train, split.validation, split.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    prepared = prepare_matrices(split, "full", config)
    train_mean = float(split.train["LIMIT_BAL"].mean())
    # The scaled training column is centered on the training mean, not the full-frame mean.
    limit_index = prepared.feature_names.index("LIMIT_BAL")
    assert abs(float(prepared.X_train[:, limit_index].mean())) < 1e-8
    leaked = pd.DataFrame(split.test[prepared.feature_columns])
    leaked["LIMIT_BAL"] = 1_000_000
    transformed = prepared.preprocessor.transform(leaked)
    assert transformed.shape[1] == prepared.X_train.shape[1]
    assert abs(float(prepared.X_train[:, limit_index].mean()) - 0) < 1e-6
    assert float(np.mean(transformed[:, limit_index])) != train_mean


def test_one_hot_aggregation_sums_back_to_the_parent() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0]])
    names = ["LIMIT_BAL", "EDUCATION_1", "EDUCATION_2", "PAY_0"]
    aggregated, parents = aggregate_columns(values, names, ["EDUCATION"])
    assert parents == ["LIMIT_BAL", "EDUCATION", "PAY_0"]
    assert aggregated.tolist() == [[1.0, 5.0, 4.0]]
    preprocessor = build_preprocessor(["LIMIT_BAL", "EDUCATION"], load_config())
    assert "EDUCATION" in preprocessor.transformers[-1][2]


def test_full_uci_cache_has_expected_shape() -> None:
    config = load_config()
    frame = load_clean_frame(config)
    assert len(frame) == int(config.data.expected_rows)
    assert int(frame.isna().sum().sum()) == 0
    rate = float(frame["default"].mean())
    assert abs(rate - float(config.data.expected_default_rate)) <= float(config.data.default_rate_tolerance)
    assert set(frame["sex_label"].unique()) == {"female", "male"}
