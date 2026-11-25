"""Feature sets and the train-only preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, feature_columns
from src.data import SplitData


@dataclass
class MatrixBundle:
    """Transformed matrices plus the original frames needed for evaluation."""

    feature_set: str
    feature_columns: list[str]
    feature_names: list[str]
    preprocessor: ColumnTransformer
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame


def categorical_columns(config: Config, columns: list[str]) -> list[str]:
    """Categorical inputs that are actually present in this feature set."""
    requested = [str(column) for column in config.features.categorical]
    return [column for column in columns if column in requested]


def build_preprocessor(columns: list[str], config: Config) -> ColumnTransformer:
    """One-hot education and marriage; standard-scale every other model input.

    The returned transformer is unfitted. Callers fit it on the training split only.
    """
    categorical = categorical_columns(config, columns)
    numeric = [column for column in columns if column not in categorical]
    encoder = OneHotEncoder(
        handle_unknown=str(config.features.one_hot.handle_unknown),
        sparse_output=False,
        dtype=np.float64,
    )
    transformers: list[tuple] = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", encoder, categorical))
    if not transformers:
        raise ValueError("Preprocessor received no columns")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def prepare_matrices(split: SplitData, feature_set: str, config: Config) -> MatrixBundle:
    """Fit preprocessing on the training split and transform all three splits."""
    columns = feature_columns(config, feature_set)
    preprocessor = build_preprocessor(columns, config)
    train = split.train
    validation = split.validation
    test = split.test
    preprocessor.fit(train[columns], train["default"].to_numpy())
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return MatrixBundle(
        feature_set=feature_set,
        feature_columns=columns,
        feature_names=names,
        preprocessor=preprocessor,
        X_train=np.asarray(preprocessor.transform(train[columns]), dtype=float),
        X_val=np.asarray(preprocessor.transform(validation[columns]), dtype=float),
        X_test=np.asarray(preprocessor.transform(test[columns]), dtype=float),
        y_train=train["default"].to_numpy(dtype=int),
        y_val=validation["default"].to_numpy(dtype=int),
        y_test=test["default"].to_numpy(dtype=int),
        train_frame=train,
        val_frame=validation,
        test_frame=test,
    )


def parent_feature(transformed_name: str, categorical: list[str]) -> str:
    """Map a one-hot column back to its source feature."""
    for column in categorical:
        if transformed_name.startswith(column + "_"):
            return column
    return transformed_name


def aggregate_columns(
    values: np.ndarray,
    names: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Sum SHAP or coefficient columns that belong to the same original feature."""
    array = np.asarray(values, dtype=float)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    if array.shape[1] != len(names):
        raise ValueError("Column count does not match feature names")
    parents = [parent_feature(name, categorical) for name in names]
    order: list[str] = []
    for parent in parents:
        if parent not in order:
            order.append(parent)
    grouped = [array[:, [index for index, parent in enumerate(parents) if parent == name]].sum(axis=1) for name in order]
    stacked = np.column_stack(grouped) if grouped else np.empty((array.shape[0], 0))
    if single:
        return stacked.ravel(), order
    return stacked, order
