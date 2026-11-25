"""Download, clean, and split the UCI Default of Credit Card Clients data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config, feature_columns, project_path
from src.utils import get_logger, save_table

logger = get_logger()


@dataclass
class SplitData:
    """One stratified train, validation, and test partition."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int


def _alias_map(config: Config) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in config.data.aliases.as_dict().items():
        mapping[str(canonical).strip().lower()] = str(canonical)
        for alias in aliases:
            mapping[str(alias).strip().lower()] = str(canonical)
    return mapping


def standardize_columns(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Rename UCI or spreadsheet columns to the canonical schema and drop IDs."""
    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    aliases = _alias_map(config)
    renamed.columns = [aliases.get(column.lower(), column) for column in renamed.columns]
    drop = [column for column in config.data.drop_columns if column in renamed.columns]
    if drop:
        renamed = renamed.drop(columns=drop)
    renamed = renamed.loc[:, ~renamed.columns.duplicated()]
    required = list(config.features.base_columns) + ["default"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found {list(renamed.columns)}")
    return renamed


def _as_int_code(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} contains values that are not numeric")
    rounded = numeric.round()
    if not np.allclose(numeric.to_numpy(dtype=float), rounded.to_numpy(dtype=float)):
        raise ValueError(f"{name} contains non-integer codes")
    return rounded.astype(int)


def _counts(series: pd.Series) -> pd.Series:
    return series.value_counts(dropna=False).sort_index()


def clean_dataframe(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, str]:
    """Recode education, marriage, sex, and age. Return the frame and a text log.

    ``PAY_0`` is kept. The source schema has no ``PAY_1``.
    """
    cleaned = standardize_columns(frame, config)
    before = {
        "EDUCATION": _counts(_as_int_code(cleaned["EDUCATION"], "EDUCATION")),
        "MARRIAGE": _counts(_as_int_code(cleaned["MARRIAGE"], "MARRIAGE")),
        "SEX": _counts(_as_int_code(cleaned["SEX"], "SEX")),
    }
    for column in config.features.base_columns:
        if column in {"EDUCATION", "MARRIAGE", "SEX", "AGE"} or str(column).startswith("PAY_"):
            cleaned[column] = _as_int_code(cleaned[column], column)
        else:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["default"] = _as_int_code(cleaned["default"], "default")

    education_from = {int(value) for value in config.data.education_recode_from}
    education_to = int(config.data.education_recode_to)
    cleaned["EDUCATION"] = cleaned["EDUCATION"].where(~cleaned["EDUCATION"].isin(education_from), education_to)

    marriage_from = {int(value) for value in config.data.marriage_recode_from}
    marriage_to = int(config.data.marriage_recode_to)
    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].where(~cleaned["MARRIAGE"].isin(marriage_from), marriage_to)

    sex_map = {int(key): str(value) for key, value in config.data.sex_labels.as_dict().items()}
    unknown_sex = set(cleaned["SEX"].unique()) - set(sex_map)
    if unknown_sex:
        raise ValueError(f"Unexpected SEX codes {sorted(unknown_sex)}")
    cleaned["sex_label"] = cleaned["SEX"].map(sex_map)

    edges = [float(edge) for edge in config.data.age_bin_edges]
    labels = [str(label) for label in config.data.age_bin_labels]
    cleaned["age_group"] = pd.cut(
        cleaned["AGE"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    if cleaned["age_group"].isna().any():
        raise ValueError("AGE values fell outside configured bins")
    cleaned["age_group"] = cleaned["age_group"].astype(str)

    threshold = int(config.data.age_young_threshold)
    cleaned["age_young"] = (cleaned["AGE"] < threshold).astype(int)
    young_labels = {int(key): str(value) for key, value in config.data.age_young_labels.as_dict().items()}
    cleaned["age_young_label"] = cleaned["age_young"].map(young_labels)
    cleaned["sex_age"] = cleaned["sex_label"] + " | " + cleaned["age_group"]
    cleaned["sex_age_young"] = cleaned["sex_label"] + " | " + cleaned["age_young_label"]

    cleaned = cleaned.reset_index(drop=True)
    cleaned.insert(0, "row_id", cleaned.index.astype(int))

    if not set(cleaned["default"].unique()).issubset({0, 1}):
        raise ValueError("default must be binary 0/1")
    if int(cleaned.isna().sum().sum()) > 0:
        raise ValueError("Cleaned frame contains nulls")

    after = {
        "EDUCATION": _counts(cleaned["EDUCATION"]),
        "MARRIAGE": _counts(cleaned["MARRIAGE"]),
        "SEX": _counts(cleaned["SEX"]),
        "sex_label": _counts(cleaned["sex_label"]),
        "age_group": _counts(cleaned["age_group"]),
        "age_young_label": _counts(cleaned["age_young_label"]),
    }
    lines = [
        "Cleaning log for UCI Default of Credit Card Clients.",
        "PAY_0 is retained. The source schema numbers repayment status as PAY_0, PAY_2, ..., PAY_6 and has no PAY_1.",
        "EDUCATION codes {0, 5, 6} are mapped to 4 (other). Documented codes: 1 graduate school, 2 university, 3 high school, 4 other.",
        "MARRIAGE code 0 is mapped to 3 (other). Documented codes: 1 married, 2 single, 3 other.",
        "SEX codes: 1 male, 2 female. sex_label stores those words.",
        f"age_group bins are right-open cuts at {edges} with labels {labels}.",
        f"age_young is 1 when AGE < {threshold}.",
        "",
    ]
    for name, before_counts in before.items():
        lines.append(f"BEFORE {name}")
        lines.append(before_counts.to_string())
        lines.append("")
    for name, after_counts in after.items():
        lines.append(f"AFTER {name}")
        lines.append(after_counts.to_string())
        lines.append("")
    lines.append(f"rows={len(cleaned)}")
    lines.append(f"default_rate={float(cleaned['default'].mean()):.6f}")
    lines.append(f"nulls={int(cleaned.isna().sum().sum())}")
    return cleaned, "\n".join(lines) + "\n"


def _download_uci(config: Config) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=int(config.data.uci_id))
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    frame = pd.concat([features, targets], axis=1)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _read_fallback(config: Config) -> pd.DataFrame:
    raw_dir = project_path(config, str(config.paths.raw_dir))
    header = int(config.data.excel_header_row)
    tried: list[str] = []
    for name in config.data.fallback_filenames:
        path = raw_dir / str(name)
        tried.append(str(path))
        if not path.exists():
            continue
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path, header=header)
        return pd.read_csv(path)
    message = (
        "Could not download UCI dataset id "
        f"{int(config.data.uci_id)} and no fallback file was found.\n"
        "Place the UCI spreadsheet at one of:\n"
        + "\n".join(tried)
        + "\nThe spreadsheet header is on the second row; the ID column is dropped automatically.\n"
        "Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
    )
    raise FileNotFoundError(message)


def processed_dir(config: Config) -> Path:
    """Directory of the cleaned cache."""
    return project_path(config, str(config.paths.processed_dir))


def cache_paths(config: Config) -> tuple[Path, Path, Path]:
    """Parquet, cleaning log, and metadata paths."""
    directory = processed_dir(config)
    return (
        directory / str(config.paths.clean_filename),
        directory / str(config.paths.cleaning_log_filename),
        directory / str(config.paths.clean_meta_filename),
    )


def load_clean_frame(config: Config, refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned 30,000-row frame, downloading it on first use."""
    parquet_path, log_path, meta_path = cache_paths(config)
    version = int(config.data.cleaning_version)
    if parquet_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("cleaning_version", -1)) == version:
            logger.info("Loading cleaned cache %s", parquet_path)
            return pd.read_parquet(parquet_path)
    try:
        raw = _download_uci(config)
        logger.info("Downloaded UCI dataset id %s", int(config.data.uci_id))
    except Exception as exc:
        logger.warning("UCI download failed: %s", exc)
        raw = _read_fallback(config)
        logger.info("Loaded fallback file from %s", config.paths.raw_dir)
    cleaned, cleaning_log = clean_dataframe(raw, config)
    _assert_full_dataset(cleaned, config)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(parquet_path, index=False)
    log_path.write_text(cleaning_log, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"cleaning_version": version, "rows": int(len(cleaned))}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote cleaned cache %s (%s rows)", parquet_path, len(cleaned))
    return cleaned


def _assert_full_dataset(frame: pd.DataFrame, config: Config) -> None:
    expected = int(config.data.expected_rows)
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(frame)}")
    if int(frame.isna().sum().sum()) != 0:
        raise AssertionError("Cleaned dataset contains nulls")
    rate = float(frame["default"].mean())
    target = float(config.data.expected_default_rate)
    tolerance = float(config.data.default_rate_tolerance)
    if abs(rate - target) > tolerance:
        raise AssertionError(f"Default rate {rate:.4f} is outside {target} ± {tolerance}")


def apply_sample(frame: pd.DataFrame, sample: int | None, seed: int) -> pd.DataFrame:
    """Stratified subsample used by ``--sample``. ``None`` returns the full frame."""
    if sample is None or sample >= len(frame):
        return frame.copy()
    strata = frame["default"].astype(str) + "_" + frame["SEX"].astype(str)
    sampled, _ = train_test_split(
        frame,
        train_size=int(sample),
        stratify=strata,
        random_state=int(seed),
    )
    return sampled.reset_index(drop=True)


def analysis_frame(config: Config, refresh: bool = False) -> pd.DataFrame:
    """Cleaned frame, optionally subsampled according to the runtime config."""
    frame = load_clean_frame(config, refresh=refresh)
    sample = config.runtime.sample if "runtime" in config.as_dict() else None
    seed = int(config.seeds[0])
    sampled = apply_sample(frame, sample, seed)
    if sample is not None and len(sampled) != len(frame):
        logger.info("Using a stratified sample of %s rows (seed %s)", len(sampled), seed)
    return sampled


def make_split(frame: pd.DataFrame, seed: int, config: Config) -> SplitData:
    """Stratify on default × SEX. Fractions come from ``data.split``."""
    test_size = float(config.data.split.test)
    validation_size = float(config.data.split.validation)
    strata = frame["default"].astype(str) + "_" + frame["SEX"].astype(str)
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=strata,
        random_state=int(seed),
    )
    relative_validation = validation_size / (1.0 - test_size)
    strata_inner = train_val["default"].astype(str) + "_" + train_val["SEX"].astype(str)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        stratify=strata_inner,
        random_state=int(seed),
    )
    parts = (train, validation, test)
    ids = [set(part["row_id"].tolist()) for part in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise AssertionError("Split row_id sets overlap")
    if len(ids[0]) + len(ids[1]) + len(ids[2]) != len(frame):
        raise AssertionError("Split does not cover the frame")
    return SplitData(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        seed=int(seed),
    )


def summarize_dataset(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Row counts and default rates overall, by sex, and by age group."""
    rows: list[dict[str, Any]] = []

    def add(group_type: str, group: str, part: pd.DataFrame) -> None:
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "n": int(len(part)),
                "share": float(len(part) / len(frame)) if len(frame) else float("nan"),
                "default_rate": float(part["default"].mean()) if len(part) else float("nan"),
            }
        )

    add("overall", "all", frame)
    for group in config.fairness.group_order.sex_label:
        add("sex", str(group), frame[frame["sex_label"] == group])
    for group in config.fairness.group_order.age_group:
        add("age_group", str(group), frame[frame["age_group"] == group])
    return pd.DataFrame(rows)


def summarize_split(split: SplitData, config: Config) -> pd.DataFrame:
    """Group sizes and default rates inside one split."""
    frames = {"train": split.train, "validation": split.validation, "test": split.test}
    rows: list[dict[str, Any]] = []
    for name, part in frames.items():
        summary = summarize_dataset(part, config)
        summary.insert(0, "seed", split.seed)
        summary.insert(1, "split", name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_data(config: Config, refresh: bool = False) -> pd.DataFrame:
    """Build the modeling frame and write dataset summary tables and the cleaning log."""
    full = load_clean_frame(config, refresh=refresh)
    _assert_full_dataset(full, config)
    _, log_path, _ = cache_paths(config)
    results_log = config.results_root / "logs" / "data_cleaning.txt"
    results_log.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        results_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        results_log.write_text(
            "Cleaned cache was loaded without a stored before/after cleaning log.\n",
            encoding="utf-8",
        )
    frame = analysis_frame(config, refresh=False)
    tables = config.results_root / "tables"
    save_table(summarize_dataset(frame, config), tables / "dataset_summary.csv")
    save_table(summarize_dataset(full, config), tables / "dataset_summary_full.csv")
    primary = make_split(frame, int(config.seeds[0]), config)
    save_table(summarize_split(primary, config), tables / "split_summary.csv")
    logger.info(
        "Data stage: modeling rows=%s default_rate=%.4f",
        len(frame),
        float(frame["default"].mean()),
    )
    return frame
