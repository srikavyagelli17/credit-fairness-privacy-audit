"""Shared I/O, logging, and formatting helpers."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(ROOT / ".joblib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


def ensure_runtime_dirs() -> None:
    """Create writable cache directories used by matplotlib and joblib."""
    for name in (".mplconfig", ".cache", ".joblib"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def slug(value: object) -> str:
    """Make a group label safe for column names."""
    text = str(value).strip().lower()
    text = text.replace("<", "lt").replace("+", "plus").replace("|", "_")
    text = text.replace(" ", "").replace("-", "_")
    return text


def json_safe(value: Any) -> Any:
    """Convert numpy values to JSON-safe objects. Non-finite floats become strings."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a dataframe as a GitHub-flavored markdown table."""
    columns = [str(col) for col in frame.columns]

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if math.isinf(number):
                return "inf" if number > 0 else "-inf"
            return f"{number:.{digits}f}"
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text

    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row[col]) for col in frame.columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, rule, *body]) + "\n"


def save_table(frame: pd.DataFrame, path: Path, digits: int = 4) -> None:
    """Write a CSV and a markdown twin with the same stem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".md").write_text(dataframe_to_markdown(frame, digits=digits), encoding="utf-8")


def setup_logging(log_path: Path) -> logging.Logger:
    """Log to the console and to ``log_path``."""
    ensure_runtime_dirs()
    logger = logging.getLogger("audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger() -> logging.Logger:
    """Return the project logger."""
    return logging.getLogger("audit")


def as_str_list(values: Iterable[Any]) -> list[str]:
    """Copy a config sequence into a list of strings."""
    return [str(value) for value in values]


def finite_float(value: Any) -> float:
    """Return a float, preserving NaN and inf."""
    if value is None:
        return float("nan")
    return float(value)


def quantile_ci(values: Sequence[float], ci_level: float) -> tuple[float, float]:
    """Percentile interval. Empty input returns NaN bounds."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha))
