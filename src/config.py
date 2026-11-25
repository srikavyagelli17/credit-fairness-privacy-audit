"""Load ``configs/config.yaml`` into an attribute-style config object."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from src.utils import ROOT

_RUNTIME_FIELDS = {"_data", "source_path", "root", "results_root"}


class Config:
    """Nested view of a YAML mapping.

    Dictionaries become child ``Config`` objects. Lists and scalars are returned
    as stored. Runtime fields (``root``, ``results_root``) live on the instance
    and are not part of the YAML mapping.
    """

    def __init__(self, data: dict[str, Any], source_path: Path, root: Path | None = None) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "root", root if root is not None else ROOT)
        object.__setattr__(self, "results_root", ROOT / "results")

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return Config(value, self.source_path, self.root)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        if key in _RUNTIME_FIELDS:
            object.__setattr__(self, key, value)
            return
        raise AttributeError("Config YAML values are immutable; use with_runtime()")

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of this node's mapping."""
        return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a raw stored value or ``default``."""
        if key not in self._data:
            return default
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value, self.source_path, self.root)
        return value

    def canonical_hash(self) -> str:
        """SHA-256 of the YAML mapping with sorted keys."""
        blob = yaml.safe_dump(self._data, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> Config:
    """Load the project config. Relative paths are resolved from the repo root."""
    config_path = ROOT / "configs" / "config.yaml" if path is None else Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = Config(payload, config_path, ROOT)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Reject inconsistent settings before any experiment runs."""
    splits = config.data.split.as_dict()
    total = float(splits["train"]) + float(splits["validation"]) + float(splits["test"])
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"data.split must sum to 1, got {total}")
    seeds = [int(seed) for seed in config.seeds]
    if not seeds:
        raise ValueError("seeds must be a non-empty list")
    if int(config.seed) != seeds[0]:
        raise ValueError("seed must equal seeds[0]")
    if str(config.threshold.method) not in {"f1", "fixed"}:
        raise ValueError("threshold.method must be f1 or fixed")
    if str(config.recommendation.rule) != "smallest_epsilon_within_auc_tolerance_and_eo_no_worse":
        raise ValueError("Unsupported recommendation.rule")
    if str(config.recommendation.fallback) != "highest_auc_with_eo_no_worse_else_nonprivate":
        raise ValueError("Unsupported recommendation.fallback")
    base = list(config.features.base_columns)
    for name, spec in config.features.sets.as_dict().items():
        missing = [col for col in spec.get("drop", []) if col not in base]
        if missing:
            raise ValueError(f"Feature set {name} drops unknown columns {missing}")
    for eps in config.privacy.epsilons:
        if float(eps) <= 0:
            raise ValueError("privacy.epsilons must be positive")


def with_runtime(
    config: Config,
    *,
    fast: bool = False,
    n_seeds: int | None = None,
    sample: int | None = None,
    results_root: str | Path | None = None,
) -> Config:
    """Return a copy with fast-mode and CLI overrides applied."""
    data = copy.deepcopy(config.as_dict())
    original_seeds = [int(seed) for seed in data["seeds"]]
    if fast:
        fast_cfg = data["fast"]
        data["seeds"] = original_seeds[: int(fast_cfg["n_seeds"])]
        data["fairness"]["n_bootstrap"] = int(fast_cfg["n_bootstrap"])
        data["privacy"]["epsilons"] = list(fast_cfg["epsilons"])
        data["models"]["tree"]["n_estimators"] = int(fast_cfg["tree_n_estimators"])
        data["explain"]["background_size"] = int(fast_cfg["explain_background_size"])
        data["explain"]["eval_size"] = int(fast_cfg["explain_eval_size"])
        data["privacy"]["max_iter"] = int(fast_cfg["privacy_max_iter"])
        data["privacy"]["shap_background_size"] = int(fast_cfg["privacy_shap_background_size"])
        data["privacy"]["shap_eval_size"] = int(fast_cfg["privacy_shap_eval_size"])
    if n_seeds is not None:
        if n_seeds < 1:
            raise ValueError("--seeds must be positive")
        if n_seeds > len(original_seeds):
            raise ValueError(f"Config defines {len(original_seeds)} seeds; cannot request {n_seeds}")
        data["seeds"] = original_seeds[:n_seeds]
    data["runtime"] = {
        "fast": bool(fast),
        "sample": None if sample is None else int(sample),
        "seeds": [int(seed) for seed in data["seeds"]],
    }
    resolved = Config(data, config.source_path, config.root)
    validate_config(resolved)
    if results_root is None:
        resolved.results_root = config.root / "results"
    else:
        root = Path(results_root)
        resolved.results_root = root if root.is_absolute() else config.root / root
    return resolved


def feature_columns(config: Config, set_name: str) -> list[str]:
    """Columns used as model inputs for a named feature set."""
    spec = config.features.sets.as_dict()[set_name]
    drop = set(spec.get("drop", []))
    columns = [str(col) for col in config.features.base_columns if col not in drop]
    if not columns:
        raise ValueError(f"Feature set {set_name} is empty")
    return columns


def project_path(config: Config, *parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return config.root.joinpath(*parts)


def results_path(config: Config, *parts: str) -> Path:
    """Resolve a path under the active results root."""
    return config.results_root.joinpath(*parts)


def docs_dir(config: Config) -> Path:
    """Model card and governance live in ``docs/`` for the default results root."""
    default = (config.root / "results").resolve()
    if config.results_root.resolve() == default:
        return config.root / "docs"
    return config.results_root / "docs"


def readme_path(config: Config) -> Path:
    """README path. Non-default results roots get their own README."""
    default = (config.root / "results").resolve()
    if config.results_root.resolve() == default:
        return config.root / "README.md"
    return config.results_root / "README.md"
