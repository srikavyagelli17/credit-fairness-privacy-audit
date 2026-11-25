"""Run the audit pipeline and write the generated documents."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_ROOT, ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_ROOT, ".cache"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.path.join(_ROOT, ".joblib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.audit import run_audit
from src.config import load_config, with_runtime
from src.data import run_data
from src.explain import run_explain
from src.findings import build_context
from src.governance import render_governance
from src.model_card import render_model_card
from src.models import run_baseline
from src.privacy import run_privacy
from src.reweighing import run_reweighing
from src.utils import ROOT, get_logger, setup_logging, write_json

STAGES = ("data", "baseline", "audit", "reweigh", "explain", "dp", "governance", "card", "all")
PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "xgboost",
    "fairlearn",
    "shap",
    "diffprivlib",
    "matplotlib",
    "seaborn",
)


def _git_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def _print_findings(config) -> None:
    context = build_context(config)
    print("\nRQ1\n" + context["rq1"] + "\n")
    print("RQ2\n" + context["rq2"] + "\n")
    print("RQ3\n" + context["rq3"] + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m src.run_all``."""
    parser = argparse.ArgumentParser(description="Credit default fairness, explanation, and privacy audit")
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--seeds", type=int, default=None, help="Use the first N configured seeds")
    parser.add_argument("--sample", type=int, default=None, help="Stratified subsample size")
    parser.add_argument("--fast", action="store_true", help="Smaller grid, fewer seeds, fewer bootstraps")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config = with_runtime(
        load_config(args.config),
        fast=args.fast,
        n_seeds=args.seeds,
        sample=args.sample,
        results_root=args.results_root,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(config.results_root / "logs" / f"run_{stamp}.log")
    logger.info("Stage=%s fast=%s sample=%s seeds=%s", args.stage, args.fast, args.sample, list(config.seeds))
    write_json(config.results_root / "logs" / "resolved_config.json", config.as_dict())

    stages = ["data", "baseline", "audit", "reweigh", "explain", "dp", "governance", "card"] if args.stage == "all" else [args.stage]
    durations: dict[str, float] = {}
    try:
        for stage in stages:
            t0 = time.perf_counter()
            logger.info("Starting %s", stage)
            if stage == "data":
                run_data(config, refresh=args.refresh_data)
            elif stage == "baseline":
                run_baseline(config)
            elif stage == "audit":
                run_audit(config)
            elif stage == "reweigh":
                run_reweighing(config)
            elif stage == "explain":
                run_explain(config)
            elif stage == "dp":
                run_privacy(config)
            elif stage == "governance":
                render_governance(config)
            elif stage == "card":
                render_model_card(config)
            durations[stage] = time.perf_counter() - t0
            logger.info("Finished %s in %.1f seconds", stage, durations[stage])
        if args.stage == "all":
            _print_findings(config)
    except Exception:
        logger.error("Pipeline failed\n%s", traceback.format_exc())
        raise
    finally:
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": _git_hash(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _versions(),
            "config_hash": config.canonical_hash(),
            "config_path": str(config.source_path),
            "argv": sys.argv if argv is None else ["python -m src.run_all", *argv],
            "stage": args.stage,
            "fast": args.fast,
            "sample": args.sample,
            "seeds": [int(seed) for seed in config.seeds],
            "stage_seconds": durations,
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(config.results_root / "run_metadata.json", metadata)
        logger.info("Runtime %.1f seconds", metadata["runtime_seconds"])


if __name__ == "__main__":
    main()
