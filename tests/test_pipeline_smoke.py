"""The fast sample pipeline writes the tables and documents the study depends on."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg

from src.findings import fmt, verify_manifest
from src.utils import ROOT

EXPECTED = [
    "tables/dataset_summary.csv",
    "tables/baseline_performance.csv",
    "tables/fairness_baseline.csv",
    "tables/proxy_analysis.csv",
    "tables/reweighing_comparison.csv",
    "tables/reweighing_summary.csv",
    "tables/dp_sweep.csv",
    "tables/dp_sweep_summary.csv",
    "tables/recommended_config.csv",
    "tables/local_explanations.csv",
    "tables/reported_numbers.json",
    "tables/shap_stability.csv",
    "figures/approval_rate_by_sex.png",
    "figures/approval_rate_by_age.png",
    "figures/tpr_fpr_by_sex.png",
    "figures/proxy_features_sex.png",
    "figures/fairness_accuracy_tradeoff.png",
    "figures/dp_auc_vs_epsilon.png",
    "figures/dp_fairness_vs_epsilon.png",
    "figures/dp_group_auc_vs_epsilon.png",
    "figures/dp_shap_stability_vs_epsilon.png",
    "figures/dp_shap_rank_heatmap.png",
    "docs/model_card.md",
    "docs/governance_mapping.md",
    "README.md",
    "run_metadata.json",
]


def test_fast_sample_pipeline(tmp_path: Path) -> None:
    results = tmp_path / "results"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_all",
            "--stage",
            "all",
            "--sample",
            "2000",
            "--fast",
            "--results-root",
            str(results),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:])
    for relative in EXPECTED:
        path = results / relative
        assert path.exists(), relative
        if path.suffix == ".png":
            image = mpimg.imread(path)
            assert image.shape[0] > 50 and image.shape[1] > 50
    shap_plots = list((results / "figures").glob("shap_*.png"))
    assert shap_plots, "expected SHAP figures"
    card = (results / "docs" / "model_card.md").read_text(encoding="utf-8")
    readme = (results / "README.md").read_text(encoding="utf-8")
    governance = (results / "docs" / "governance_mapping.md").read_text(encoding="utf-8")
    for text in (card, readme, governance):
        assert "{{" not in text
        assert "{%" not in text
    from src.config import load_config, with_runtime

    config = with_runtime(load_config(), fast=True, sample=2000, results_root=results)
    verify_manifest(config)
    # Headline floats in the README use the same four-decimal formatter as the manifest.
    import json

    payload = json.loads((results / "tables" / "reported_numbers.json").read_text(encoding="utf-8"))
    for item in payload["numbers"]:
        if not item.get("headline"):
            continue
        value = item["value"]
        if isinstance(value, float):
            assert fmt(value) in readme
        elif isinstance(value, int) and abs(value) >= 100:
            assert str(value) in readme
