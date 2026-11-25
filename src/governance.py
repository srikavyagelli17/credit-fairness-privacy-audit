"""Render the EU AI Act and NIST AI RMF mapping from result tables."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import Config, docs_dir
from src.findings import build_context, manifest_payload
from src.utils import ROOT, write_json


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render_governance(config: Config, context: dict | None = None) -> dict:
    """Write ``docs/governance_mapping.md`` and return the template context."""
    context = build_context(config) if context is None else context
    text = _environment().get_template("governance_mapping.md.j2").render(**context)
    if "{{" in text or "}}" in text:
        raise RuntimeError("Governance mapping still contains a template marker")
    path = docs_dir(config) / "governance_mapping.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    write_json(config.results_root / "tables" / "reported_numbers.json", manifest_payload(context))
    return context
