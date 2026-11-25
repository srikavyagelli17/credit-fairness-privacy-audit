"""Render the model card and the README from result tables."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import Config, docs_dir, readme_path
from src.findings import build_context
from src.governance import render_governance
from src.utils import ROOT, get_logger

logger = get_logger()


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def _write(template_name: str, path, context: dict) -> None:
    text = _environment().get_template(template_name).render(**context)
    if "{{" in text or "{%" in text:
        raise RuntimeError(f"{path} still contains a template marker")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)


def render_model_card(config: Config, context: dict | None = None) -> dict:
    """Write the model card, README, and governance mapping."""
    context = render_governance(config, context)
    _write("model_card.md.j2", docs_dir(config) / "model_card.md", context)
    _write("README.md.j2", readme_path(config), context)
    return context
