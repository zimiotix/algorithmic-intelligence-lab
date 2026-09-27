"""Tiny persistent preferences (~/.config/ailab/settings.json). Read before Qt starts,
because some choices (which GPU renders) must be made before the first GL context."""

from __future__ import annotations

import json

from .paths import config_dir

PATH = config_dir() / "settings.json"
DEFAULTS = {"device": "auto", "render_gpu": "system", "last_chapter": None}


def load() -> dict:
    try:
        return {**DEFAULTS, **json.loads(PATH.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(values: dict) -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps({**load(), **values}, indent=2), encoding="utf-8")
    except OSError:
        pass
