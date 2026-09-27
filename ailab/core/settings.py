"""Tiny persistent preferences (~/.config/ailab/settings.json). Read before Qt starts,
because some choices (which GPU renders) must be made before the first GL context."""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ailab" / "settings.json"
DEFAULTS = {"device": "auto", "render_gpu": "system", "last_chapter": None}


def load() -> dict:
    try:
        return {**DEFAULTS, **json.loads(PATH.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(values: dict) -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps({**load(), **values}, indent=2))
    except OSError:
        pass
