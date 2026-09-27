"""Where the app keeps its own files, per OS conventions. Nothing here is ever in the repo.

Linux:   ~/.cache/ailab, ~/.config/ailab (XDG variables respected)
Windows: %LOCALAPPDATA%\\ailab\\cache, %APPDATA%\\ailab
macOS:   ~/Library/Caches/ailab, ~/Library/Application Support/ailab
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ailab" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ailab"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ailab"


def config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "ailab"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ailab"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ailab"
