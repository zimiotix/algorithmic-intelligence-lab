"""Render a chapter's Manim explainer scenes to chapters/<track>/<slug>/media/*.mp4.

    uv sync --extra media        # installs Manim Community Edition (needs cairo/pango)
    uv run tools/render_media.py swarms.fish-school

Scenes live in the chapter's media/scenes.py. Videos are build output (gitignored);
the app shows a play button for each one in the Deep dive.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ailab.core.catalog import discover


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    info = discover().chapters[sys.argv[1]]
    scenes = info.media_dir / "scenes.py"
    if not scenes.exists():
        print(f"{scenes} not found")
        return 1
    quality = sys.argv[2] if len(sys.argv) > 2 else "-qh"
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-m", "manim", quality, "-a", str(scenes),
                            "--media_dir", tmp], check=False)
        if r.returncode:
            return r.returncode
        for mp4 in Path(tmp).rglob("*.mp4"):
            if "partial_movie_files" not in mp4.parts:
                shutil.copy(mp4, info.media_dir / mp4.name)
                print("wrote", info.media_dir / mp4.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
