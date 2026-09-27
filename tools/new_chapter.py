"""Scaffold a chapter:  uv run tools/new_chapter.py <track> <slug> "Title"

Creates chapters/<track>/<slug>/ from templates/chapter with status = "draft".
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    track, slug, title = sys.argv[1:]
    tracks = tomllib.loads((ROOT / "chapters/tracks.toml").read_text(encoding="utf-8"))["tracks"]
    if track not in tracks:
        print(f"unknown track '{track}'. Known: {', '.join(tracks)}")
        return 1
    dest = ROOT / "chapters" / track / slug
    if dest.exists():
        print(f"{dest} already exists")
        return 1
    for src in (ROOT / "templates/chapter").rglob("*"):
        if src.is_file():
            out = dest / src.relative_to(ROOT / "templates/chapter")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(src.read_text(encoding="utf-8").replace("{title}", title), encoding="utf-8")
    print(f"created {dest.relative_to(ROOT)}  (id: {track}.{slug})")
    print(f"try it:  uv run ailab --chapter {track}.{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
