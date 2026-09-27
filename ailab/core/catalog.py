"""Chapter discovery.

Layout (folder names carry no numbers; ordering lives in TOML so chapters can be
re-ordered without renames):

    chapters/tracks.toml                 # the tracks (parts of the curriculum)
    chapters/<track>/<slug>/chapter.toml # one chapter's metadata
    chapters/<track>/<slug>/sim.py       # defines `class Sim(Simulation)`
    chapters/<track>/<slug>/intro.md     # the story, high-school level
    chapters/<track>/<slug>/advanced.md  # "# Intuition", "# The Math", "# Research"

Only TOML is read at startup. ``sim.py`` is imported on first open, so hundreds of
chapters cost nothing until used.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS_DIR = ROOT / "chapters"


@dataclass
class Track:
    id: str
    title: str
    order: int
    color: str
    description: str = ""
    chapters: list[ChapterInfo] = field(default_factory=list)


@dataclass
class ChapterInfo:
    id: str                 # "<track>.<slug>", stable forever
    title: str
    track: str
    order: int
    path: Path
    summary: str = ""
    difficulty: int = 1     # 1..5
    tags: list[str] = field(default_factory=list)
    compute: str = "cpu"    # "gpu" = benefits from CUDA, "cpu" = light
    memory: list[str] = field(default_factory=list)   # memory systems used
    prerequisites: list[str] = field(default_factory=list)
    controls: list[tuple[str, str]] = field(default_factory=list)
    status: str = "ready"   # "ready" | "draft"
    reality: str = ""       # "model vs. nature": what the simulation simplifies
    glossary: list[tuple[str, str]] = field(default_factory=list)  # (TeX symbol, meaning)

    def text(self, name: str) -> str:
        p = self.path / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    @property
    def media_dir(self) -> Path:
        return self.path / "media"


@dataclass
class Catalog:
    tracks: list[Track]                 # tracks that have chapters, in order
    chapters: dict[str, ChapterInfo]
    planned: list[Track] = field(default_factory=list)   # announced, no chapters yet

    def ordered(self) -> list[ChapterInfo]:
        return [c for t in self.tracks for c in t.chapters]

    def track(self, track_id: str) -> Track:
        return next(t for t in self.tracks if t.id == track_id)


class CatalogError(ValueError):
    pass


def discover(root: Path = CHAPTERS_DIR, include_drafts: bool = True) -> Catalog:
    with open(root / "tracks.toml", "rb") as f:
        tdata = tomllib.load(f)
    tracks = {
        tid: Track(id=tid, title=t["title"], order=t.get("order", 999),
                   color=t.get("color", "#38bdf8"), description=t.get("description", ""))
        for tid, t in tdata.get("tracks", {}).items()
    }
    chapters: dict[str, ChapterInfo] = {}
    for toml_path in sorted(root.glob("*/*/chapter.toml")):
        folder = toml_path.parent
        track_id, slug = folder.parent.name, folder.name
        with open(toml_path, "rb") as f:
            d = tomllib.load(f)
        if track_id not in tracks:
            raise CatalogError(f"{toml_path}: unknown track '{track_id}' (add it to tracks.toml)")
        cid = f"{track_id}.{slug}"
        info = ChapterInfo(
            id=cid, title=d["title"], track=track_id, order=d.get("order", 999), path=folder,
            summary=d.get("summary", ""), difficulty=int(d.get("difficulty", 1)),
            tags=list(d.get("tags", [])), compute=d.get("compute", "cpu"),
            memory=list(d.get("memory", [])), prerequisites=list(d.get("prerequisites", [])),
            controls=[tuple(c) for c in d.get("controls", [])], status=d.get("status", "ready"),
            reality=" ".join(d.get("reality", "").split()),
            glossary=[(k, " ".join(str(v).split())) for k, v in d.get("glossary", {}).items()],
        )
        if info.status == "draft" and not include_drafts:
            continue
        chapters[cid] = info
        tracks[track_id].chapters.append(info)
    for t in tracks.values():
        t.chapters.sort(key=lambda c: (c.order, c.title))
    for c in chapters.values():
        missing = [p for p in c.prerequisites if p not in chapters]
        if missing:
            raise CatalogError(f"{c.id}: unknown prerequisites {missing}")
    ordered = sorted(tracks.values(), key=lambda t: (t.order, t.title))
    return Catalog(tracks=[t for t in ordered if t.chapters], chapters=chapters,
                   planned=[t for t in ordered if not t.chapters])


def load_sim_class(info: ChapterInfo):
    """Import the chapter's sim.py (once) and return its `Sim` class."""
    mod_name = "ailab_chapters." + info.id.replace("-", "_")
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, info.path / "sim.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {info.path / 'sim.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod   # Warp needs the module registered for kernels
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[mod_name]
            raise
    return mod.Sim
