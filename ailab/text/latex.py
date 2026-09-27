"""LaTeX -> SVG with the system TeX install (latex + dvisvgm), cached by content hash.

All uncached equations of a page are rendered in ONE latex run (one page each, via the
`preview` package), so opening a chapter costs one TeX invocation at most. Results live
in ~/.cache/ailab/eq and are never committed.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ailab" / "eq"
COLOR = "E6EDF3"
PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath,amssymb,bm}
\usepackage{xcolor}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{1.5pt}
\begin{document}
\color[HTML]{%s}
"""


@dataclass(frozen=True)
class Equation:
    tex: str
    display: bool

    @property
    def key(self) -> str:
        h = hashlib.sha1(f"{self.display}|{COLOR}|{self.tex}".encode()).hexdigest()
        return h[:16]

    @property
    def path(self) -> Path:
        return CACHE / f"{self.key}.svg"


def available() -> bool:
    return shutil.which("latex") is not None and shutil.which("dvisvgm") is not None


def _body(eq: Equation) -> str:
    style = r"\displaystyle " if eq.display else ""
    return r"\begin{preview}$" + style + eq.tex + r"$\end{preview}" + "\n"


def _run(eqs: list[Equation], workdir: Path) -> bool:
    tex = workdir / "eq.tex"
    tex.write_text(PREAMBLE % COLOR + "".join(_body(e) for e in eqs) + r"\end{document}")
    r = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"],
                       cwd=workdir, capture_output=True, timeout=60)
    if r.returncode != 0:
        return False
    r = subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "--page=1-", "-o",
                        "eq-%p.svg", "eq.dvi"], cwd=workdir, capture_output=True, timeout=60)
    if r.returncode != 0:
        return False
    pages = sorted(workdir.glob("eq-*.svg"),
                   key=lambda p: int(re.search(r"(\d+)\.svg$", p.name).group(1)))
    if len(pages) != len(eqs):
        return False
    for eq, page in zip(eqs, pages, strict=True):
        shutil.move(page, eq.path)
    return True


def render(eqs: list[Equation]) -> dict[str, Path | None]:
    """Make sure every equation has an SVG. Returns key -> path (None if it failed)."""
    result: dict[str, Path | None] = {}
    todo = []
    for e in eqs:
        if e.path.exists():
            result[e.key] = e.path
        elif e.key not in {t.key for t in todo}:
            todo.append(e)
    if todo and available():
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            ok = _run(todo, Path(d))
        if not ok:  # isolate the broken one(s) so the rest still render
            for e in todo:
                with tempfile.TemporaryDirectory() as d:
                    _run([e], Path(d))
    for e in todo:
        result[e.key] = e.path if e.path.exists() else None
    return result
