"""LaTeX -> SVG, cached by content hash.

With a TeX install (latex + dvisvgm: TeX Live, MiKTeX), all uncached equations of a page
are rendered in ONE latex run (one page each, via the `preview` package), so opening a
chapter costs one TeX invocation at most. Without one (typical on Windows), the pure-Python
`ziamath` renderer draws them instead: no install needed, slightly less polished.
Results live in ~/.cache/ailab/eq and are never committed. Set AILAB_NO_LATEX=1 to force
the built-in renderer.
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

from ..core.paths import cache_dir

CACHE = cache_dir() / "eq"
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
    """True when the system TeX toolchain is usable."""
    if os.environ.get("AILAB_NO_LATEX"):
        return False
    return shutil.which("latex") is not None and shutil.which("dvisvgm") is not None


def backend() -> str:
    return "TeX" if available() else "built-in"


BUILTIN_SIZE = 9.5   # ziamath px size that matches TeX's 10 pt output after the view's scaling


def _fallback_path(eq: Equation) -> Path:
    return CACHE / f"{eq.key}.z.svg"


def render_builtin(tex: str, size: float = BUILTIN_SIZE) -> str | None:
    """SVG text for one equation from the pure-Python renderer (None if it can't)."""
    try:
        import ziamath as zm
    except ImportError:
        return None
    zm.config.svg2 = False   # SVG 1.1 without <use>/<symbol>: what QtSvg understands
    # ziamath stretches \Vert-style norms to absurd heights; plain \| renders right.
    tex = re.sub(r"\\(?:left|right)?\\?(?:lVert|rVert|Vert)\b|\\(?:left|right)\\\|", r"\\|", tex)
    for attempt in (tex, _fixed_fences(tex)):
        try:
            svg = zm.Latex(attempt, size=size, color=f"#{COLOR}").svg()
        except Exception:
            continue
        m = re.search(r'height="([\d.]+)"', svg)
        if m and float(m.group(1)) < 8 * size:
            return svg
        # A known ziamath quirk: \left...\right around a big operator inside a norm
        # grows without bound. Fixed-size \Bigg fences are the retry.
    return None


def _fixed_fences(tex: str) -> str:
    tex = re.sub(r"\\(?:left|right)\s*\.", "", tex)
    return re.sub(r"\\(left|right)", r"\\Bigg", tex)


def _body(eq: Equation) -> str:
    style = r"\displaystyle " if eq.display else ""
    return r"\begin{preview}$" + style + eq.tex + r"$\end{preview}" + "\n"


def _run(eqs: list[Equation], workdir: Path) -> bool:
    tex = workdir / "eq.tex"
    tex.write_text(PREAMBLE % COLOR + "".join(_body(e) for e in eqs) + r"\end{document}",
                   encoding="utf-8")
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
        if e.path.exists():
            result[e.key] = e.path
            continue
        fb = _fallback_path(e)
        if not fb.exists():
            svg = render_builtin(e.tex, BUILTIN_SIZE if e.display else 8.6)
            if svg is not None:
                CACHE.mkdir(parents=True, exist_ok=True)
                fb.write_text(svg, encoding="utf-8")
        result[e.key] = fb if fb.exists() else None
    return result
