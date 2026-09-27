"""Chapter markdown -> HTML for the native text view, with $inline$ and $$display$$ math.

Math is pulled out before markdown parsing (so `_` and `*` inside equations survive),
replaced with image placeholders, then put back as <img src="eq:KEY">.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from .latex import Equation

_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)
_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE = re.compile(r"(?<![\\$\w])\$(?![\s$])([^$\n]+?)(?<![\s\\])\$(?![\w$])")

_md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable(["table",
                                                                           "strikethrough"])


def extract_math(text: str) -> tuple[str, list[Equation]]:
    eqs: list[Equation] = []

    def block(m):
        e = Equation(m.group(1).strip(), True)
        eqs.append(e)
        return f"\n\n@@EQB:{len(eqs) - 1}@@\n\n"

    def inline(m):
        e = Equation(m.group(1).strip(), False)
        eqs.append(e)
        return f"@@EQI:{len(eqs) - 1}@@"

    out = []
    for part in _CODE.split(text):
        if part.startswith("`"):
            out.append(part)
        else:
            out.append(_INLINE.sub(inline, _BLOCK.sub(block, part)))
    return "".join(out), eqs


def to_html(text: str, eq_html) -> tuple[str, list[Equation]]:
    """eq_html(equation, index) -> HTML snippet for that equation."""
    body, eqs = extract_math(text)
    html = _md.render(body)
    html = re.sub(r"<p>@@EQB:(\d+)@@</p>",
                  lambda m: eq_html(eqs[int(m.group(1))], int(m.group(1))), html)
    html = re.sub(r"@@EQ[BI]:(\d+)@@",
                  lambda m: eq_html(eqs[int(m.group(1))], int(m.group(1))), html)
    return html, eqs


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on top-level '# ' headings -> [(title, body)]."""
    sections: list[tuple[str, str]] = []
    title, buf = None, []
    for line in text.splitlines():
        if line.startswith("# "):
            if title is not None:
                sections.append((title, "\n".join(buf).strip()))
            title, buf = line[2:].strip(), []
        else:
            buf.append(line)
    if title is not None:
        sections.append((title, "\n".join(buf).strip()))
    elif text.strip():
        sections.append(("Notes", text.strip()))
    return sections


NOTATION = r"""> **Before the equations: how to read them.** A **bold** letter is a vector, an arrow
> with a length and a direction (like $\mathbf{v}$, a velocity). A hat means "direction
> only", an arrow of length 1 (like $\hat{\mathbf{h}}$, a heading). $\lVert\mathbf{a}\rVert$
> is the length of $\mathbf{a}$. Subscript $i$ means "this agent" and $j$ "another one".
> $\Delta t$ is one simulation step (1/60 s), and $\leftarrow$ means "becomes, this step".
> Every other symbol is in the table below; the numbers are the ones running in the Lab.
"""


def symbol_table(params, values=None, glossary=()) -> str:
    """Markdown: the notation primer plus a table of every symbol on the Math page.

    ``params`` are the chapter's Param specs (those with a ``symbol``), ``values`` their
    current values (a ``Values`` object or dict), ``glossary`` extra (TeX, meaning) pairs."""
    rows = []
    for tex, meaning in glossary:
        rows.append(f"| ${tex}$ | {meaning} | |")
    for p in params:
        if not p.symbol:
            continue
        v = p.default if values is None else values.get(p.key)
        if isinstance(v, float):
            v = f"{v:g}"
        unit = f" {p.unit}" if p.unit and p.unit != "deg" else ("°" if p.unit == "deg" else "")
        what = p.label + (f": {p.help[0].lower()}{p.help[1:].rstrip('.')}" if p.help else "")
        rows.append(f"| ${p.symbol}$ | {what} | {v}{unit} |")
    if not rows:
        return NOTATION
    return (NOTATION + "\n| Symbol | Meaning | In the Lab now |\n|---|---|---|\n"
            + "\n".join(rows) + "\n")
