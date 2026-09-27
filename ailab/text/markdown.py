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
