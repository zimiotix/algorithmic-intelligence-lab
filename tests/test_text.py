from ailab.text.markdown import extract_math, split_sections, to_html


def test_math_is_extracted_but_code_is_left_alone():
    body, eqs = extract_math("Speed $v^2 = 2ad$ and `cost $5` and\n\n$$x_i^2$$\n")
    assert [e.tex for e in eqs] == ["v^2 = 2ad", "x_i^2"]
    assert [e.display for e in eqs] == [False, True]
    assert "`cost $5`" in body


def test_underscores_in_math_do_not_become_italics():
    html, eqs = to_html("a $x_i + y_i$ b", lambda e, i: f"[{e.tex}]")
    assert "[x_i + y_i]" in html and "<em>" not in html


def test_sections_split_on_level_one_headings():
    s = split_sections("# A\none\n## sub\n# B\ntwo")
    assert [t for t, _ in s] == ["A", "B"]
    assert "## sub" in s[0][1]


def test_builtin_renderer_draws_every_chapter_equation():
    """Without TeX (e.g. Windows) every equation must still come out at a sane size."""
    import re
    from pathlib import Path

    from ailab.text.latex import render_builtin

    root = Path(__file__).resolve().parents[1] / "chapters"
    for md in root.glob("*/*/*.md"):
        for eq in extract_math(md.read_text(encoding="utf-8"))[1]:
            svg = render_builtin(eq.tex)
            assert svg is not None, (md, eq.tex)
            height = float(re.search(r'height="([\d.]+)"', svg).group(1))
            assert 3 < height < 80, (md.parent.name, eq.tex, height)


def test_symbol_table_lists_params_and_glossary():
    from ailab.core.params import Param
    from ailab.text.markdown import symbol_table

    md = symbol_table([Param("r", "Radius", 1.5, 0, 5, symbol="r_r", unit="m"),
                       Param("n", "No symbol", 3, 0, 5)], None, [(r"\mathbf{x}", "Position")])
    assert "| $r_r$ | Radius | 1.5 m |" in md
    assert "$\\mathbf{x}$ | Position" in md
    assert "No symbol" not in md
    assert md.index("how to read them") < md.index("| Symbol")
