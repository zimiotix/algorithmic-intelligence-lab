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
