from ailab.core.catalog import discover


def test_catalog_loads_and_ids_are_unique():
    cat = discover()
    assert cat.chapters, "no chapters found"
    ids = [c.id for c in cat.ordered()]
    assert len(ids) == len(set(ids))


def test_every_chapter_has_its_files_and_three_levels():
    from ailab.text.markdown import split_sections

    for c in discover().ordered():
        for name in ("sim.py", "intro.md", "advanced.md"):
            assert (c.path / name).exists(), f"{c.id}: missing {name}"
        titles = [t for t, _ in split_sections(c.text("advanced.md"))]
        assert titles[:3] == ["Intuition", "The Math", "Research"], f"{c.id}: {titles}"
        assert 1 <= c.difficulty <= 5
        assert c.compute in ("cpu", "gpu")
        assert set(c.memory) <= {"environmental", "working", "episodic"}
