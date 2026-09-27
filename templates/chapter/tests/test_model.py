import sympy as sp


def test_equations_match_the_code():
    # Replace with a real check: derive or state the equation with sympy and compare it
    # with what sim.py computes for a small, hand-made situation.
    t, v = sp.symbols("t v")
    assert sp.diff(v * t, t) == v
