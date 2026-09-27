import numpy as np
import sympy as sp

from ailab.core.memory import FieldMemory, TraceMemory, forget_np


def test_forgetting_is_exponential_decay():
    # d s/dt = -s/tau  =>  s(t) = s0 exp(-t/tau): the rule the code applies per step.
    t, tau, s0 = sp.symbols("t tau s0", positive=True)
    s = sp.Function("s")
    sol = sp.dsolve(sp.Eq(s(t).diff(t), -s(t) / tau), ics={s(0): s0}).rhs
    assert sp.simplify(sol - s0 * sp.exp(-t / tau)) == 0
    x = 1.0
    for _ in range(60):
        x = forget_np(x, 1 / 60, 2.0)
    assert np.isclose(x, np.exp(-1 / 2.0))


def test_evaporation_matches_closed_form():
    f = FieldMemory((0, 0, 10, 10), 1.0, tau=5.0)
    f.deposit_points(np.array([[5.5, 5.5]]), 1.0)
    f.update(0.0 + 1e-9)          # commit the deposit
    for _ in range(120):
        f.update(1 / 60)
    assert np.isclose(f.numpy().sum(), np.exp(-2.0 / 5.0), rtol=1e-4)


def test_diffusion_conserves_mass_with_no_flux_walls():
    f = FieldMemory((0, 0, 20, 20), 1.0, tau=1e12, diffusion=3.0)
    mask = np.zeros((20, 20), np.uint8)
    mask[:, 12] = 1                  # a wall
    f.set_blocked(mask)
    f.deposit_points(np.array([[5.5, 5.5]] * 10), 1.0)
    for _ in range(200):
        f.update(1 / 60)
    v = f.numpy()[0]
    assert np.isclose(v.sum(), 10.0, rtol=1e-4)
    assert v[:, 12:].sum() == 0.0    # nothing leaks through the wall


def test_fixed_point_deposits_are_order_independent():
    pts = np.random.default_rng(0).uniform(0, 4, (5000, 2))
    a = FieldMemory((0, 0, 4, 4), 1.0)
    b = FieldMemory((0, 0, 4, 4), 1.0)
    a.deposit_points(pts, 0.123)
    b.deposit_points(pts[::-1].copy(), 0.123)
    assert (a.deposit.numpy() == b.deposit.numpy()).all()


def test_trace_memory_keeps_the_newest_in_order():
    m = TraceMemory(3, 1)
    for i in range(5):
        m.push(i, [i])
    t, v = m.ordered()
    assert list(t) == [2, 3, 4] and list(v[:, 0]) == [2, 3, 4]
