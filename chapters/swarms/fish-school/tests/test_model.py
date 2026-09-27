"""The equations in advanced.md are what the kernel computes."""

import sys

import numpy as np
import sympy as sp

from ailab.core.catalog import discover, load_sim_class
from ailab.core.sim import InputState, SimContext

INFO = discover().chapters["swarms.fish-school"]


def two_fish(dist: float, **params):
    """Fish 0 at the centre, fish 1 `dist` ahead of it; both swimming along +x."""
    Sim = load_sim_class(INFO)
    sim = Sim(SimContext("cpu", 1))
    sim.set_param("n_fish", 50)          # restarts; then replace with 2 fish
    for k, v in params.items():
        sim.set_param(k, v)
    sim.load_state([[80, 45], [80 + dist, 45]], [[8, 0], [8, 0]])
    sim.advance(InputState())
    return sim, sim.forces.numpy()


def test_separation_force_matches_the_formula():
    d = 0.9
    sim, F = two_fish(d)
    r, w = sim.p.r_rep, sim.p.w_sep
    s = -(1 - d / r)                       # s_i = -u (1 - d/r_r), u = +x
    expected = w * s / max(abs(s), 1.0)
    assert np.isclose(F[0, 0][0], expected, atol=1e-5)
    assert np.isclose(F[0, 0][1], 0.0, atol=1e-6)


def test_blind_spot_hides_the_fish_behind():
    # Without the lateral line, fish 1 cannot sense fish 0 directly behind it (300 deg
    # vision leaves a 60 deg blind spot), while fish 0 sees fish 1 ahead.
    _, F = two_fish(0.9, r_lat=0.0, fov=300.0)
    assert np.allclose(F[1, 0], 0.0)
    assert not np.allclose(F[0, 0], 0.0)


def test_lateral_line_feels_behind():
    _, F = two_fish(0.9, r_lat=2.0, fov=300.0)
    assert not np.allclose(F[1, 0], 0.0)


def test_alarm_hop_count():
    c, eps, h = sp.symbols("c epsilon h", positive=True)
    hops = sp.solve(sp.Eq(c**h, eps), h)[0]
    assert sp.simplify(hops - sp.log(eps) / sp.log(c)) == 0
    assert float(hops.subs({c: 0.85, eps: 0.3})) == np.log(0.3) / np.log(0.85)


def test_module_exposes_kernel():
    mod = sys.modules[load_sim_class(INFO).__module__]
    assert hasattr(mod, "school_step")


def one_fish_near_rock(rock_x: float, **params):
    """One fish at (80, 45) swimming +x toward a rock wall at x = rock_x."""
    Sim = load_sim_class(INFO)
    sim = Sim(SimContext("cpu", 1))
    sim.set_param("n_fish", 50)
    for k, v in params.items():
        sim.set_param(k, v)
    mask = np.zeros((sim.rocks.ny, sim.rocks.nx), np.uint8)
    mask[:, int(rock_x):] = 1
    sim.rocks.set(mask)
    sim.load_state([[80, 45]], [[8, 0]])
    sim.advance(InputState())
    return sim, sim.forces.numpy()[0]


def test_rock_pressure_matches_the_formula():
    # Rock fills x >= 82 (cells 82..). Only cells within r_rock = 3 of (80, 45) push.
    sim, F = one_fish_near_rock(82, w_rock=1.0)
    p = np.array([80.0, 45.0])
    push = np.zeros(2)
    for iy in range(sim.rocks.ny):
        for ix in range(82, sim.rocks.nx):
            c = np.array([ix + 0.5, iy + 0.5])
            d = np.linalg.norm(p - c)
            if d < 3.0:
                push += (p - c) / d * (1 - d / 3.0) ** 2
    # the look ahead hits rock at l = 2.0 (first half-cell step inside x >= 82)
    ell, L = 2.0, 7.0
    turn = np.array([0.0, 1.0])     # h = +x so h_perp = +y; the side looks tie -> s = +1
    assert np.allclose(F[5], push + (1 - ell / L) * turn, atol=1e-4)


def test_hunger_grows_and_weakens_flight():
    tau, dt = sp.symbols("tau_h Delta_t", positive=True)
    h = sp.Symbol("h", nonnegative=True)
    step = sp.Min(h + dt / tau, 1)
    assert step.subs({h: sp.Rational(1, 2), dt: 1, tau: 20}) == sp.Rational(11, 20)
    # a fish fed at t=0 is starving after tau_h seconds of no food
    Sim = load_sim_class(INFO)
    sim = Sim(SimContext("cpu", 1))
    sim.set_param("n_fish", 50)
    sim.set_param("hunger_tau", 2.0)
    sim.load_state([[40, 45]], [[8, 0]], hunger=[0.0])
    for _ in range(int(1.0 / sim.dt)):
        sim.advance(InputState())
    assert np.isclose(sim.hunger.numpy()[0], 0.5, atol=1e-3)


def test_starving_fish_flees_at_one_minus_rho():
    def flee(h0):
        Sim = load_sim_class(INFO)
        sim = Sim(SimContext("cpu", 1))
        sim.set_param("n_fish", 50)
        sim.set_param("risk", 0.6)
        sim.set_param("hunger_tau", 120.0)
        sim.load_state([[80, 45]], [[8, 0]], hunger=[h0])
        sim.advance(InputState(mouse=(90.0, 45.0), tool="hunt"))   # predator ahead
        return np.linalg.norm(sim.forces.numpy()[0, 3]), float(sim.hunger.numpy()[0])

    f_fed, h_fed = flee(0.0)
    f_hungry, h_hungry = flee(1.0)
    assert f_fed > 0
    assert np.isclose(f_hungry / f_fed, (1 - 0.6 * h_hungry) / (1 - 0.6 * h_fed), rtol=1e-4)


def test_food_is_eaten_deterministically():
    Sim = load_sim_class(INFO)
    sim = Sim(SimContext("cpu", 1))
    sim.set_param("n_fish", 50)
    sim.load_state([[80, 45], [80.3, 45.2]], [[8, 0], [8, 0]], hunger=[1.0, 1.0])
    sim.advance(InputState(mouse=(80.6, 45.0), tool="food",
                           clicks=(("left", (80.6, 45.0)),)))
    stats = sim.stats.numpy()
    assert stats[0] == 2                       # both fish bit in the same step
    assert sim.pellet_amt_d.numpy().sum() == 7 * 6 - 2
