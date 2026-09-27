"""The equations in advanced.md are what the kernel computes."""

import sys

import numpy as np
import sympy as sp
import warp as wp

from ailab.core.catalog import discover, load_sim_class
from ailab.core.sim import InputState, SimContext

INFO = discover().chapters["swarms.fish-school"]


def two_fish(dist: float, **params):
    """Fish 0 at the origin, fish 1 `dist` ahead of it; both swimming along +x."""
    Sim = load_sim_class(INFO)
    sim = Sim(SimContext("cpu", 1))
    sim.set_param("n_fish", 50)          # restarts; then overwrite with 2 fish
    for k, v in params.items():
        sim.set_param(k, v)
    pos = np.array([[80, 45, 0], [80 + dist, 45, 0]], np.float32)
    vel = np.array([[8, 0], [8, 0]], np.float32)
    n = 2
    sim.pos = wp.array(pos, dtype=wp.vec3, device="cpu")
    sim.vel = wp.array(vel, dtype=wp.vec2, device="cpu")
    for name in ("fear", "mem_str"):
        setattr(sim, name, wp.zeros(n, dtype=float, device="cpu"))
    sim.mem_pos = wp.zeros(n, dtype=wp.vec2, device="cpu")
    sim._back = [wp.empty_like(a) for a in (sim.pos, sim.vel, sim.fear, sim.mem_pos,
                                             sim.mem_str)]
    sim.forces = wp.zeros((n, 5), dtype=wp.vec2, device="cpu")
    sim.hue = np.zeros(n, np.float32)
    sim.swim = np.zeros(n, np.float32)
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
