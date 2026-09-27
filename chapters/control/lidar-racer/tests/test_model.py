import sys

import numpy as np
import sympy as sp

from ailab.core.catalog import discover, load_sim_class
from ailab.core.sim import InputState, SimContext

INFO = discover().chapters["control.lidar-racer"]
M = sys.modules[load_sim_class(INFO).__module__]


def test_ray_segment_hit_matches_geometry():
    # wall x = 10 from y=-50..50; ray at angle a from the origin hits at 10 / cos(a)
    a = np.radians(np.array([-60, -20, 0, 35]))
    dirs = np.stack([np.cos(a), np.sin(a)], 1)
    r = M.raycast(np.zeros(2), dirs, np.array([[10.0, -50]]), np.array([[10.0, 50]]),
                  np.zeros((0, 2)), 0.5, 100.0)
    assert np.allclose(r, 10 / np.cos(a))


def test_ray_circle_hit():
    r = M.raycast(np.zeros(2), np.array([[1.0, 0.0]]), np.zeros((0, 2)), np.zeros((0, 2)),
                  np.array([[5.0, 0.0]]), 0.5, 100.0)
    assert np.isclose(r[0], 4.5)


def test_pure_pursuit_curvature_from_chord_geometry():
    R, a, Ld = sp.symbols("R alpha L_d", positive=True)
    # a chord of length L_d subtending 2*alpha at the centre: L_d = 2 R sin(alpha)
    kappa = 1 / sp.solve(sp.Eq(Ld, 2 * R * sp.sin(a)), R)[0]
    assert sp.simplify(kappa - 2 * sp.sin(a) / Ld) == 0
    delta, ld = M.pure_pursuit(0.4, 50.0, 10.0, 0.3, 3.0)
    assert np.isclose(ld, 0.3 * 10 + 3)
    assert np.isclose(delta, np.arctan(2 * M.WHEELBASE * np.sin(0.4) / ld))


def test_stopping_distance():
    t, v0, a = sp.symbols("t v0 a", positive=True)
    t_stop = v0 / a
    d = sp.integrate(v0 - a * t, (t, 0, t_stop))
    assert sp.simplify(d - v0**2 / (2 * a)) == 0


def test_follow_the_gap_picks_the_open_side():
    angles = np.radians(np.linspace(-90, 90, 19))
    ranges = np.full(19, 3.0)
    ranges[12:17] = 25.0                     # open space to the left (positive angles)
    plan = M.follow_the_gap(ranges, angles, 30.0, 0.5, 6.0, "Gap centre")
    assert plan["alpha"] > 0


def test_tracks_are_drivable_geometry():
    for name in M.TRACKS:
        c = M.track_centerline(name)
        assert M.min_turn_radius(c) > M.HALF_TRACK + 2, name
        assert (c.min(0) - M.HALF_TRACK > 0).all() and (c.max(0) + M.HALF_TRACK < (160, 90)).all()


def test_default_controller_laps_without_crashing():
    Sim = load_sim_class(INFO)
    for track in M.TRACKS:
        sim = Sim(SimContext("cpu", 1))
        sim.set_param("track", track)
        for _ in range(60 * 45):
            sim.advance(InputState())
        assert sim.crashes == 0 and sim.lap >= 1, track
