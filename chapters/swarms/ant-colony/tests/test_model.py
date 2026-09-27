import numpy as np
import sympy as sp
import warp as wp

from ailab.core.catalog import discover, load_sim_class
from ailab.core.sim import InputState, SimContext

INFO = discover().chapters["swarms.ant-colony"]


def test_deneubourg_choice_is_a_probability_and_breaks_symmetry():
    A, B, k, n = sp.symbols("A B k n", positive=True)
    PA = (k + A) ** n / ((k + A) ** n + (k + B) ** n)
    PB = PA.subs({A: B, B: A}, simultaneous=True)
    assert sp.simplify(PA + PB - 1) == 0
    assert sp.simplify(PA.subs(B, A) - sp.Rational(1, 2)) == 0
    # a small lead is amplified more when n = 2 than when n = 1
    lead = {A: 12, B: 10, k: 20}
    assert PA.subs({**lead, n: 2}) > PA.subs({**lead, n: 1}) > sp.Rational(1, 2)


def test_log_response_keeps_weak_cues_visible():
    # a busy trail (1000) vs. a faint one (10): linear ratio 100, log ratio ~2.9
    assert np.log1p(1000) / np.log1p(10) < 3.0


def test_food_is_conserved():
    sim = load_sim_class(INFO)(SimContext("cpu", 3))
    total = int(sim.food.numpy().sum())
    for _ in range(600):
        sim.advance(InputState())
    left = int(sim.food.numpy().sum())
    carrying = int(sim.carry.numpy().sum())
    delivered = int(sim.delivered.numpy()[0])
    assert left + carrying + delivered == total


def test_double_bridge_is_solvable():
    sim = load_sim_class(INFO)(SimContext("cpu", 1))
    sim.set_param("scenario", "Double bridge")
    for _ in range(60 * 100):       # ~60 s of exploring before the first delivery (600 ants)
        sim.advance(InputState())
    assert int(sim.delivered.numpy()[0]) > 0


def test_wall_tool_builds_rock_that_ants_never_enter():
    sim = load_sim_class(INFO)(SimContext("cpu", 2))
    for k in range(40):             # drag a wall straight across the field, right of the nest
        y = 5 + k * 2.0
        sim.advance(InputState(mouse=(60.0, y), buttons=frozenset({"left"}), tool="wall"))
    assert sim.rocks.painted > 0
    for _ in range(300):
        sim.advance(InputState())
    pos = sim.pos.numpy()
    ix, iy = (pos[:, 0] / 0.5).astype(int), (pos[:, 1] / 0.5).astype(int)
    assert not sim.rocks.mask[iy, ix].any()


def test_food_tool_adds_food_and_counts_it_as_the_learners():
    sim = load_sim_class(INFO)(SimContext("cpu", 2))
    before = int(sim.food.numpy().sum())
    sim.advance(InputState(mouse=(100.0, 45.0), tool="food",
                           clicks=(("left", (100.0, 45.0)),)))
    added = int(sim.food.numpy().sum()) - before
    assert added > 0 and added == sim.user_food_added


def test_mark_strength_formula():
    q, t, tau = sp.symbols("q t_a tau_a", positive=True)
    rate = q * sp.exp(-t / tau)
    # after one "memory time" the mark is 1/e as strong
    assert sp.simplify(rate.subs(t, tau) / q - sp.exp(-1)) == 0


def test_steady_turning_adds_up_to_omega_tau():
    # W <- W e^(-dt/tau) + omega dt, forever: the geometric series tends to ~ omega tau
    w, dt, tau = sp.symbols("omega dt tau_w", positive=True)
    limit = w * dt / (1 - sp.exp(-dt / tau))
    assert sp.limit(limit, dt, 0) == w * tau
    # so an ant stuck turning one way (omega = 8, tau_w = 3 -> 24 rad) is soon "lost" at
    # 1.5 circles (9.4 rad), while a straight walk (no steering) never is
    assert 8 * 3 > 2 * np.pi * 1.5


def _lost_colony(navigators: bool):
    sim = load_sim_class(INFO)(SimContext("cpu", 4))
    sim.set_param("navigators", navigators)
    n = len(sim.pos)
    spot = next(q for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                if sim.rocks.free(q := np.asarray(sim.nest) + 30 * np.array([np.cos(a),
                                                                            np.sin(a)]))
                and 5 < q[0] < 155 and 5 < q[1] < 85)
    far = np.tile(spot, (n, 1)).astype(np.float32)
    sim.pos = wp.array(far, dtype=wp.vec2, device=sim.device)
    sim.wind = wp.array(np.full(n, 100.0, np.float32), dtype=float, device=sim.device)
    return sim


def test_lost_ants_navigate_home_and_lay_no_trail():
    sim = _lost_colony(True)
    d0 = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean()
    sim.advance(InputState())
    assert (sim.nav.numpy() > 0).all()               # every circling ant turned navigator
    assert sim.field.numpy().sum() == 0              # and none of them marked the ground
    for _ in range(120):                             # two seconds of compass walking
        sim.advance(InputState())
    d1 = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean()
    assert d1 < d0 - 10


def test_navigators_can_be_switched_off():
    sim = _lost_colony(False)
    sim.advance(InputState())
    assert (sim.nav.numpy() == 0).all()
    assert sim.field.numpy().sum() > 0
