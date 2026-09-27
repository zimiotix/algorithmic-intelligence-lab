import numpy as np
import sympy as sp

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
