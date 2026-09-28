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
    # so a loaded ant stuck turning one way (omega = 8, tau_w = 8 -> 64 rad, about ten
    # circles) is soon "lost" at 4 circles (25 rad), while a straight walk never is
    assert 8 * 8 > 2 * np.pi * 4


def _lost_colony(navigators: bool, how: str):
    """Every ant 30 units from the nest, lost in one of two ways: 'circling' (carrying
    food and turning round and round) or 'searching' (out far longer than T_lost)."""
    sim = load_sim_class(INFO)(SimContext("cpu", 4))
    sim.set_param("navigators", navigators)
    n = len(sim.pos)
    spot = next(q for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                if sim.rocks.free(q := np.asarray(sim.nest) + 30 * np.array([np.cos(a),
                                                                            np.sin(a)]))
                and 5 < q[0] < 155 and 5 < q[1] < 85)
    far = np.tile(spot, (n, 1)).astype(np.float32)
    dev = sim.device
    sim.pos = wp.array(far, dtype=wp.vec2, device=dev)
    if how == "circling":
        sim.carry = wp.array(np.ones(n, np.int32), dtype=int, device=dev)
        sim.wind = wp.array(np.full(n, 100.0, np.float32), dtype=float, device=dev)
    else:
        sim.lost_clock = wp.array(np.full(n, 1e4, np.float32), dtype=float, device=dev)
    return sim


def test_lost_ants_navigate_home_and_lay_no_trail():
    for how in ("circling", "searching"):
        sim = _lost_colony(True, how)
        d0 = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean()
        sim.advance(InputState())
        assert (sim.nav.numpy() > 0).all(), how      # every lost ant turned navigator
        assert sim.field.numpy().sum() == 0, how     # and none of them marked the ground
        for _ in range(120):                         # two seconds of compass walking
            sim.advance(InputState())
        d1 = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean()
        assert d1 < d0 - 10, how


def test_circling_only_counts_for_ants_carrying_food():
    # a searching ant turning a lot is just searching; an ant mill is made of loaded ants
    sim = _lost_colony(True, "circling")
    sim.carry = wp.zeros(len(sim.pos), dtype=int, device=sim.device)
    sim.advance(InputState())
    assert (sim.nav.numpy() == 0).all()


def test_navigators_can_be_switched_off():
    sim = _lost_colony(False, "circling")
    sim.advance(InputState())
    assert (sim.nav.numpy() == 0).all()


def _out_for(sim, seconds: float) -> None:
    sim.away = wp.array(np.full(len(sim.pos), seconds, np.float32), dtype=float,
                        device=sim.device)


def test_hunger_rule_t_out_over_t_trip():
    # t_out > T_trip => home by compass (no marks); a moment less => still foraging
    for over, hungry in ((1.0, True), (-1.0, False)):
        sim = _lost_colony(False, "searching")
        sim.lost_clock = wp.zeros(len(sim.pos), dtype=float, device=sim.device)
        _out_for(sim, sim.p.trip + over)
        sim.advance(InputState())
        assert (sim.field.numpy().sum() == 0) == hungry
    sim = _lost_colony(False, "searching")
    sim.set_param("trips", False)                   # hunger off: never hungry
    _out_for(sim, 1e4)
    sim.advance(InputState())
    assert sim.field.numpy().sum() > 0


def test_hungry_ants_walk_home_rest_and_go_out_again():
    sim = _lost_colony(False, "searching")
    sim.lost_clock = wp.zeros(len(sim.pos), dtype=float, device=sim.device)
    sim.set_param("rest", 2.0)
    _out_for(sim, sim.p.trip + 1)
    d0 = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean()
    for _ in range(120):                             # two seconds of compass walking
        sim.advance(InputState())
    assert np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1).mean() < d0 - 10
    rested = np.zeros(len(sim.pos), bool)
    for _ in range(60 * 30):
        before = sim.pos.numpy().copy()
        resting = sim.away.numpy() < 0
        sim.advance(InputState())
        assert (sim.pos.numpy()[resting] == before[resting]).all()   # resting: still
        rested |= sim.away.numpy() < 0
        if rested.all():
            break
    assert rested.all(), f"{int((~rested).sum())} hungry ants never got home to rest"
    for _ in range(int(2.0 * 60) + 2):               # T_rest later everyone is out again
        sim.advance(InputState())
    assert (sim.away.numpy() >= 0).all()
    assert sim.field.numpy().sum() > 0


# ------------------------------------------------------------------ personalities
def test_variety_zero_means_identical_ants():
    sim = load_sim_class(INFO)(SimContext("cpu", 5))
    sim.set_param("variety", 0.0)
    t = sim.trait_np
    assert (t[:, :3] == 1.0).all()
    assert set(np.unique(t[:, 3])) == {-1.0, 1.0}      # wall sides still differ


def test_personalities_are_seeded_and_centred_on_the_colony_values():
    a = load_sim_class(INFO)(SimContext("cpu", 5)).trait_np
    b = load_sim_class(INFO)(SimContext("cpu", 5)).trait_np
    assert (a == b).all()
    # e^(kappa z) with z ~ N(0, 1): the median factor is 1 (half faster, half slower)
    assert abs(np.median(np.log(a[:, :3]))) < 0.05
    assert a[:, :3].min() > 0


# ------------------------------------------------------------------ walls and mazes
def _free_components(mask: np.ndarray) -> int:
    """Connected pieces of rock (4-neighbour flood fill)."""
    seen = np.zeros_like(mask, bool)
    pieces = 0
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        pieces += 1
        stack = [(y0, x0)]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u] \
                        and not seen[v, u]:
                    seen[v, u] = True
                    stack.append((v, u))
    return pieces


def test_the_maze_has_one_unbroken_wall():
    # A perfect maze (randomised depth-first carving: no loops) is one tree of corridors,
    # so all of its rock is a single connected piece. That is what makes the hand-on-wall
    # rule work: following that one wall passes every corridor.
    for seed in (1, 2, 3):
        sim = load_sim_class(INFO)(SimContext("cpu", seed))
        sim.set_param("scenario", "Maze")
        assert _free_components(sim.rocks.mask == 1) == 1


PURE_WALL_FOLLOWER = {"scenario": "Maze", "n_ants": 50, "variety": 0.0, "wander": 0.0,
                      "deposit": 0.0, "food_cue": 0.0, "nest_cue": 0.0, "homing": 0.0,
                      "navigators": False, "trips": False, "hug_release": 0.0,
                      "hug_time": 20.0}


def test_hand_on_wall_takes_every_ant_to_the_food_and_back():
    # the limiting case of the model: no smells, no wander, never letting go of the wall
    sim = load_sim_class(INFO)(SimContext("cpu", 1))
    sim.apply_values(PURE_WALL_FOLLOWER)
    n = len(sim.pos)
    found = np.zeros(n, bool)
    back = np.zeros(n, bool)
    for k in range(200 * 60):
        sim.advance(InputState())
        if k % 30 == 0:
            found |= sim.carry.numpy() == 1
            near = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1) < 5.0
            back |= found & near
            if back.all():
                break
    assert found.all(), f"{int((~found).sum())} ants never reached the food"
    assert back.all(), f"{int((~back).sum())} ants never came back to the nest room"


def _in_rock(sim) -> int:
    p = sim.pos.numpy()
    ix, iy = (p[:, 0] / 0.5).astype(int), (p[:, 1] / 0.5).astype(int)
    return int(sim.rocks.mask[iy, ix].sum())


def _maze_run(seed: int, seconds: float):
    sim = load_sim_class(INFO)(SimContext("cpu", seed))
    maze = next(pr for pr in sim.PRESETS if pr.key == "maze")
    sim.apply_values(maze.values)
    n = len(sim.pos)
    away = np.zeros(n, bool)
    found = np.zeros(n, bool)
    back = np.zeros(n, bool)
    first = None
    for k in range(int(seconds * 60)):
        sim.advance(InputState())
        if k % 30 == 0:
            assert _in_rock(sim) == 0, "an ant walked into rock"
            dn = np.linalg.norm(sim.pos.numpy() - sim.nest, axis=1)
            away |= dn > 12
            back |= away & (dn < 3.5)
            found |= sim.carry.numpy() == 1
            if first is None and sim.delivered.numpy()[0] > 0:
                first = k / 60
    return first, found | back


def test_the_maze_is_solvable_and_every_ant_finds_its_way():
    first, ok = _maze_run(1, 600)
    assert first is not None and first < 300, "no food delivered within 300 s"
    assert ok.mean() >= 0.99, f"only {ok.mean():.1%} of ants ever reached food or home"


def test_the_maze_is_solvable_for_other_seeds():
    for seed in (2, 3):
        first, _ = _maze_run(seed, 300)
        assert first is not None, f"seed {seed}: no food delivered within 300 s"


# ------------------------------------------------------------------ no entry, quality
def _set_field(sim, channel: int, values: np.ndarray) -> None:
    f = sim.field.value.numpy()
    f[channel] = values
    sim.field.value = wp.array(f, dtype=float, device=sim.device)


def test_frustrated_searchers_mark_no_entry_and_only_then():
    for on in (True, False):
        sim = load_sim_class(INFO)(SimContext("cpu", 6))
        sim.set_param("noentry", on)
        n = len(sim.pos)
        _set_field(sim, 1, np.full(sim.field.value.shape[1:], 50.0, np.float32))  # on a trail
        sim.frust = wp.array(np.full(n, 1e4, np.float32), dtype=float, device=sim.device)
        sim.advance(InputState())
        marks = float(sim.field.numpy()[2].sum())
        assert (marks > 0) == on


def test_no_entry_lowers_what_searchers_smell():
    # S = ln(1 + sum c) - k_ne ln(1 + sum n): the host mirror uses the kernel's rule
    sim = load_sim_class(INFO)(SimContext("cpu", 6))
    field = np.zeros(sim.field.value.shape, np.float32)
    field[1] = 9.0 / 9                        # sum over the 3x3 cells = 9 -> ln(10)
    field[2] = 9.0 / 9
    food = np.zeros(field.shape[1:], np.int32)
    p = np.array([80.0, 45.0])
    _, clean = sim._sense(p, 0.0, 0, field * np.array([1, 1, 0])[:, None, None], food)
    _, marked = sim._sense(p, 0.0, 0, field, food)
    k = sim.p.ne_weight
    assert np.allclose(clean - marked, k * np.log(10.0), atol=1e-4)
    _, loaded = sim._sense(p, 0.0, 1, field, food)   # loaded ants ignore no-entry
    _, loaded_clean = sim._sense(p, 0.0, 1, field * np.array([1, 1, 0])[:, None, None], food)
    assert np.allclose(loaded, loaded_clean)


def test_rich_food_is_marked_more_strongly():
    # delta c = q Q e^(-t_a / tau_a) dt: same ants, same step, only Q differs
    totals = []
    for q in (1.0, 2.5):
        sim = load_sim_class(INFO)(SimContext("cpu", 6))
        n = len(sim.pos)
        sim.carry = wp.array(np.ones(n, np.int32), dtype=int, device=sim.device)
        sim.carried_q = wp.array(np.full(n, q, np.float32), dtype=float, device=sim.device)
        sim.advance(InputState())
        totals.append(float(sim.field.numpy()[1].sum()))
    assert np.isclose(totals[1] / totals[0], 2.5, rtol=0.02)


def test_the_colony_chooses_the_richer_food():
    sim = load_sim_class(INFO)(SimContext("cpu", 1))
    sim.set_param("scenario", "Two foods")
    rich = sim.quality_np > 1.0
    before = sim.food.numpy().copy()
    for _ in range(90 * 60):
        sim.advance(InputState())
    eaten = before - sim.food.numpy()
    share = eaten[rich].sum() / max(eaten.sum(), 1)
    assert eaten.sum() > 0 and share > 0.7, f"rich share {share:.2f}"


# ------------------------------------------------------------------ colony store
def test_store_fills_with_loads_and_empties_at_the_colony_appetite():
    # S <- max(S + loads - N e / 60 * dt, 0); H = max(0, 1 - S / S_full)
    sim = load_sim_class(INFO)(SimContext("cpu", 1))
    sim.apply_values({"store0": 100.0, "appetite": 0.6, "full": 400.0})
    n, dev = len(sim.pos), sim.device
    sim.delivered = wp.array([30], dtype=int, device=dev)       # 30 loads just came home
    sim.pos = wp.array(np.tile(np.float32([5.0, 5.0]), (n, 1)), dtype=wp.vec2, device=dev)
    sim.advance(InputState())                                    # (ants parked in a corner)
    store, hunger = sim.colony.numpy()
    expected = 100 + 30 - n * 0.6 / 60 * sim.dt
    assert np.isclose(store, expected, rtol=1e-5)
    assert np.isclose(hunger, 1 - expected / 400, rtol=1e-5)
    sim.colony = wp.array([0.0, 0.0], dtype=float, device=dev)
    sim.advance(InputState())
    assert tuple(sim.colony.numpy()) == (0.0, 1.0)              # never below empty


def _hungry_test_colony(hunger: float, **values):
    sim = load_sim_class(INFO)(SimContext("cpu", 6))
    sim.apply_values({"variety": 0.0, "deposit": 0.0, "food_cue": 0.0, "nest_cue": 0.0,
                      "trips": False, "hug_time": 0.0, "appetite": 0.0, **values})
    sim.set_param("full", 100.0)
    sim.colony = wp.array([100.0 * (1 - hunger), hunger], dtype=float, device=sim.device)
    return sim


def test_a_hungry_colony_s_searchers_wander_more():
    # sigma_search = sigma_i (1 + k_r H): the spread of random turns grows by (1 + k_r)
    spread = []
    for hunger in (0.0, 1.0):
        sim = _hungry_test_colony(hunger, wander=1.0, restless=1.5)
        a0 = sim.ang.numpy().copy()
        sim.advance(InputState())
        spread.append(np.std(sim.ang.numpy() - a0))
    assert np.isclose(spread[1] / spread[0], 2.5, rtol=0.1)


def test_a_hungry_colony_s_loaded_ants_rush_home():
    # h_loaded = h + k_d H: the turn toward home grows with the colony's hunger
    turns = []
    for hunger in (0.0, 1.0):
        sim = _hungry_test_colony(hunger, wander=0.0, homing=0.5, desperate=1.0)
        n, dev = len(sim.pos), sim.device
        at = np.asarray(sim.nest) + np.array([0.0, 30.0])
        sim.pos = wp.array(np.tile(at.astype(np.float32), (n, 1)), dtype=wp.vec2, device=dev)
        sim.ang = wp.zeros(n, dtype=float, device=dev)            # nest 90 degrees to the right
        sim.carry = wp.ones(n, dtype=int, device=dev)
        sim.advance(InputState())
        turns.append(float(-sim.ang.numpy().mean()))
    p = sim.p
    for hunger, turn in zip((0.0, 1.0), turns, strict=True):
        assert np.isclose(turn, p.turn_rate * (0.5 + 1.0 * hunger) * sim.dt, rtol=0.02)


def test_new_food_grows_on_free_ground_away_from_the_nest():
    sim = load_sim_class(INFO)(SimContext("cpu", 2))
    sim.apply_values({"regrow": 5.0})
    before = sim.food.numpy() > 0
    for _ in range(int(5.0 * 60) + 1):
        sim.advance(InputState())
    new = (sim.food.numpy() > 0) & ~before
    assert new.sum() > 20
    assert not (new & (sim.rocks.mask == 1)).any()
    iy, ix = np.nonzero(new)
    d = np.hypot((ix + 0.5) * 0.5 - sim.nest[0], (iy + 0.5) * 0.5 - sim.nest[1])
    assert d.min() > 25
