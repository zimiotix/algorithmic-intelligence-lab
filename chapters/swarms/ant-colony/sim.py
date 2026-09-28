"""Ant colony foraging: stigmergy, i.e. memory written into the environment.

Each ant is almost blind and has no map. It follows three sensors (left, front, right)
that smell pheromone, and it lays pheromone itself:
  searching ants lay the "home" trail, ants carrying food lay the "food" trail.
The mark gets weaker the longer the ant has been away from where it came from (a
per-ant working memory). Short paths get reinforced faster than they evaporate, so
the colony converges on them. No ant ever knows the path; the colony does.

Determinism: randomness is counter-based (seed, step, ant id); pheromone deposits
are integer atomics; when two ants reach the same crumb, the lower id wins (atomic_min).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ailab.core import (
    Experiment,
    HudItem,
    InputState,
    LiveEq,
    LiveValue,
    Overlay,
    Param,
    Preset,
    Simulation,
    Tool,
    section,
)
from ailab.core.memory import (
    FieldMemory,
    Grid2D,
    field_deposit,
    field_sample3,
    grid_cell,
    grid_inside,
)
from ailab.core.params import fmt
from ailab.core.rng import step_seed
from ailab.core.sandbox import ObstacleGrid, obstacle_at
from ailab.render import palette as pal

WIDTH, HEIGHT, CELL = 160.0, 90.0, 0.5
HOME, FOOD, NOENTRY = 0, 1, 2      # pheromone channels
NO_CLAIM = 2**31 - 1
SCENARIOS = ("Open field", "Double bridge", "Maze", "Two foods")
NAV_COLOR = "#8b5cf6"   # navigator ants: violet
RICH_COLOR = "#fbbf24"  # rich food: gold


@wp.struct
class Colony:
    speed: float
    sensor_angle: float
    sensor_dist: float
    turn_rate: float
    wander: float
    deposit: float
    trail_tau: float
    nest: wp.vec2
    nest_r: float
    scent_r: float
    food_cue: float
    nest_cue: float
    nav_on: int          # navigator ants enabled
    nav_loops: float     # full circles of recent turning before an ant counts as lost
    wind_tau: float      # how fast old turning is forgotten
    nav_gain: float      # compass strength of a navigator
    nav_time: float      # how long it navigates before trusting smells again
    give_up: float       # out this long without reaching food or home: lost
    hug_time: float      # how long an ant looks for a wall it has lost from its side
    hug_gain: float      # how hard it turns back toward a lost wall (round corners)
    feel_angle: float    # side feeler direction, from the heading toward the wall side
    feel_dist: float     # side feeler reach
    slide_step: float    # turning increments when searching for a free way along a wall
    hug_release: float   # chance per second of letting go of a wall
    homing: float
    dt: float
    seed: int


@wp.struct
class Marks:
    """The "no entry" signal (kept apart from Colony: Warp struct arguments stay small)."""
    on: int              # "no entry" marks enabled
    after: float         # following a food trail this long without food: frustrated
    rate: float          # how much "no entry" a frustrated ant lays per second
    weight: float        # how strongly searchers avoid "no entry"
    trail: float         # the food-trail reading that counts as "on a trail"


@wp.struct
class Trips:
    """Why and how an ant goes home (kept apart from Colony: small struct arguments)."""
    on: int              # limited trips: an ant must come home to eat and rest
    trip: float          # out this long: hungry and tired, it heads home
    rest: float          # how long it rests in the nest before going out again


@wp.func
def smell(P: Colony, M: Marks, g: Grid2D, value: wp.array3d(dtype=float), food: wp.array2d(dtype=int),
          q: wp.vec2, carrying: int) -> float:
    # Weber-Fechner: sensors respond to the logarithm of concentration, so a busy trail
    # never drowns out the direct smell of food or of the nest.
    s = float(0.0)
    if carrying == 0:
        s = wp.log(1.0 + field_sample3(value, 1, g, q))     # follow the food trail
        if M.on == 1:                                       # ...but not into "no entry"
            s -= M.weight * wp.log(1.0 + field_sample3(value, 2, g, q))
        c = grid_cell(g, q)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cc = wp.vec2i(c[0] + dx, c[1] + dy)
                if grid_inside(g, cc):
                    if food[cc[1], cc[0]] > 0:
                        s += P.food_cue                        # food itself
    else:
        s = wp.log(1.0 + field_sample3(value, 0, g, q))     # follow the home trail
        dn = wp.length(q - P.nest)
        if dn < P.scent_r:
            s += P.nest_cue * (1.0 - dn / P.scent_r)           # the nest smells too
    return s


@wp.func
def heading(a: float) -> wp.vec2:
    return wp.vec2(wp.cos(a), wp.sin(a))


@wp.kernel
def ant_step(P: Colony, M: Marks, T: Trips, g: Grid2D,
             pos: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
             carry: wp.array(dtype=int), timer: wp.array(dtype=float),
             value: wp.array3d(dtype=float), deposit: wp.array3d(dtype=wp.int32),
             food: wp.array2d(dtype=int), blocked: wp.array2d(dtype=wp.uint8),
             claims: wp.array2d(dtype=int), delivered: wp.array(dtype=int),
             wind: wp.array(dtype=float), nav: wp.array(dtype=float),
             rescued: wp.array(dtype=int), hug: wp.array(dtype=float),
             lost_clock: wp.array(dtype=float), traits: wp.array(dtype=wp.vec4),
             hit_d: wp.array(dtype=float), frust: wp.array(dtype=float),
             carried_q: wp.array(dtype=float), away: wp.array(dtype=float)):
    i = wp.tid()
    p = pos[i]
    a = ang[i]
    rng = wp.rand_init(P.seed, i)
    aw = away[i]              # seconds since it left the nest; < 0: resting in the nest
    if aw < 0.0:
        aw = aw + P.dt
        if aw >= 0.0:
            aw = 0.0
            ang[i] = wp.randf(rng) * 6.28318531   # rested: out again, any direction
        away[i] = aw
        return
    c = carry[i]
    tm = timer[i]
    w = wind[i]
    nv = nav[i]
    hg = hug[i]
    lc = lost_clock[i]
    hd = hit_d[i]             # navigators: distance from home where they met this wall
    fr = frust[i]             # seconds spent following a food trail without finding food
    tr = traits[i]            # this ant's own speed, wander and trail loyalty, and wall side
    side = tr[3]
    v = P.speed * tr[0]
    to_nest = P.nest - p
    home_err = wp.sin(wp.atan2(to_nest[1], to_nest[0]) - a)

    dist_home = wp.length(to_nest)
    dir_home = to_nest / wp.max(dist_home, 1.0e-6)

    # 1. sense: three sensors ahead-left, ahead, ahead-right
    sa = P.sensor_angle
    sd = P.sensor_dist
    s_l = smell(P, M, g, value, food, p + heading(a + sa) * sd, c)
    s_f = smell(P, M, g, value, food, p + heading(a) * sd, c)
    s_r = smell(P, M, g, value, food, p + heading(a - sa) * sd, c)
    sig = float(0.0)
    if s_f < s_l or s_f < s_r:
        if s_l > s_r:
            sig = 1.0
        elif s_r > s_l:
            sig = -1.0

    # 2. decide. Out too long: tired and hungry, it goes home. nav > 0: lost, navigating
    #    (seconds left); nav < 0: trusting smells for a while after a failed try
    tired = T.on == 1 and aw > T.trip
    turn = float(0.0)
    if nv > 0.0 or tired:
        # homebound by compass: smells ignored (a trail can lead into a dead end). The Bug
        # algorithm: head for home; blocked, follow the wall (step 3) until home is in the
        # clear and closer than where it met the wall
        if hg > 0.0:
            clear = not obstacle_at(blocked, g, p + dir_home * P.feel_dist)
            if clear and dist_home < hd:
                hg = 0.0                          # leave the wall, head home again
        if hg <= 0.0:
            turn = P.nav_gain * home_err
        if nv > 0.0:
            nv = nv - P.dt
            if nv <= 0.0:
                nv = -P.nav_time                  # didn't make it home: trust smells a while
                w = 0.0
        else:
            nv = wp.min(nv + P.dt, 0.0)
    else:
        hd = 1.0e9                                # not homebound: no wall to measure from
        nv = wp.min(nv + P.dt, 0.0)
        turn = tr[2] * sig                        # trail loyalty scales the pull of smells
        # going in circles? recent smell-steering adds up; old turning fades. (Only smells:
        # a mill is ants following each other's trail round and round.)
        w = w * wp.exp(-P.dt / P.wind_tau) + turn * P.turn_rate * P.dt
        # path integration: loaded ants also feel the direction home (breaks "ant mills")
        if c == 1:
            turn = turn + P.homing * home_err
        # lost: a loaded ant circling (an ant mill), or any ant out far too long
        circling = c == 1 and wp.abs(w) > 6.28318531 * P.nav_loops
        if P.nav_on == 1 and nv == 0.0 and (circling or lc > P.give_up):
            nv = P.nav_time                       # lost: become a navigator, face home
            w = 0.0
            hg = 0.0
            a = wp.atan2(to_nest[1], to_nest[0])

    # 3. wall following (thigmotaxis): keep a wall at your side; if it disappears (an
    #    outside corner), turn back toward it; now and then, let go
    homebound = nv > 0.0 or tired
    if hg > 0.0:
        if obstacle_at(blocked, g, p + heading(a + side * P.feel_angle) * P.feel_dist):
            hg = P.hug_time                       # the wall is still there
        else:
            turn = turn + side * P.hug_gain       # lost it: turn toward where it was
        hg = hg - P.dt
        if not homebound and wp.randf(rng) < 1.0 - wp.exp(-P.hug_release * P.dt):
            hg = -P.hug_time                      # let go, and don't grab a wall for a while
    elif hg < 0.0:
        hg = wp.min(hg + P.dt, 0.0)
    a = a + turn * P.turn_rate * P.dt + (wp.randf(rng) - 0.5) * 2.0 * P.wander * tr[1] * wp.sqrt(P.dt)

    # 4. act: move; at a wall, turn away from your wall side just enough to slide along it
    step = heading(a) * (v * P.dt)
    if obstacle_at(blocked, g, p + step):
        found = int(0)
        tries = int(wp.ceil(3.14159265 / P.slide_step))
        for k in range(1, tries + 1):
            if found == 0:
                b = a - side * P.slide_step * float(k)
                if not obstacle_at(blocked, g, p + heading(b) * (v * P.dt)):
                    a = b
                    found = 1
        if found == 0:
            a = a + 3.14159265                    # boxed in: turn right round
        else:
            p = p + heading(a) * (v * P.dt)
        if homebound and hg <= 0.0:
            hd = dist_home                        # homebound, meets a wall: remember where
        if (hg >= 0.0 or homebound) and P.hug_time > 0.0:
            hg = P.hug_time
    else:
        p = p + step

    # 5. mark: strength fades with time since the ant left home / found food
    ch = int(0)
    if c == 1:
        ch = 1
    if not homebound:                             # compass walkers don't feed the loop
        amount = P.deposit * wp.exp(-tm / P.trail_tau) * P.dt
        if c == 1:
            amount = amount * carried_q[i]        # better food, stronger trail
        field_deposit(deposit, ch, g, p, amount)
        # frustration: a searcher that follows a food trail for long without finding food
        # marks the place "no entry", so others stop following that trail here
        if c == 0 and wp.log(1.0 + field_sample3(value, 1, g, p)) > M.trail:
            fr = fr + P.dt
            if M.on == 1 and fr > M.after:
                field_deposit(deposit, 2, g, p, M.rate * P.dt)
    tm = tm + P.dt
    aw = aw + P.dt
    if hg <= 0.0:                                 # following a wall is progress, not lost
        lc = lc + P.dt

    if wp.length(p - P.nest) < P.nest_r:
        if nv > 0.0:
            wp.atomic_add(rescued, 0, 1)
        nv = 0.0
        w = 0.0
        lc = 0.0
        fr = 0.0
        if c == 1:
            c = 0
            wp.atomic_add(delivered, 0, 1)
            a = a + 3.14159265
        tm = 0.0
        if tired:
            aw = -T.rest                          # came home to eat: rest a while
        else:
            aw = 0.0                              # passing through: a bite, and on
    if c == 0:
        cell = grid_cell(g, p)
        if grid_inside(g, cell):
            if food[cell[1], cell[0]] > 0:
                wp.atomic_min(claims, cell[1], cell[0], i)   # lowest id wins, any order
    pos[i] = p
    ang[i] = a
    carry[i] = c
    timer[i] = tm
    wind[i] = w
    nav[i] = nv
    hug[i] = hg
    lost_clock[i] = lc
    hit_d[i] = hd
    frust[i] = fr
    away[i] = aw


@wp.kernel
def ant_pickup(g: Grid2D, pos: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
               carry: wp.array(dtype=int), timer: wp.array(dtype=float),
               lost_clock: wp.array(dtype=float), nav: wp.array(dtype=float),
               frust: wp.array(dtype=float), carried_q: wp.array(dtype=float),
               food: wp.array2d(dtype=int), quality: wp.array2d(dtype=float),
               claims: wp.array2d(dtype=int)):
    i = wp.tid()
    if carry[i] == 0:
        cell = grid_cell(g, pos[i])
        if grid_inside(g, cell):
            if claims[cell[1], cell[0]] == i:        # exactly one winner per cell
                food[cell[1], cell[0]] = food[cell[1], cell[0]] - 1
                carry[i] = 1
                carried_q[i] = quality[cell[1], cell[0]]
                timer[i] = 0.0
                lost_clock[i] = 0.0
                frust[i] = 0.0
                if nav[i] > 0.0:
                    nav[i] = 0.0
                ang[i] = ang[i] + 3.14159265


# ------------------------------------------------------------------ worlds
def _centers(nx, ny):
    xs = (np.arange(nx) + 0.5) * CELL
    ys = (np.arange(ny) + 0.5) * CELL
    return np.meshgrid(xs, ys)


def _capsule(X, Y, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    t = np.clip(((X - a[0]) * ba[0] + (Y - a[1]) * ba[1]) / max(ba @ ba, 1e-9), 0, 1)
    return (X - a[0] - ba[0] * t) ** 2 + (Y - a[1] - ba[1] * t) ** 2 < r * r


def build_world(name: str, rng: np.random.Generator, nx: int, ny: int, rich: float = 2.5):
    """Returns (blocked mask, food grid, food quality grid, nest position)."""
    X, Y = _centers(nx, ny)
    blocked = np.zeros((ny, nx), bool)
    food = np.zeros((ny, nx), np.int32)
    quality = np.ones((ny, nx), np.float32)
    if name == "Two foods":
        # Beckers, Deneubourg & Goss (1993): a fork with two equal branches to two feeders,
        # one richer. Which branch is rich is up to the seed.
        nest, fork = (16.0, 45.0), (64.0, 45.0)
        blocked[:] = True
        blocked &= ~_capsule(X, Y, nest, nest, 9.0)
        blocked &= ~_capsule(X, Y, (24.0, 45.0), fork, 2.6)
        rooms = [(140.0, 18.0), (140.0, 72.0)]
        rich_side = int(rng.integers(2))
        for k, room in enumerate(rooms):
            bend = (104.0, room[1])
            blocked &= ~_capsule(X, Y, fork, bend, 2.6)
            blocked &= ~_capsule(X, Y, bend, room, 2.6)
            blocked &= ~_capsule(X, Y, room, room, 8.0)
            pile = _capsule(X, Y, room, room, 4.5)
            food[pile] = 25
            if k == rich_side:
                quality[pile] = rich
    elif name == "Double bridge":
        nest, target = (16.0, 45.0), (144.0, 45.0)
        blocked[:] = True
        for c in (nest, target):
            blocked &= ~_capsule(X, Y, c, c, 9.0)
        short = [(24, 45), (58, 64), (102, 64), (136, 45)]
        long_ = [(24, 45), (40, 12), (70, 5), (90, 5), (120, 12), (136, 45)]
        for path in (short, long_):
            for a, b in zip(path[:-1], path[1:], strict=False):
                blocked &= ~_capsule(X, Y, a, b, 2.6)
        food[_capsule(X, Y, target, target, 4.5)] = 25
    elif name == "Maze":
        cols, rows = 8, 4
        cw, ch = WIDTH / cols, HEIGHT / rows
        blocked[:] = True
        seen = np.zeros((rows, cols), bool)
        stack = [(0, 0)]
        seen[0, 0] = True
        center = lambda c, r: ((c + 0.5) * cw, (r + 0.5) * ch)  # noqa: E731
        blocked &= ~_capsule(X, Y, center(0, 0), center(0, 0), 4.0)
        while stack:                       # randomised depth-first carving
            c, r = stack[-1]
            nbrs = [(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= c + dc < cols and 0 <= r + dr < rows and not seen[r + dr, c + dc]]
            if not nbrs:
                stack.pop()
                continue
            nc, nr = nbrs[rng.integers(len(nbrs))]
            seen[nr, nc] = True
            blocked &= ~_capsule(X, Y, center(c, r), center(nc, nr), 3.4)
            blocked &= ~_capsule(X, Y, center(nc, nr), center(nc, nr), 4.0)
            stack.append((nc, nr))
        nest = center(0, 0)
        tgt = center(cols - 1, rows - 1)
        food[_capsule(X, Y, tgt, tgt, 3.5)] = 25
    else:
        nest = (30.0, 45.0)
        for c, r in (((125, 70), 4.0), ((122, 20), 4.0), ((72, 80), 3.0)):
            food[_capsule(X, Y, c, c, r)] = 8
        placed = 0
        while placed < 6:
            c = rng.uniform((50, 10), (145, 80))
            if min(np.hypot(*(c - np.array(q))) for q in ((30, 45), (125, 70), (122, 20),
                                                           (72, 80))) < 14:
                continue
            a = c + rng.normal(0, 5, 2)
            blocked |= _capsule(X, Y, c, a, rng.uniform(1.5, 3.5))
            placed += 1
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    food[blocked] = 0
    return blocked.astype(np.uint8), food, quality, np.array(nest, np.float32)


class Sim(Simulation):
    world = (0.0, 0.0, WIDTH, HEIGHT)
    background = "soil"
    PARAMS = [
        *section(
            "Colony",
            Param("scenario", "Scenario", "Open field", choices=SCENARIOS, restart=True,
                  help="Double bridge is the classic experiment: two routes, one shorter."),
            Param("n_ants", "Ants", 1500, 50, 20000, 50,
                  "Colony size. More ants find food sooner and keep trails fresher.", "N",
                  restart=True, cpu_default=600),
            Param("speed", "Walking speed", 10.0, 2.0, 25.0, 0.5,
                  "How fast every ant walks.", "v", "u/s"),
            Param("wander", "Wander", 1.5, 0.0, 6.0, 0.1,
                  "Random turning. Too little: no exploring. Too much: no trail following.",
                  "\\sigma"),
            Param("variety", "Individual variety", 0.2, 0.0, 1.0, 0.05,
                  "How different ants are from each other in speed, wander and trail "
                  "loyalty. 0: identical copies. High: bold scouts and faithful followers.",
                  "\\kappa", restart=True),
        ),
        *section(
            "Walls",
            Param("hug_time", "Wall following", 4.0, 0.0, 20.0, 0.5,
                  "After touching a wall an ant keeps it at its side, and looks for it this "
                  "long if it loses it (thigmotaxis). 0: it just slides off.", "T_w", "s"),
            Param("hug_gain", "Corner pull", 1.0, 0.0, 3.0, 0.05,
                  "How hard a wall-following ant turns back toward a wall it lost, so it "
                  "goes round corners.", "g_w"),
            Param("hug_release", "Letting go", 0.1, 0.0, 2.0, 0.05,
                  "Chance per second of leaving a wall, so no ant circles a pillar forever.",
                  "\\lambda_w", "1/s"),
            Param("feel_angle", "Side feeler angle", 50.0, 10.0, 90.0, 1.0,
                  "Where the ant feels for the wall at its side, measured from its heading.",
                  "\\alpha_w", "deg"),
            Param("feel_dist", "Side feeler reach", 1.2, 0.3, 4.0, 0.1,
                  "How far to the side the ant can feel a wall.", "d_w"),
            Param("slide_step", "Slide search step", 17.0, 5.0, 45.0, 1.0,
                  "At a wall, the ant turns in steps of this size until the way along the "
                  "wall is free. Big steps: jerky, wide turns.", "\\Delta_w", "deg"),
        ),
        *section(
            "Sensing",
            Param("sensor_angle", "Sensor angle", 35.0, 5.0, 90.0, 1.0,
                  "Angle between the front sensor and the side sensors.", "\\theta_s", "deg"),
            Param("sensor_dist", "Sensor distance", 2.5, 0.5, 8.0, 0.1,
                  "How far ahead the sensors reach.", "d_s"),
            Param("turn_rate", "Turn rate", 8.0, 0.5, 20.0, 0.5,
                  "How sharply an ant turns toward a stronger smell. Low: it overshoots "
                  "trails.", "\\omega", "rad/s"),
            Param("food_cue", "Food smell", 3.0, 0.0, 10.0, 0.1,
                  "How strongly a sensor touching food pulls a searching ant. At 0 ants "
                  "find food only by bumping into it."),
        ),
        *section(
            "Going home",
            Param("trips", "Ants get hungry", True,
                  help="An ant can only stay out so long. Then, hungry and tired, it walks "
                       "home by compass (violet), eats and rests in the nest, and goes out "
                       "again. Off: a searcher never comes home unless it finds food."),
            Param("trip", "Trip length", 90.0, 20.0, 600.0, 5.0,
                  "How long an ant can stay out before it must go home to eat.",
                  "T_{trip}", "s"),
            Param("rest", "Rest in the nest", 10.0, 0.0, 120.0, 1.0,
                  "How long a hungry ant stays in the nest to eat and rest.", "T_{rest}", "s"),
            Param("homing", "Path integration", 0.5, 0.0, 2.0, 0.05,
                  "Loaded ants also steer toward home by dead reckoning. 0 = pheromone only "
                  "(try it: ants can get stuck circling in an 'ant mill').", "h"),
            Param("nest_cue", "Nest smell", 6.0, 0.0, 15.0, 0.1,
                  "How strongly the nest's own smell pulls loaded ants when they are close."),
            Param("scent_r", "Nest smell range", 10.0, 2.0, 40.0, 0.5,
                  "How far from the nest its smell reaches."),
        ),
        *section(
            "Navigator ants",
            Param("navigators", "Navigator ants", False,
                  help="An ant that notices it is lost (circling with food, or out far too long) "
                       "turns navigator (violet): compass toward home, and around walls in its "
                       "way (the Bug algorithm from robotics)."),
            Param("nav_loops", "Lost after circling", 4.0, 0.5, 10.0, 0.25,
                  "How many full circles of recent turning make an ant carrying food decide "
                  "it is lost (stuck in an ant mill).",
                  "n_{lost}", "turns"),
            Param("give_up", "Lost after searching", 120.0, 5.0, 600.0, 5.0,
                  "An ant out this long without reaching food or home gives up and heads "
                  "home, as real foragers do. Time spent following a wall doesn't count.",
                  "T_{lost}", "s"),
            Param("wind_tau", "Turning memory", 8.0, 0.5, 20.0, 0.5,
                  "How long an ant remembers its own turning. Long: slow, wide loops count "
                  "too.", "\\tau_w", "s"),
            Param("nav_gain", "Compass strength", 3.0, 0.5, 10.0, 0.1,
                  "How hard a navigator steers toward home.", "g_n"),
            Param("nav_time", "Navigate for", 60.0, 5.0, 300.0, 5.0,
                  "How long a navigator ignores smells. If it isn't home by then (a wall in "
                  "the way), it follows smells again for as long before it can retry.",
                  "T_n", "s"),
        ),
        *section(
            "Pheromone",
            Param("deposit", "Pheromone per second", 1.0, 0.0, 5.0, 0.05,
                  "How much scent each ant lays. At 0 the colony has no shared memory.", "q"),
            Param("trail_tau", "Ant memory", 20.0, 1.0, 300.0, 1.0,
                  "Marks weaken with time since the ant left home or found food.",
                  "\\tau_a", "s"),
            Param("evap_tau", "Evaporation time", 40.0, 2.0, 300.0, 1.0,
                  "How long pheromone lasts in the world.", "\\tau_e", "s"),
            Param("diffusion", "Diffusion", 1.0, 0.0, 12.0, 0.1,
                  "How fast pheromone spreads to neighbouring cells.", "D", "cells^2/s"),
        ),
        *section(
            "No-entry marks",
            Param("noentry", "No-entry marks", True,
                  help="Frustrated searchers (long on a food trail, no food) mark the spot "
                       "with a repellent, as Pharaoh's ants do. Others then avoid that "
                       "trail there (red in the overlay)."),
            Param("ne_after", "Frustrated after", 30.0, 2.0, 300.0, 1.0,
                  "Seconds a searcher follows a food trail without finding food before it "
                  "starts marking no-entry. Too short: it marks good trails too.",
                  "T_f", "s"),
            Param("ne_rate", "No-entry per second", 1.0, 0.0, 5.0, 0.05,
                  "How much repellent a frustrated ant lays.", "q_{ne}"),
            Param("ne_weight", "No-entry strength", 1.0, 0.0, 5.0, 0.05,
                  "How strongly searchers avoid no-entry marks when choosing a way.",
                  "k_{ne}"),
            Param("ne_tau", "No-entry fades over", 15.0, 1.0, 120.0, 1.0,
                  "How long a no-entry mark lasts. Real ones fade faster than trails.",
                  "\\tau_{ne}", "s"),
            Param("ne_trail", "On a trail above", 0.5, 0.05, 3.0, 0.05,
                  "The food-trail reading ln(1 + c) that counts as following a trail, for "
                  "frustration.", "S_{trail}"),
        ),
        *section(
            "Food",
            Param("rich", "Rich food quality", 2.5, 1.0, 5.0, 0.1,
                  "How much more strongly ants mark the trail from rich food than from "
                  "ordinary food (quality 1). Used by Two foods and the Rich food tool.",
                  "Q_{rich}", restart=True),
        ),
    ]
    PRESETS = [
        Preset("default", "Default", {}, "The standard colony in an open field."),
        Preset("bridge", "Double bridge", {"scenario": "Double bridge"},
               "Two routes to food, one shorter. Watch the colony pick one."),
        Preset("maze", "Maze", {"scenario": "Maze", "homing": 0.0, "trail_tau": 40.0,
                                "evap_tau": 80.0, "hug_time": 8.0, "navigators": True},
               "Food at the far end of a maze. Longer memories for a long route, walls "
               "followed longer, no compass pulling ants into walls, and ants that give "
               "up a fruitless search and head home (violet)."),
        Preset("forget", "Fast fading", {"evap_tau": 6.0},
               "Trails vanish quickly: the colony struggles to settle on a route."),
        Preset("stubborn", "Long memory", {"evap_tau": 300.0},
               "Trails last: the colony locks onto the first route it finds, even a bad one."),
        Preset("explore", "Explorers", {"wander": 4.5},
               "Restless ants find new food fast but lose trails."),
        Preset("two", "Two foods", {"scenario": "Two foods"},
               "Two piles at the same distance, one richer. Watch the colony pick one."),
        Preset("same", "Identical ants", {"variety": 0.0},
               "Every ant exactly alike, as in classic models. Compare with the default."),
        Preset("mill", "Ant mill", {"homing": 0.0, "trips": False},
               "No sense of direction and no hunger: loaded ants can circle their own trail "
               "forever."),
        Preset("rescue", "Mill rescue", {"homing": 0.0, "trips": False, "navigators": True},
               "Mills start to form, but ants that notice they are circling turn navigator "
               "(violet) and walk home."),
    ]
    OVERLAYS = [
        Overlay("home", "Home trail", True, "Laid by searching ants: points back to the nest."),
        Overlay("food_trail", "Food trail", True, "Laid by ants carrying food."),
        Overlay("noentry", "No-entry marks", True, "Repellent laid by frustrated searchers."),
        Overlay("sensors", "Focus ant sensors", True, "What the focus ant smells."),
        Overlay("memory", "Ant memory", False, "Tint ants by time since they left home/food."),
        Overlay("ants", "Ants", True),
    ]
    TOOLS = [
        Tool("wall", "Wall", "wall", "Drag to build walls", "Drag to erase walls", radius=1.6,
             tip="Block a busy trail and watch the colony find a way around."),
        Tool("food", "Food", "food", "Click to drop a food pile", "Click to remove food",
             radius=3.0, tip="Put food far from the nest: scouts find it by chance, then a "
                             "trail builds up."),
        Tool("rich", "Rich food", "food", "Click to drop rich food", "Click to remove food",
             radius=3.0, tip="Ants mark the way to rich food more strongly. Drop it as far "
                             "away as some ordinary food and watch the colony choose."),
        Tool("inspect", "Inspect", "inspect", "Click an ant to follow it",
             tip="The sensors overlay and Live Math explain this one ant."),
    ]
    EXPERIMENTS = [
        Experiment("deliver", "The first supply line", "Just watch the Open field for a while.",
                   "No ant knew the way. Ants that found food marked the trail home; others "
                   "followed and strengthened it. The map lives in the dirt: stigmergy.",
                   check="exp_deliver"),
        Experiment("bridge", "Win the double bridge",
                   "Set Scenario to Double bridge and wait about a minute.",
                   "Both routes get pheromone, but the short one is refreshed more often, so "
                   "it wins. Goss and Deneubourg saw real Argentine ants do this in 1989.",
                   check="exp_bridge"),
        Experiment("cut", "Cut a busy trail",
                   "Once a trail glows, use Wall (1) to block it. Keep watching.",
                   "Blocked ants wander, find a way around, and the new route is reinforced "
                   "while the old one evaporates. Forgetting matters as much as remembering.",
                   check="exp_cut"),
        Experiment("newfood", "Open a new food source",
                   "With Food (2), drop a pile far from the nest.",
                   "A scout finds it by chance; one loaded trip home lays the first trail, "
                   "and the colony switches over in a positive feedback loop.",
                   check="exp_newfood"),
        Experiment("maze", "Solve the maze",
                   "Pick the Maze preset and wait for 50 deliveries. Then try it with Wall "
                   "following (T_w) at 0.",
                   "No ant knows the maze. Ants that keep a wall at their side sweep the "
                   "corridors instead of bouncing around, and their trails do the rest. "
                   "Real ants follow edges too (thigmotaxis).", check="exp_maze"),
        Experiment("rich", "Choose the richer food",
                   "Pick the Two foods preset (gold = rich) and wait about a minute.",
                   "Ants carrying rich food mark more strongly, so that trail wins even at "
                   "the same distance. Lasius niger colonies choose this way (Beckers et al. "
                   "1993); no ant compares the two piles.", check="exp_rich"),
        Experiment("noentry", "Mark the dead ends",
                   "Pick the Maze preset and watch for red no-entry marks.",
                   "A frustrated searcher, long on a trail with no food, marks the spot. "
                   "Others then stop following the trail there. Pharaoh's ants use such a "
                   "repellent (Robinson et al. 2005).", check="exp_noentry"),
        Experiment("mill", "Make an ant mill",
                   "Pick the Ant mill preset: Path integration (h) at 0, no navigators, "
                   "no hunger.",
                   "Following only each other's trail, ants can circle forever. Army ants "
                   "really do this, until they die of exhaustion. Real ants also use a sense "
                   "of direction home, as h does. Switch Ants get hungry back on and watch "
                   "hungry ants (violet) walk out of the loop."),
        Experiment("rescue", "Break the loop",
                   "Pick the Mill rescue preset and watch for violet navigator ants.",
                   "An ant can't see a mill from inside it, but it can notice that it keeps "
                   "turning the same way. That signal, which the loop can't fake, tells it to "
                   "stop trusting the trail, head home by compass and go round walls in its "
                   "way (the Bug algorithm).", check="exp_rescue"),
        Experiment("hungry", "Why ants go home",
                   "Watch the Open field for a few minutes, until the food runs low.",
                   "An ant can carry only so much fuel. After T_trip it walks home by compass "
                   "(violet) to eat and rest, then goes out again. Without that, lost ants and "
                   "ants stuck circling a rock with food would never come back: switch Ants "
                   "get hungry off and compare the food delivered.", check="exp_hungry"),
        Experiment("amnesia", "A colony without memory",
                   "Set Evaporation time (τ_e) to 2 s and compare deliveries.",
                   "When marks vanish faster than a round trip, no trail can form. "
                   "Stigmergy needs memory that outlasts the journey."),
    ]
    LIVE_MATH = [
        LiveEq("smell", "What its three sensors smell",
               r"S = \ln\!\big(1 + \textstyle\sum c\big) - k_{ne}\ln\!\big(1 + "
               r"\textstyle\sum n\big) + \text{cues}",
               "S: one sensor's reading · c: trail pheromone in the 3×3 cells under it · "
               "ln: natural logarithm (senses respond to ratios) · k_ne: no-entry strength · "
               "n: no-entry marks (searchers only) · cues: food or nest nearby"),
        LiveEq("turn", "Is the strongest smell ahead?", r"S_F \ge \max(S_L,\, S_R)",
               "S_L, S_F, S_R: readings of the left, front and right sensors"),
        LiveEq("mark", "How strongly it marks",
               r"\Delta c = q\,Q\, e^{-t_a/\tau_a}\,\Delta t",
               "Δc: pheromone laid this step · q: pheromone per second · Q: quality of the "
               "food it carries (1 when searching) · t_a: time since it left the nest or "
               "found food · τ_a: ant memory · Δt: one step (1/60 s)"),
        LiveEq("home", "The pull toward home",
               r"\Delta\theta = \omega\,h\,\sin(\theta_{nest} - \theta)\,\Delta t",
               "Δθ: extra turn this step · ω: turn rate · h: path-integration strength · "
               "θ_nest: direction of the nest · θ: its heading"),
        LiveEq("wall", "Is the wall still at my side?",
               r"\text{wall at } \mathbf{x} + d_w\,\hat{\mathbf{u}}(\theta + s_i\,\alpha_w)"
               r"\;\Rightarrow\; \text{keep it}",
               "x: its position · d_w: feeler reach · û(·): a unit direction · θ: heading · "
               "s_i: its wall side, +1 left or −1 right · α_w: feeler angle"),
        LiveEq("traits", "This ant's personality",
               r"v_i = v\,e^{\kappa z_1},\;\; \sigma_i = \sigma\,e^{\kappa z_2},\;\; "
               r"f_i = e^{\kappa z_3}",
               "v_i, σ_i: its own speed and wander · f_i: how loyally it follows trails · "
               "κ: individual variety · z: its random draws, fixed at birth"),
        LiveEq("trip", "Time to go home?",
               r"t_{out} > T_{trip} \;\Rightarrow\; \text{home by compass, rest } T_{rest}",
               "t_out: time since it last left the nest · T_trip: how long it can stay out · "
               "⇒: then · T_rest: how long it rests in the nest"),
        LiveEq("lost", "Am I lost?", r"|W| > 2\pi\, n_{lost} \;\vee\; t_{lost} > T_{lost}",
               "W: how far it has turned lately, in radians (older turning fades over τ_w) · "
               "2π: one full circle · n_lost: circles before it counts as lost · ∨: or · "
               "t_lost: time since it last reached home or food, not counting time along "
               "walls · T_lost: when it gives up"),
    ]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        rng = np.random.default_rng(seed)
        dev = self.device
        self.field = FieldMemory(self.world, CELL, channels=3, device=dev)
        nx, ny = self.field.nx, self.field.ny
        blocked, food_np, quality, self.nest = build_world(self.p.scenario, rng, nx, ny,
                                                           self.p.rich)
        self.quality_np = quality
        self.quality = wp.array(quality, dtype=float, device=dev)
        self.rocks = ObstacleGrid(self.world, CELL, dev, border=True)
        self.rocks.set(blocked)
        self.rocks.protect(self.nest, 5.0)          # never bury the nest
        self.food_total = int(food_np.sum())
        self.food = wp.array(food_np, dtype=int, device=dev)
        self.field.set_blocked(self.rocks.mask)
        self.user_food = np.zeros((ny, nx), bool)     # where the learner dropped food
        self.user_food_added = 0
        self.cut_at = None                            # deliveries when a trail was cut
        self.claims = wp.full((ny, nx), NO_CLAIM, dtype=int, device=dev)
        n = int(self.p.n_ants)
        a = rng.uniform(0, 2 * np.pi, n)
        r = rng.uniform(0, 2.5, n)
        pos = self.nest + np.stack([np.cos(a) * r, np.sin(a) * r], 1)
        self.pos = wp.array(pos.astype(np.float32), dtype=wp.vec2, device=dev)
        self.ang = wp.array(a.astype(np.float32), dtype=float, device=dev)
        self.carry = wp.zeros(n, dtype=int, device=dev)
        self.timer = wp.zeros(n, dtype=float, device=dev)
        self.delivered = wp.zeros(1, dtype=int, device=dev)
        self.wind = wp.zeros(n, dtype=float, device=dev)
        self.nav = wp.zeros(n, dtype=float, device=dev)
        self.rescued = wp.zeros(1, dtype=int, device=dev)
        self.hug = wp.zeros(n, dtype=float, device=dev)
        self.lost_clock = wp.zeros(n, dtype=float, device=dev)
        self.hit_d = wp.zeros(n, dtype=float, device=dev)
        self.frust = wp.zeros(n, dtype=float, device=dev)
        self.carried_q = wp.ones(n, dtype=float, device=dev)
        self.away = wp.zeros(n, dtype=float, device=dev)
        # personalities: speed, wander and trail loyalty factors e^(kappa z), z ~ N(0, 1)
        # (kappa = 0: identical ants), and which side each keeps a wall on
        k = float(self.p.variety)
        self.trait_np = np.ones((n, 4), np.float32)
        self.trait_np[:, :3] = np.exp(k * rng.standard_normal((n, 3)))
        self.trait_np[:, 3] = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        self.traits = wp.array(self.trait_np, dtype=wp.vec4, device=dev)
        self.leg_phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.seed_jitter = rng.uniform(-0.3, 0.3, (ny, nx, 2)).astype(np.float32)
        self._rock_img = None
        self._rock_version = -1
        self.focus = min(self.focus, n - 1)
        self._host = None
        self._apply_field_params()

    def _apply_field_params(self) -> None:
        self.field.tau[:] = self.p.evap_tau
        self.field.tau[NOENTRY] = self.p.ne_tau        # "no entry" fades faster
        self.field.diffusion[:] = self.p.diffusion

    def on_param(self, key: str) -> None:
        self._apply_field_params()

    def _colony(self) -> Colony:
        p = self.p
        c = Colony()
        c.speed, c.sensor_angle = p.speed, float(np.radians(p.sensor_angle))
        c.sensor_dist, c.turn_rate, c.wander = p.sensor_dist, p.turn_rate, p.wander
        c.deposit, c.trail_tau = p.deposit, p.trail_tau
        c.nest = wp.vec2(*self.nest)
        c.nest_r, c.scent_r, c.homing = 3.0, p.scent_r, p.homing
        c.food_cue, c.nest_cue = p.food_cue, p.nest_cue
        c.nav_on = 1 if p.navigators else 0
        c.nav_loops, c.wind_tau = p.nav_loops, p.wind_tau
        c.nav_gain, c.nav_time = p.nav_gain, p.nav_time
        c.give_up = p.give_up
        c.hug_time, c.hug_gain = p.hug_time, p.hug_gain
        c.feel_angle, c.feel_dist = float(np.radians(p.feel_angle)), p.feel_dist
        c.slide_step, c.hug_release = float(np.radians(p.slide_step)), p.hug_release
        c.dt = self.dt
        c.seed = step_seed(self.seed, self.steps)
        return c

    def _trips(self) -> Trips:
        p, t = self.p, Trips()
        t.on = 1 if p.trips else 0
        t.trip, t.rest = p.trip, p.rest
        return t

    def _marks(self) -> Marks:
        p, m = self.p, Marks()
        m.on = 1 if p.noentry else 0
        m.after, m.rate, m.weight, m.trail = p.ne_after, p.ne_rate, p.ne_weight, p.ne_trail
        return m

    # ------------------------------------------------------------ interaction
    def _use_tools(self, inp: InputState) -> None:
        tool = self.tool_of(inp)
        for button, at in inp.clicks:
            if tool == "inspect" and button == "left":
                xy = self.pos.numpy()
                self.focus = int(np.argmin(np.sum((xy - np.asarray(at)) ** 2, axis=1)))
            elif tool in ("food", "rich"):
                food = self.food.numpy()
                disk = self.rocks.capsule(at, at, 3.0) & (self.rocks.mask == 0)
                if button == "left":
                    food[disk] += 8
                    self.quality_np[disk] = self.p.rich if tool == "rich" else 1.0
                    self.food_total += int(8 * disk.sum())
                    self.user_food |= disk
                    self.user_food_added += int(8 * disk.sum())
                else:
                    food[disk] = 0
                    self.quality_np[disk] = 1.0
                self.food = wp.array(food, dtype=int, device=self.device)
                self.quality = wp.array(self.quality_np, dtype=float, device=self.device)
        if tool == "wall" and inp.mouse is not None and {"left", "right"} & inp.buttons:
            building = "left" in inp.buttons
            before = self.rocks.mask.copy() if building else None
            if self.rocks.stroke(inp.mouse, 1.6, 1 if building else 0):
                self.field.set_blocked(self.rocks.mask)
                if building and self.cut_at is None:
                    new = (self.rocks.mask == 1) & (before == 0)
                    trail = self.field.numpy()[FOOD]
                    if new.any() and trail[new].max() > 2.0:     # it landed on a busy trail
                        self.cut_at = int(self.delivered.numpy()[0])
        else:
            self.rocks.end_stroke()

    def step(self, inp: InputState) -> None:
        self._use_tools(inp)
        dev, g, f = self.device, self.field.grid, self.field
        self.claims.fill_(NO_CLAIM)
        wp.launch(ant_step, dim=len(self.pos),
                  inputs=[self._colony(), self._marks(), self._trips(), g, self.pos, self.ang,
                          self.carry, self.timer,
                          f.value, f.deposit, self.food, self.rocks.blocked, self.claims,
                          self.delivered, self.wind, self.nav, self.rescued, self.hug,
                          self.lost_clock, self.traits, self.hit_d, self.frust,
                          self.carried_q, self.away], device=dev)
        wp.launch(ant_pickup, dim=len(self.pos),
                  inputs=[g, self.pos, self.ang, self.carry, self.timer, self.lost_clock,
                          self.nav, self.frust, self.carried_q, self.food, self.quality,
                          self.claims], device=dev)
        f.update(self.dt)
        self._host = None

    def _fetch(self):
        if self._host is None:
            self._host = dict(pos=self.pos.numpy(), ang=self.ang.numpy(),
                              carry=self.carry.numpy(), timer=self.timer.numpy(),
                              field=self.field.numpy(), food=self.food.numpy(),
                              delivered=int(self.delivered.numpy()[0]),
                              wind=self.wind.numpy(), nav=self.nav.numpy(),
                              hug=self.hug.numpy(), lost=self.lost_clock.numpy(),
                              frust=self.frust.numpy(), cq=self.carried_q.numpy(),
                              away=self.away.numpy(),
                              rescued=int(self.rescued.numpy()[0]))
        return self._host

    def state_arrays(self):
        h = self._fetch()
        return [h["pos"], h["ang"], h["carry"], h["field"], h["food"], h["nav"], h["hug"],
                h["lost"], h["frust"], h["cq"], h["away"], self.trait_np, self.quality_np, self.rocks.mask]

    # ------------------------------------------------------------------ draw
    def draw(self, s) -> None:
        h = self._fetch()
        pos, ang, carry = h["pos"], h["ang"], h["carry"]
        field = h["field"]
        s.background("soil")
        b = self.world

        if self._rock_version != self.rocks.version:
            self._rock_img = self.rocks.rock_image(seed=self.seed)
            self._rock_version = self.rocks.version
        k = 0.22
        home = (1 - np.exp(-k * field[HOME])) if self.show.home else 0 * field[HOME]
        food_t = (1 - np.exp(-k * field[FOOD])) if self.show.food_trail else 0 * field[FOOD]
        pher = np.zeros(field.shape[1:] + (4,), np.float32)
        pher[..., 0] = home * 0.25 + food_t * 1.6
        pher[..., 1] = home * 0.55 + food_t * 0.95
        pher[..., 2] = home * 1.6 + food_t * 0.20
        pher[..., 3] = np.clip(np.maximum(home, food_t), 0, 1)
        s.image(pher, b, additive=True)
        if self.show.noentry and field[NOENTRY].max() > 1e-3:
            ne = 1 - np.exp(-0.6 * field[NOENTRY])
            red = np.zeros(field.shape[1:] + (4,), np.float32)
            red[..., 0], red[..., 1], red[..., 2] = 1.0, 0.18, 0.22
            red[..., 3] = np.clip(ne, 0, 0.85)
            s.image(red, b)

        food = h["food"]
        fy, fx = np.nonzero(food > 0)
        if len(fx):
            jit = self.seed_jitter[fy, fx]
            seeds = np.stack([(fx + 0.5 + jit[:, 0]) * CELL, (fy + 0.5 + jit[:, 1]) * CELL], 1)
            amt = np.clip(food[fy, fx] / 8.0, 0.25, 1.0)
            rich = self.quality_np[fy, fx] > 1.0
            col = np.where(rich[:, None], pal.rgba(RICH_COLOR, 1.0, 1.35),
                           pal.rgba("#bef264", 1.0, 1.25)).astype(np.float32)
            s.circles(seeds, 0.20 * np.sqrt(amt) + 0.05, col)
        s.image(self._rock_img, b, mode="mask")
        s.glow(self.nest, 7.0, pal.rgba(pal.AMBER, 0.25, 1.2))
        s.circles(self.nest, 3.0, pal.rgba("#0b0704"))
        s.circles(self.nest, 3.0, pal.rgba(pal.AMBER, 0.9, 1.5), ring=0.35)

        if self.show.ants:
            dt_vis = self.dt
            self.leg_phase += dt_vis * 22.0
            col = np.tile(pal.rgba("#a0522d"), (len(pos), 1))
            if self.show.memory:
                t = np.clip(h["timer"] / self.p.trail_tau, 0, 1)
                col = pal.lerp_colors("#fde68a", "#6b21a8", t)
            navs = self._homebound(h)
            if navs.any():
                col = col.copy()
                col[navs] = pal.rgba(NAV_COLOR, 1.0, 1.15)
            s.sprites("ant", pos, ang, (1.5, 0.8), col, self.leg_phase)
            idx = np.nonzero(carry == 1)[0]
            if len(idx):
                tip = pos[idx] + np.stack([np.cos(ang[idx]), np.sin(ang[idx])], 1) * 0.75
                s.circles(tip, 0.28, pal.rgba("#86efac", 1.0, 1.6))

        f = self.focus
        if self.show.sensors and f < len(pos):
            self._draw_sensors(s, pos[f], ang[f], carry[f], field, food)
            s.circles(pos[f], 1.6, pal.rgba("#ffffff", 0.9, 1.5), ring=0.12)

    def _sense(self, p, a, c, field, food):
        """Host mirror of the kernel's three sensors: (points, values)."""
        P = self.p
        sa, sd = np.radians(P.sensor_angle), P.sensor_dist
        pts = [p + sd * np.array([np.cos(a + o), np.sin(a + o)]) for o in (sa, 0.0, -sa)]
        vals = []
        ch = FOOD if c == 0 else HOME
        for q in pts:
            ix, iy = int(np.floor(q[0] / CELL)), int(np.floor(q[1] / CELL))
            y0, y1 = max(iy - 1, 0), min(iy + 2, field.shape[1])
            x0, x1 = max(ix - 1, 0), min(ix + 2, field.shape[2])
            v = np.log1p(field[ch, y0:y1, x0:x1].sum())      # same rule as the kernel
            if c == 0:
                if P.noentry:
                    v -= P.ne_weight * np.log1p(field[NOENTRY, y0:y1, x0:x1].sum())
                v += P.food_cue * (food[y0:y1, x0:x1] > 0).sum()
            else:
                dn = np.hypot(*(q - self.nest))
                v += P.nest_cue * max(0.0, 1 - dn / P.scent_r)
            vals.append(v)
        return pts, np.array(vals)

    def _draw_sensors(self, s, p, a, c, field, food) -> None:
        pts, vals = self._sense(p, a, c, field, food)
        best = int(np.argmax(vals)) if vals.max() > 0 else 1
        rel = vals / max(vals.max(), 1e-6)
        for k, q in enumerate(pts):
            col = pal.rgba(pal.ATTRACT if k == best else pal.SENSE, 0.9, 1.4 if k == best else 1.0)
            s.lines(p, q, 0.06, pal.rgba(pal.SENSE, 0.5))
            s.circles(q, 0.75, col, ring=0.08)
            s.circles(q, 0.75 * rel[k], pal.rgba(pal.ATTRACT, 0.35))

    def _homebound(self, h) -> np.ndarray:
        """Ants walking home by compass: lost navigators, and hungry ants."""
        tired = (h["away"] > self.p.trip) if self.p.trips else False
        return (h["nav"] > 0) | tired

    def hud(self) -> list[HudItem]:
        h = self._fetch()
        carry = h["carry"]
        left = int(h["food"].sum())
        return [
            HudItem("ants", f"{len(carry):,}"),
            HudItem("carrying food", f"{100 * carry.mean():.0f}%"),
            HudItem("delivered", f"{h['delivered']:,}", accent=True),
            HudItem("following walls", f"{100 * (h['hug'] > 0).mean():.0f}%"),
            *([HudItem("hungry, going home", f"{int((h['away'] > self.p.trip).sum()):,}"),
               HudItem("resting in the nest", f"{int((h['away'] < 0).sum()):,}")]
              if self.p.trips else []),
            *([HudItem("navigating home", f"{int((h['nav'] > 0).sum()):,} · "
                                          f"{h['rescued']:,} got home")]
              if self.p.navigators else []),
            HudItem("food left", f"{left:,} / {self.food_total:,}"),
            HudItem("time", f"{self.t:.0f} s"),
        ]

    # ------------------------------------------------------------- live math
    def live_math(self) -> dict[str, LiveValue]:
        h = self._fetch()
        f, P = self.focus, self.p
        p, a, c = h["pos"][f], float(h["ang"][f]), int(h["carry"][f])
        _, (sl, sf, sr) = self._sense(p, a, c, h["field"], h["food"])
        trail = "home trail" if c else "food trail"
        out = {"smell": LiveValue(f"S_L {fmt(sl)}   S_F {fmt(sf)}   S_R {fmt(sr)}", None,
                                  f"{'carrying food' if c else 'searching'}: it smells the "
                                  f"{trail}")}
        ahead = sf >= max(sl, sr)
        side = "left" if sl > sr else ("right" if sr > sl else "nowhere: a tie")
        out["turn"] = LiveValue(f"S_F = {fmt(sf)} {'≥' if ahead else '<'} max = "
                                f"{fmt(max(sl, sr))}", ahead,
                                "straight on: the best smell is ahead" if ahead
                                else f"turn {side}, toward the stronger smell")
        ta = float(h["timer"][f])
        q = float(h["cq"][f]) if c else 1.0
        rate = P.deposit * q * np.exp(-ta / P.trail_tau)
        out["mark"] = LiveValue(f"t_a = {fmt(ta, 1)} s, Q = {fmt(q, 1)} → {fmt(P.deposit)}·"
                                f"{fmt(q, 1)}·e^(−{fmt(ta, 1)}/{fmt(P.trail_tau, 0)}) = "
                                f"{fmt(rate)} per s", None,
                                "fresh from the nest or food: strong marks" if rate > 0.5 * P.deposit
                                else "long trip so far: faint marks (long routes lose)")
        to = self.nest - p
        err = float(np.arctan2(np.sin(np.arctan2(to[1], to[0]) - a),
                               np.cos(np.arctan2(to[1], to[0]) - a)))
        nav = float(h["nav"][f])
        aw = float(h["away"][f])
        tired = P.trips and aw > P.trip
        if aw < 0:
            out["trip"] = LiveValue(f"resting in the nest: {-aw:.0f} s left", None,
                                    "it came home hungry: eating and resting")
        elif not P.trips:
            out["trip"] = LiveValue(f"t_out = {aw:.0f} s", None,
                                    "hunger is off (Ants get hungry): it can stay out forever")
        else:
            out["trip"] = LiveValue(f"t_out = {aw:.0f} s {'>' if tired else '≤'} T_trip = "
                                    f"{P.trip:.0f} s", tired,
                                    "hungry: walking home by compass" if tired else
                                    f"{P.trip - aw:.0f} s of fuel left")
        if tired:
            out["home"] = LiveValue(f"nest is {np.degrees(err):+.0f}° off → turns "
                                    f"{fmt(P.turn_rate * P.nav_gain * np.sin(err))} rad/s",
                                    None, "hungry: compass only, smells ignored")
        elif nav > 0:
            out["home"] = LiveValue(f"nest is {np.degrees(err):+.0f}° off → navigator turns "
                                    f"{fmt(P.turn_rate * P.nav_gain * np.sin(err))} rad/s",
                                    None, "navigating: compass only, smells ignored")
        elif c:
            out["home"] = LiveValue(f"nest is {np.degrees(err):+.0f}° off → turn "
                                    f"{fmt(P.turn_rate * P.homing * np.sin(err))} rad/s", None,
                                    "path integration: it knows roughly where home is")
        else:
            out["home"] = LiveValue("not carrying food: no pull home", None,
                                    "searching ants go wherever the smell leads")
        wv, limit = float(h["wind"][f]), 2 * np.pi * P.nav_loops
        lc = float(h["lost"][f])
        text = (f"|W| = {fmt(abs(wv), 1)} of {fmt(limit, 1)} rad · t_lost = {lc:.0f} of "
                f"{P.give_up:.0f} s")
        if nav > 0:
            out["lost"] = LiveValue(f"was lost → navigating home, {nav:.0f} s left", True,
                                    "it noticed it was lost and switched to its compass")
        elif not P.navigators:
            out["lost"] = LiveValue(text, None, "navigators are off (switch them on under "
                                                "Navigator ants): it just follows smells")
        elif nav < 0:
            out["lost"] = LiveValue(text, False, "its compass run failed (a wall?): trusting "
                                                 f"smells for {-nav:.0f} s before it retries")
        else:
            out["lost"] = LiveValue(text, False, "on track: it trusts the smells")
        tr = self.trait_np[f]
        out["traits"] = LiveValue(f"v_i = {fmt(P.speed * tr[0], 1)}   σ_i = "
                                  f"{fmt(P.wander * tr[1])}   f_i = {fmt(tr[2])}   wall on "
                                  f"its {'left' if tr[3] > 0 else 'right'}", None,
                                  "a bold scout: fast, restless, loosely loyal"
                                  if tr[0] > 1.1 and tr[1] > 1.1 and tr[2] < 1 else
                                  "a faithful follower: sticks to trails" if tr[2] > 1.15 else
                                  "an ordinary worker" if P.variety > 0 else
                                  "variety is 0: every ant is identical")
        hug = float(h["hug"][f])
        side = "left" if tr[3] > 0 else "right"
        if hug > 0:
            out["wall"] = LiveValue(f"following a wall on its {side}: {hug:.1f} s of "
                                    f"searching left if it loses it", True,
                                    "thigmotaxis: the wall guides it along the corridor")
        else:
            out["wall"] = LiveValue(f"no wall at its {side}", None,
                                    "letting go for a moment" if hug < 0 else
                                    "open ground: it walks by smell alone")
        return out

    # ------------------------------------------------------------ experiments
    def exp_deliver(self) -> bool:
        return self._fetch()["delivered"] >= 100

    def exp_bridge(self) -> bool:
        if self.p.scenario != "Double bridge" or self.t < 30:
            return False
        pos = self._fetch()["pos"]
        mid = (pos[:, 0] > 35) & (pos[:, 0] < 125)
        short = int((mid & (pos[:, 1] > 52)).sum())
        long_ = int((mid & (pos[:, 1] < 30)).sum())
        return short + long_ >= 40 and short >= 3 * long_

    def exp_rich(self) -> bool:
        if self.p.scenario != "Two foods" or self.t < 45:
            return False
        h = self._fetch()
        loaded = h["carry"] == 1
        return loaded.sum() >= 30 and (h["cq"][loaded] > 1.0).mean() >= 0.75

    def exp_noentry(self) -> bool:
        return self.p.noentry and float(self._fetch()["field"][NOENTRY].max()) > 2.0

    def exp_hungry(self) -> bool:
        return self.p.trips and int((self._fetch()["away"] < 0).sum()) >= 30

    def exp_maze(self) -> bool:
        return self.p.scenario == "Maze" and self._fetch()["delivered"] >= 50

    def exp_rescue(self) -> bool:
        return self.p.navigators and self.p.homing < 0.05 and self._fetch()["rescued"] >= 25

    def exp_cut(self) -> bool:
        return self.cut_at is not None and self._fetch()["delivered"] >= self.cut_at + 60

    def exp_newfood(self) -> bool:
        if not self.user_food_added:
            return False
        left = int(self._fetch()["food"][self.user_food].sum())
        return self.user_food_added - left >= 20
