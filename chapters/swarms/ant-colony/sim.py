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
from ailab.core.sandbox import ObstacleGrid, obstacle_at
from ailab.render import palette as pal

WIDTH, HEIGHT, CELL = 160.0, 90.0, 0.5
HOME, FOOD = 0, 1
NO_CLAIM = 2**31 - 1
SCENARIOS = ("Open field", "Double bridge", "Maze")
NAV_COLOR = "#c4b5fd"   # navigator ants: violet


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
    homing: float
    dt: float
    seed: int


@wp.func
def smell(P: Colony, g: Grid2D, value: wp.array3d(dtype=float), food: wp.array2d(dtype=int),
          q: wp.vec2, carrying: int) -> float:
    # Weber-Fechner: sensors respond to the logarithm of concentration, so a busy trail
    # never drowns out the direct smell of food or of the nest.
    s = float(0.0)
    if carrying == 0:
        s = wp.log(1.0 + field_sample3(value, 1, g, q))     # follow the food trail
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


@wp.kernel
def ant_step(P: Colony, g: Grid2D, pos: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
             carry: wp.array(dtype=int), timer: wp.array(dtype=float),
             value: wp.array3d(dtype=float), deposit: wp.array3d(dtype=wp.int32),
             food: wp.array2d(dtype=int), blocked: wp.array2d(dtype=wp.uint8),
             claims: wp.array2d(dtype=int), delivered: wp.array(dtype=int),
             wind: wp.array(dtype=float), nav: wp.array(dtype=float),
             rescued: wp.array(dtype=int)):
    i = wp.tid()
    p = pos[i]
    a = ang[i]
    c = carry[i]
    tm = timer[i]
    w = wind[i]
    nv = nav[i]
    rng = wp.rand_init(P.seed, i)
    to_nest = P.nest - p
    home_err = wp.sin(wp.atan2(to_nest[1], to_nest[0]) - a)

    # nav > 0: navigating (seconds left); nav < 0: resting from a failed try (can't retrigger)
    turn = float(0.0)
    if nv > 0.0:
        # a navigator ignores smells and walks home by its inner compass
        turn = P.nav_gain * home_err
        nv = nv - P.dt
        if nv <= 0.0:
            nv = -P.nav_time                      # didn't make it home: trust smells a while
            w = 0.0
    else:
        nv = wp.min(nv + P.dt, 0.0)
        # 1. sense: three sensors ahead-left, ahead, ahead-right
        sa = P.sensor_angle
        sd = P.sensor_dist
        s_l = smell(P, g, value, food, p + wp.vec2(wp.cos(a + sa), wp.sin(a + sa)) * sd, c)
        s_f = smell(P, g, value, food, p + wp.vec2(wp.cos(a), wp.sin(a)) * sd, c)
        s_r = smell(P, g, value, food, p + wp.vec2(wp.cos(a - sa), wp.sin(a - sa)) * sd, c)

        # 2. decide: turn toward the strongest smell, plus a little random wandering
        if s_f < s_l or s_f < s_r:
            if s_l > s_r:
                turn = 1.0
            elif s_r > s_l:
                turn = -1.0
        # path integration: loaded ants also feel the direction home (breaks "ant mills")
        if c == 1:
            turn = turn + P.homing * home_err
        # going in circles? recent steering adds up; old turning fades
        w = w * wp.exp(-P.dt / P.wind_tau) + turn * P.turn_rate * P.dt
        lost = wp.abs(w) > 6.28318531 * P.nav_loops or tm > P.give_up
        if P.nav_on == 1 and nv == 0.0 and lost:
            nv = P.nav_time                       # lost: become a navigator
            w = 0.0
    a = a + turn * P.turn_rate * P.dt + (wp.randf(rng) - 0.5) * 2.0 * P.wander * wp.sqrt(P.dt)

    # 3. act: move, or turn around at walls
    step = wp.vec2(wp.cos(a), wp.sin(a)) * (P.speed * P.dt)
    if obstacle_at(blocked, g, p + step):
        a = a + 3.14159265 + (wp.randf(rng) - 0.5) * 1.2
    else:
        p = p + step

    # 4. mark: strength fades with time since the ant left home / found food
    ch = int(0)
    if c == 1:
        ch = 1
    if nv <= 0.0:                                 # navigators don't feed the loop
        field_deposit(deposit, ch, g, p, P.deposit * wp.exp(-tm / P.trail_tau) * P.dt)
    tm = tm + P.dt

    if wp.length(p - P.nest) < P.nest_r:
        if nv > 0.0:
            wp.atomic_add(rescued, 0, 1)
        nv = 0.0
        w = 0.0
        if c == 1:
            c = 0
            wp.atomic_add(delivered, 0, 1)
            a = a + 3.14159265
        tm = 0.0
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


@wp.kernel
def ant_pickup(g: Grid2D, pos: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
               carry: wp.array(dtype=int), timer: wp.array(dtype=float),
               food: wp.array2d(dtype=int), claims: wp.array2d(dtype=int)):
    i = wp.tid()
    if carry[i] == 0:
        cell = grid_cell(g, pos[i])
        if grid_inside(g, cell):
            if claims[cell[1], cell[0]] == i:        # exactly one winner per cell
                food[cell[1], cell[0]] = food[cell[1], cell[0]] - 1
                carry[i] = 1
                timer[i] = 0.0
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


def build_world(name: str, rng: np.random.Generator, nx: int, ny: int):
    """Returns (blocked mask, food grid, nest position)."""
    X, Y = _centers(nx, ny)
    blocked = np.zeros((ny, nx), bool)
    food = np.zeros((ny, nx), np.int32)
    if name == "Double bridge":
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
    return blocked.astype(np.uint8), food, np.array(nest, np.float32)


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
            Param("navigators", "Navigator ants", True,
                  help="An ant that notices it is lost (circling, or out far too long) stops "
                       "following smells and walks home by its inner compass (violet)."),
            Param("nav_loops", "Lost after circling", 1.5, 0.5, 5.0, 0.25,
                  "How many full circles of recent turning make an ant decide it is lost.",
                  "n_{lost}", "turns"),
            Param("give_up", "Lost after searching", 60.0, 5.0, 300.0, 5.0,
                  "An ant out this long without reaching food or home gives up and heads "
                  "home, as real foragers do.", "T_{lost}", "s"),
            Param("wind_tau", "Turning memory", 8.0, 0.5, 20.0, 0.5,
                  "How long an ant remembers its own turning. Long: slow, wide loops count "
                  "too.", "\\tau_w", "s"),
            Param("nav_gain", "Compass strength", 3.0, 0.5, 10.0, 0.1,
                  "How hard a navigator steers toward home.", "g_n"),
            Param("nav_time", "Navigate for", 10.0, 1.0, 60.0, 1.0,
                  "How long a navigator ignores smells. If it isn't home by then (a wall in "
                  "the way), it follows smells again for as long before it can retry.",
                  "T_n", "s"),
        ),
        *section(
            "Pheromone",
            Param("deposit", "Pheromone per second", 1.0, 0.0, 5.0, 0.05,
                  "How much scent each ant lays. At 0 the colony has no shared memory.", "q"),
            Param("trail_tau", "Ant memory", 20.0, 1.0, 120.0, 1.0,
                  "Marks weaken with time since the ant left home or found food.",
                  "\\tau_a", "s"),
            Param("evap_tau", "Evaporation time", 40.0, 2.0, 300.0, 1.0,
                  "How long pheromone lasts in the world.", "\\tau_e", "s"),
            Param("diffusion", "Diffusion", 1.0, 0.0, 12.0, 0.1,
                  "How fast pheromone spreads to neighbouring cells.", "D", "cells^2/s"),
        ),
    ]
    PRESETS = [
        Preset("default", "Default", {}, "The standard colony in an open field."),
        Preset("bridge", "Double bridge", {"scenario": "Double bridge"},
               "Two routes to food, one shorter. Watch the colony pick one."),
        Preset("maze", "Maze", {"scenario": "Maze"}, "Food hidden behind walls."),
        Preset("forget", "Fast fading", {"evap_tau": 6.0},
               "Trails vanish quickly: the colony struggles to settle on a route."),
        Preset("stubborn", "Long memory", {"evap_tau": 300.0},
               "Trails last: the colony locks onto the first route it finds, even a bad one."),
        Preset("explore", "Explorers", {"wander": 4.5},
               "Restless ants find new food fast but lose trails."),
        Preset("mill", "Ant mill", {"homing": 0.0, "navigators": False},
               "No sense of direction: loaded ants can circle their own trail forever."),
        Preset("rescue", "Mill rescue", {"homing": 0.0},
               "Mills start to form, but ants that notice they are circling turn navigator "
               "(violet) and walk home."),
    ]
    OVERLAYS = [
        Overlay("home", "Home trail", True, "Laid by searching ants: points back to the nest."),
        Overlay("food_trail", "Food trail", True, "Laid by ants carrying food."),
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
        Experiment("mill", "Make an ant mill",
                   "Pick the Ant mill preset: Path integration (h) at 0 and no navigators.",
                   "Following only each other's trail, ants can circle forever. Army ants "
                   "really do this. Real ants also use a sense of direction home, as h does."),
        Experiment("rescue", "Break the loop",
                   "Pick the Mill rescue preset and watch for violet navigator ants.",
                   "An ant can't see a mill from inside it, but it can notice that it keeps "
                   "turning the same way. That signal, which the loop can't fake, tells it to "
                   "stop trusting the trail and use its compass.", check="exp_rescue"),
        Experiment("amnesia", "A colony without memory",
                   "Set Evaporation time (τ_e) to 2 s and compare deliveries.",
                   "When marks vanish faster than a round trip, no trail can form. "
                   "Stigmergy needs memory that outlasts the journey."),
    ]
    LIVE_MATH = [
        LiveEq("smell", "What its three sensors smell", r"S = \ln\!\big(1 + \textstyle\sum c\big)"
               r" + \text{cues}",
               "S: one sensor's reading · c: pheromone in the 3×3 cells under it · "
               "ln: natural logarithm (senses respond to ratios) · cues: food or nest nearby"),
        LiveEq("turn", "Is the strongest smell ahead?", r"S_F \ge \max(S_L,\, S_R)",
               "S_L, S_F, S_R: readings of the left, front and right sensors"),
        LiveEq("mark", "How strongly it marks", r"\Delta c = q\, e^{-t_a/\tau_a}\,\Delta t",
               "Δc: pheromone laid this step · q: pheromone per second · t_a: time since it "
               "left the nest or found food · τ_a: ant memory · Δt: one step (1/60 s)"),
        LiveEq("home", "The pull toward home",
               r"\Delta\theta = \omega\,h\,\sin(\theta_{nest} - \theta)\,\Delta t",
               "Δθ: extra turn this step · ω: turn rate · h: path-integration strength · "
               "θ_nest: direction of the nest · θ: its heading"),
        LiveEq("lost", "Am I lost?", r"|W| > 2\pi\, n_{lost} \;\vee\; t_a > T_{lost}",
               "W: how far it has turned lately, in radians (older turning fades over τ_w) · "
               "2π: one full circle · n_lost: circles before it counts as lost · ∨: or · "
               "t_a: time since it left home or found food · T_lost: when it gives up"),
    ]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        rng = np.random.default_rng(seed)
        dev = self.device
        self.field = FieldMemory(self.world, CELL, channels=2, device=dev)
        nx, ny = self.field.nx, self.field.ny
        blocked, food_np, self.nest = build_world(self.p.scenario, rng, nx, ny)
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
        self.leg_phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.seed_jitter = rng.uniform(-0.3, 0.3, (ny, nx, 2)).astype(np.float32)
        self._rock_img = None
        self._rock_version = -1
        self.focus = min(self.focus, n - 1)
        self._host = None
        self._apply_field_params()

    def _apply_field_params(self) -> None:
        self.field.tau[:] = self.p.evap_tau
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
        c.dt = self.dt
        c.seed = int((self.seed * 1_000_003 + self.steps) % (2**31 - 1))
        return c

    # ------------------------------------------------------------ interaction
    def _use_tools(self, inp: InputState) -> None:
        tool = self.tool_of(inp)
        for button, at in inp.clicks:
            if tool == "inspect" and button == "left":
                xy = self.pos.numpy()
                self.focus = int(np.argmin(np.sum((xy - np.asarray(at)) ** 2, axis=1)))
            elif tool == "food":
                food = self.food.numpy()
                disk = self.rocks.capsule(at, at, 3.0) & (self.rocks.mask == 0)
                if button == "left":
                    food[disk] += 8
                    self.food_total += int(8 * disk.sum())
                    self.user_food |= disk
                    self.user_food_added += int(8 * disk.sum())
                else:
                    food[disk] = 0
                self.food = wp.array(food, dtype=int, device=self.device)
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
                  inputs=[self._colony(), g, self.pos, self.ang, self.carry, self.timer,
                          f.value, f.deposit, self.food, self.rocks.blocked, self.claims,
                          self.delivered, self.wind, self.nav, self.rescued], device=dev)
        wp.launch(ant_pickup, dim=len(self.pos),
                  inputs=[g, self.pos, self.ang, self.carry, self.timer, self.food,
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
                              rescued=int(self.rescued.numpy()[0]))
        return self._host

    def state_arrays(self):
        h = self._fetch()
        return [h["pos"], h["ang"], h["carry"], h["field"], h["food"], h["nav"],
                self.rocks.mask]

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

        food = h["food"]
        fy, fx = np.nonzero(food > 0)
        if len(fx):
            jit = self.seed_jitter[fy, fx]
            seeds = np.stack([(fx + 0.5 + jit[:, 0]) * CELL, (fy + 0.5 + jit[:, 1]) * CELL], 1)
            amt = np.clip(food[fy, fx] / 8.0, 0.25, 1.0)
            s.circles(seeds, 0.20 * np.sqrt(amt) + 0.05, pal.rgba("#bef264", 1.0, 1.25))
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
            navs = h["nav"] > 0
            if navs.any():
                col = col.copy()
                col[navs] = pal.rgba(NAV_COLOR, 1.0, 1.5)
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

    def hud(self) -> list[HudItem]:
        h = self._fetch()
        carry = h["carry"]
        left = int(h["food"].sum())
        return [
            HudItem("ants", f"{len(carry):,}"),
            HudItem("carrying food", f"{100 * carry.mean():.0f}%"),
            HudItem("delivered", f"{h['delivered']:,}", accent=True),
            HudItem("navigating home", f"{int((h['nav'] > 0).sum()):,} · "
                                       f"{h['rescued']:,} got home"),
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
        rate = P.deposit * np.exp(-ta / P.trail_tau)
        out["mark"] = LiveValue(f"t_a = {fmt(ta, 1)} s → {fmt(P.deposit)}·e^(−{fmt(ta, 1)}/"
                                f"{fmt(P.trail_tau, 0)}) = {fmt(rate)} per s", None,
                                "fresh from the nest or food: strong marks" if rate > 0.5 * P.deposit
                                else "long trip so far: faint marks (long routes lose)")
        to = self.nest - p
        err = float(np.arctan2(np.sin(np.arctan2(to[1], to[0]) - a),
                               np.cos(np.arctan2(to[1], to[0]) - a)))
        nav = float(h["nav"][f])
        if nav > 0:
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
        text = (f"|W| = {fmt(abs(wv), 1)} of {fmt(limit, 1)} rad · t_a = {ta:.0f} of "
                f"{P.give_up:.0f} s")
        if nav > 0:
            out["lost"] = LiveValue(f"was lost → navigating home, {nav:.0f} s left", True,
                                    "it noticed it was lost and switched to its compass")
        elif not P.navigators:
            out["lost"] = LiveValue(text, None, "navigators are off: it follows smells "
                                                "wherever they lead, even in circles")
        elif nav < 0:
            out["lost"] = LiveValue(text, False, "its compass run failed (a wall?): trusting "
                                                 f"smells for {-nav:.0f} s before it retries")
        else:
            out["lost"] = LiveValue(text, False, "on track: it trusts the smells")
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

    def exp_rescue(self) -> bool:
        return self.p.navigators and self.p.homing < 0.05 and self._fetch()["rescued"] >= 25

    def exp_cut(self) -> bool:
        return self.cut_at is not None and self._fetch()["delivered"] >= self.cut_at + 60

    def exp_newfood(self) -> bool:
        if not self.user_food_added:
            return False
        left = int(self._fetch()["food"][self.user_food].sum())
        return self.user_food_added - left >= 20
