"""Fish school under predation, in a tank you can build in.

Every fish runs the same local rules (Reynolds boids with Couzin's zones):
  repulsion zone   -> move away       (don't collide)
  orientation zone -> match heading   (swim together)
  attraction zone  -> move closer     (don't get left behind)
plus senses (vision cone with a blind spot, a 360 degree lateral line), fear that spreads
between neighbours, a *working memory* of where the predator was last seen, a feel for
rock (pressure up close, a look ahead) and hunger, which makes fish trade safety for food.

The GPU kernel reads the previous state and writes a new buffer (double buffering),
so no fish ever sees a half-updated neighbour. Bites of food are counted with integer
atomics and applied after the step, so the run is deterministic.
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
    Simulation,
    Tool,
)
from ailab.core.memory import Grid2D, TraceMemory, forget
from ailab.core.params import fmt
from ailab.core.sandbox import ObstacleGrid, obstacle_at, obstacle_clearance, obstacle_push
from ailab.render import palette as pal

WIDTH, HEIGHT = 160.0, 90.0
ROCK_CELL = 1.0
MAX_PELLETS = 192
PELLET_BITES = 6
N_FORCES = 7  # separation, alignment, cohesion, flee, tank, rock, food
SCENARIOS = ("Open tank", "Reef", "Channel")
LOOK_SPREAD = wp.constant(0.6)  # radians between the straight-ahead look and the side looks


@wp.struct
class School:
    r_rep: float
    r_ori: float
    r_att: float
    cos_fov: float
    r_lat: float
    w_sep: float
    w_ali: float
    w_coh: float
    w_flee: float
    w_wall: float
    w_rock: float
    r_rock: float
    look: float
    w_food: float
    r_food: float
    eat_r: float
    hunger_tau: float
    bite: float
    risk: float
    n_pellets: int
    cruise: float
    burst: float
    max_acc: float
    k_speed: float
    r_pred: float
    flee_side: float
    fear_tau: float
    contagion: float
    mem_tau: float
    r_catch: float
    v_attack: float
    width: float
    height: float
    margin: float
    dt: float
    pred_pos: wp.vec2
    pred_vel: wp.vec2
    pred_on: int


@wp.func
def rotate(v: wp.vec2, a: float) -> wp.vec2:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec2(v[0] * c - v[1] * s, v[0] * s + v[1] * c)


@wp.kernel
def school_step(grid: wp.uint64, P: School, g: Grid2D, blocked: wp.array2d(dtype=wp.uint8),
                pellet_pos: wp.array(dtype=wp.vec2), pellet_amt: wp.array(dtype=wp.int32),
                bites: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.int32),
                pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec2),
                fear: wp.array(dtype=float), mem_pos: wp.array(dtype=wp.vec2),
                mem_str: wp.array(dtype=float), hunger: wp.array(dtype=float),
                pos_out: wp.array(dtype=wp.vec3), vel_out: wp.array(dtype=wp.vec2),
                fear_out: wp.array(dtype=float), mem_pos_out: wp.array(dtype=wp.vec2),
                mem_str_out: wp.array(dtype=float), hunger_out: wp.array(dtype=float),
                forces: wp.array2d(dtype=wp.vec2), caught: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p3 = pos[i]
    p = wp.vec2(p3[0], p3[1])
    v = vel[i]
    spd = wp.length(v)
    h = v / wp.max(spd, 1.0e-5)

    # ---- 1. perceive neighbours (only those in the vision cone or lateral-line range)
    sep = wp.vec2(0.0, 0.0)
    ali = wp.vec2(0.0, 0.0)
    coh = wp.vec2(0.0, 0.0)
    n_rep = int(0)
    n_ori = int(0)
    n_att = int(0)
    alarm = float(0.0)
    query = wp.hash_grid_query(grid, p3, P.r_att)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            q3 = pos[j]
            d = wp.vec2(q3[0] - p[0], q3[1] - p[1])
            dist = wp.length(d)
            if dist < P.r_att and dist > 1.0e-6:
                u = d / dist
                if dist < P.r_lat or wp.dot(h, u) > P.cos_fov:
                    if dist < P.r_rep:
                        sep = sep - u * (1.0 - dist / P.r_rep)
                        n_rep += 1
                    elif dist < P.r_ori:
                        ali = ali + wp.normalize(vel[j])
                        n_ori += 1
                    else:
                        coh = coh + u
                        n_att += 1
                    alarm = wp.max(alarm, fear[j])

    # ---- 2. the three schooling rules
    f_sep = wp.vec2(0.0, 0.0)
    f_ali = wp.vec2(0.0, 0.0)
    f_coh = wp.vec2(0.0, 0.0)
    if n_rep > 0:
        sl = wp.length(sep)
        f_sep = sep / wp.max(sl, 1.0) * P.w_sep
    if n_ori > 0:
        f_ali = (wp.normalize(ali) - h) * P.w_ali
    if n_att > 0:
        f_coh = wp.normalize(coh) * P.w_coh

    # ---- 3. fear + working memory of the predator
    fr = wp.max(forget(fear[i], P.dt, P.fear_tau), P.contagion * alarm)
    ms = forget(mem_str[i], P.dt, P.mem_tau)
    mp = mem_pos[i]
    got_caught = int(0)
    if P.pred_on == 1:
        dp = p - P.pred_pos
        dist = wp.length(dp)
        u = dp / wp.max(dist, 1.0e-5)
        # sees it (in the cone) or feels its pressure wave (lateral line, 3x range: big body)
        if dist < P.r_pred and (dist < P.r_lat * 3.0 or wp.dot(h, -u) > P.cos_fov):
            threat = 0.5 + 0.5 * wp.min(wp.length(P.pred_vel) / P.v_attack, 1.0)
            fr = wp.max(fr, threat)
            ms = 1.0
            mp = P.pred_pos
        if dist < P.r_catch and wp.length(P.pred_vel) > P.v_attack:
            got_caught = 1

    # ---- 4. hunger: grows by itself; a hungry fish flees less (the risk trade-off)
    hg = wp.min(hunger[i] + P.dt / P.hunger_tau, 1.0)

    f_flee = wp.vec2(0.0, 0.0)
    if ms > 0.02:
        dp = p - mp
        dist = wp.length(dp)
        away = dp / wp.max(dist, 1.0e-5)
        side = wp.vec2(0.0, 0.0)
        pvl = wp.length(P.pred_vel)
        if pvl > 1.0:
            pd = P.pred_vel / pvl
            perp = wp.vec2(-pd[1], pd[0])
            side = perp * wp.sign(wp.dot(perp, dp))   # escape sideways off its path
        urgency = ms * wp.clamp(1.0 - dist / (P.r_pred * 1.3), 0.0, 1.0)
        f_flee = wp.normalize(away + side * P.flee_side) * (urgency * P.w_flee
                                                            * (1.0 - P.risk * hg))

    # ---- 5. food: the nearest pellet it can see, pulled by hunger
    f_food = wp.vec2(0.0, 0.0)
    best = P.r_food
    bi = int(-1)
    for k in range(P.n_pellets):
        if pellet_amt[k] > 0:
            d = pellet_pos[k] - p
            dist = wp.length(d)
            if dist < best and dist > 1.0e-6:
                if dist < P.r_lat * 2.0 or wp.dot(h, d / dist) > P.cos_fov:
                    best = dist
                    bi = k
    if bi >= 0:
        f_food = wp.normalize(pellet_pos[bi] - p) * (P.w_food * hg)
        if best < P.eat_r and hg > 0.05:
            wp.atomic_add(bites, bi, 1)                 # applied after the step
            wp.atomic_add(stats, 0, 1)
            if fr > 0.3:
                wp.atomic_add(stats, 1, 1)              # ate while afraid: a risky bite
            hg = wp.max(hg - P.bite, 0.0)

    # ---- 6. walls of the tank
    m = P.margin
    wx = float(0.0)
    wy = float(0.0)
    if p[0] < m:
        t = (m - p[0]) / m
        wx += t * t
    if p[0] > P.width - m:
        t = (p[0] - P.width + m) / m
        wx -= t * t
    if p[1] < m:
        t = (m - p[1]) / m
        wy += t * t
    if p[1] > P.height - m:
        t = (p[1] - P.height + m) / m
        wy -= t * t
    f_wall = wp.vec2(wx, wy) * P.w_wall

    # ---- 7. rock: pressure up close, and a look ahead to turn toward the freer side
    f_rock = obstacle_push(blocked, g, p, P.r_rock) * P.w_rock
    ahead = obstacle_clearance(blocked, g, p, h, P.look)
    if ahead < P.look:
        c_left = obstacle_clearance(blocked, g, p, rotate(h, LOOK_SPREAD), P.look)
        c_right = obstacle_clearance(blocked, g, p, rotate(h, -LOOK_SPREAD), P.look)
        turn = wp.vec2(-h[1], h[0])
        if c_right > c_left:
            turn = -turn
        f_rock = f_rock + turn * ((1.0 - ahead / P.look) * P.w_rock)

    # ---- 8. integrate
    acc = (f_sep + f_ali + f_coh + f_flee + f_wall + f_rock + f_food) * P.max_acc
    al = wp.length(acc)
    amax = P.max_acc * 1.5
    if al > amax:
        acc = acc * (amax / al)
    target = P.cruise + (P.burst - P.cruise) * fr
    acc = acc + h * ((target - spd) * P.k_speed)
    v2 = v + acc * P.dt
    s2 = wp.length(v2)
    vmin = P.cruise * 0.4
    if s2 < vmin:
        v2 = h * vmin
    elif s2 > P.burst * 1.2:
        v2 = v2 * (P.burst * 1.2 / s2)
    x = p[0] + v2[0] * P.dt
    y = p[1] + v2[1] * P.dt
    vx = v2[0]
    vy = v2[1]
    if x < 0.0:
        x = -x
        vx = wp.abs(vx)
    if x > P.width:
        x = 2.0 * P.width - x
        vx = -wp.abs(vx)
    if y < 0.0:
        y = -y
        vy = wp.abs(vy)
    if y > P.height:
        y = 2.0 * P.height - y
        vy = -wp.abs(vy)
    # rock is solid: slide along it (a fish buried by fresh paint may swim out)
    if obstacle_at(blocked, g, wp.vec2(x, y)) and not obstacle_at(blocked, g, p):
        if not obstacle_at(blocked, g, wp.vec2(x, p[1])):
            y = p[1]
            vy = -vy * 0.3
        elif not obstacle_at(blocked, g, wp.vec2(p[0], y)):
            x = p[0]
            vx = -vx * 0.3
        else:
            x = p[0]
            y = p[1]
            vx = -vx
            vy = -vy
    if got_caught == 1:          # respawn on the far side of the tank, in open water
        wp.atomic_add(caught, 0, 1)
        x = P.pred_pos[0] + P.width * 0.5
        if x > P.width:
            x = x - P.width
        y = P.pred_pos[1] + P.height * 0.5
        if y > P.height:
            y = y - P.height
        x = wp.clamp(x + (float(i % 13) - 6.0) * 0.6, 1.0, P.width - 1.0)
        y = wp.clamp(y + (float(i % 7) - 3.0) * 0.6, 1.0, P.height - 1.0)
        for _k in range(12):
            if obstacle_at(blocked, g, wp.vec2(x, y)):
                x = x + 11.0
                if x > P.width - 1.0:
                    x = x - P.width + 2.0
                    y = wp.clamp(y + 9.0, 1.0, P.height - 1.0)
        fr = 0.0
        ms = 0.0

    pos_out[i] = wp.vec3(x, y, 0.0)
    vel_out[i] = wp.vec2(vx, vy)
    fear_out[i] = fr
    mem_pos_out[i] = mp
    mem_str_out[i] = ms
    hunger_out[i] = hg
    forces[i, 0] = f_sep
    forces[i, 1] = f_ali
    forces[i, 2] = f_coh
    forces[i, 3] = f_flee
    forces[i, 4] = f_wall
    forces[i, 5] = f_rock
    forces[i, 6] = f_food


@wp.kernel
def pellets_eaten(amt: wp.array(dtype=wp.int32), bites: wp.array(dtype=wp.int32)):
    k = wp.tid()
    amt[k] = wp.max(amt[k] - bites[k], 0)
    bites[k] = 0


FORCE_STYLE = [("separation", pal.REPULSE), ("alignment", pal.ALIGN),
               ("cohesion", pal.ATTRACT), ("flee", pal.ROSE), ("tank", pal.SKY),
               ("rock", "#cbd5e1"), ("food", pal.GOLD)]
FOOD_COLOR = "#fcd34d"


def build_rocks(name: str, rocks: ObstacleGrid, rng: np.random.Generator) -> None:
    mask = np.zeros((rocks.ny, rocks.nx), bool)
    if name == "Reef":
        for _ in range(7):
            c = rng.uniform((22, 14), (138, 76))
            for _ in range(4):                        # a lumpy blob of capsules
                a = c + rng.normal(0, 3.0, 2)
                b = a + rng.normal(0, 4.0, 2)
                mask |= rocks.capsule(a, b, rng.uniform(1.8, 3.6))
    elif name == "Channel":
        mask |= rocks.capsule((80, -5), (80, 37), 3.2)
        mask |= rocks.capsule((80, 53), (80, 95), 3.2)
    rocks.set(mask)


class Sim(Simulation):
    world = (0.0, 0.0, WIDTH, HEIGHT)
    background = "water"
    PARAMS = [
        Param("scenario", "Tank", "Open tank", choices=SCENARIOS, restart=True,
              help="Start empty, with a reef, or with a wall that has one gap."),
        Param("n_fish", "Fish", 1500, 50, 20000, 50, "How many fish in the tank.", "N",
              restart=True, cpu_default=400),
        Param("r_rep", "Repulsion radius", 1.8, 0.2, 5.0, 0.1,
              "Closer than this, a fish moves away to avoid collision.", "r_r"),
        Param("r_ori", "Orientation radius", 4.0, 0.5, 12.0, 0.1,
              "Within this band, fish copy their neighbours' heading.", "r_o"),
        Param("r_att", "Attraction radius", 9.0, 2.0, 20.0, 0.5,
              "Up to this far, fish are drawn toward the group.", "r_a"),
        Param("fov", "Field of view", 300.0, 60.0, 360.0, 5.0,
              "Width of the vision cone; the rest is a blind spot behind.", "\\phi", "deg"),
        Param("r_lat", "Lateral line range", 2.0, 0.0, 8.0, 0.1,
              "Pressure sense along the body: works all around, but only up close.", "r_l"),
        Param("w_sep", "Separation weight", 3.0, 0.0, 6.0, 0.1,
              "How hard fish avoid bumping into each other. At 0 they pile up.", "w_s"),
        Param("w_ali", "Alignment weight", 1.4, 0.0, 6.0, 0.1,
              "How strongly fish copy their neighbours' heading. At 0 the school becomes a "
              "buzzing cloud.", "w_a"),
        Param("w_coh", "Cohesion weight", 0.5, 0.0, 6.0, 0.1,
              "How strongly fish are drawn toward the group. High: tight balls.", "w_c"),
        Param("w_flee", "Flee weight", 4.0, 0.0, 10.0, 0.1,
              "How hard a fish steers away from the predator it remembers.", "w_f"),
        Param("w_rock", "Rock avoidance", 3.0, 0.0, 8.0, 0.1,
              "How hard fish steer away from rock they feel or see ahead.", "w_r"),
        Param("w_food", "Food drive", 2.5, 0.0, 8.0, 0.1,
              "Pull toward visible food, scaled by hunger.", "w_h"),
        Param("hunger_tau", "Hunger grows over", 20.0, 2.0, 120.0, 1.0,
              "Seconds for an unfed fish to go from full to starving.", "\\tau_h", "s"),
        Param("risk", "Hunger overrides fear", 0.6, 0.0, 1.0, 0.05,
              "How much hunger weakens the urge to flee. 0 = safety always first.", "\\rho"),
        Param("cruise", "Cruise speed", 8.0, 1.0, 20.0, 0.5,
              "The relaxed swimming speed of a calm fish.", "v_0", "u/s"),
        Param("burst", "Burst speed", 24.0, 5.0, 45.0, 0.5, "Speed when terrified.", "v_b"),
        Param("r_pred", "Predator detection range", 18.0, 4.0, 40.0, 0.5,
              "How far away a fish can notice the predator (if it is in view).", "r_p"),
        Param("flee_side", "Sideways escape", 0.8, 0.0, 3.0, 0.05,
              "How much fish dodge sideways off the predator's path (fountain effect).", "k"),
        Param("contagion", "Alarm contagion", 0.85, 0.0, 1.0, 0.01,
              "Fraction of a neighbour's fear a fish copies. Near 1: panic spreads far.", "c"),
        Param("fear_tau", "Fear fades over", 1.5, 0.1, 10.0, 0.1,
              "How quickly a scared fish calms down. Long: panic lingers.", "\\tau_f", "s"),
        Param("mem_tau", "Memory fades over", 2.0, 0.1, 15.0, 0.1,
              "How long a fish keeps fleeing from where it last saw the predator.", "\\tau_m",
              "s"),
    ]
    OVERLAYS = [
        Overlay("zones", "Perception zones", True, "The focus fish's three zones and vision cone."),
        Overlay("links", "Neighbour links", True, "Who the focus fish is reacting to."),
        Overlay("forces", "Steering forces", True, "Each rule as an arrow; they add up."),
        Overlay("look", "Rock sense", True, "The focus fish's three looks ahead for rock."),
        Overlay("alarm", "Alarm level", False, "Colour every fish by fear."),
        Overlay("hunger", "Hunger", False, "Colour every fish by hunger (gold = starving)."),
        Overlay("memory", "Predator memory", False, "Where scared fish remember the predator."),
        Overlay("grid", "Spatial hash grid", False, "How the GPU finds neighbours fast."),
        Overlay("trail", "Focus trail", True, "The recent path of the focus fish."),
    ]
    TOOLS = [
        Tool("hunt", "Hunt", "hunt", "Move to chase. Fast swipes can catch fish.",
             tip="Your motion is the predator's direction vector. The school reads it and "
                 "dodges sideways off your path."),
        Tool("rock", "Rock", "wall", "Drag to build rock", "Drag to erase rock", radius=2.2,
             tip="Build reefs, channels and hiding places. The predator rests while you "
                 "build."),
        Tool("food", "Food", "food", "Click to drop food", "Click to clear food nearby",
             radius=2.5, tip="Hungry fish (gold in the Hunger overlay) leave the school for "
                             "food, even when it is dangerous."),
        Tool("inspect", "Inspect", "inspect", "Click a fish to follow it",
             tip="Overlays and Live Math explain this one fish."),
    ]
    EXPERIMENTS = [
        Experiment("scatter", "Scatter the school",
                   "With Hunt, dive fast straight through a dense school.",
                   "Most fish never saw you. Fear hopped from fish to fish faster than you "
                   "can swim: alarm contagion (c).", check="exp_scatter"),
        Experiment("catch", "Catch five fish",
                   "Slow approaches fail. Try fast, straight attacks.",
                   "Fish dodge sideways off your path (the fountain effect), so only fast, "
                   "straight strikes connect, just like real predators' strikes.",
                   check="exp_catch"),
        Experiment("feed", "Feed the school", "Pick Food (3) and drop some in open water.",
                   "Hungry fish break formation for food, then the school pulls them back: "
                   "the rules compete, and the weights decide who wins.", check="exp_feed"),
        Experiment("risk", "A risky dinner",
                   "Drop food, let the fish get hungry (Hunger overlay), then hunt near it.",
                   "Hungry fish flee less (ρ·h). Ecologists call this the foraging–predation "
                   "trade-off: real fish accept more risk the hungrier they are.",
                   check="exp_risk"),
        Experiment("reef", "Build a hiding reef",
                   "Pick Rock (2), paint a reef in open water, then hunt around it.",
                   "No fish knows the reef's shape. Each only feels rock nearby and looks "
                   "ahead, yet the school flows around it and re-forms behind it.",
                   check="exp_reef"),
        Experiment("blind", "Remove the blind spot",
                   "Set Field of view (φ) to 360° and try sneaking up from behind.",
                   "With no blind spot, no approach is hidden and the school reacts sooner. "
                   "Real fish trade rear vision for sharper forward vision."),
    ]
    LIVE_MATH = [
        LiveEq("sense", "Who does it notice?",
               r"d_{ij} < r_a \;\wedge\; \big(d_{ij} < r_l \,\vee\, "
               r"\hat{\mathbf{h}}_i\cdot\hat{\mathbf{u}}_{ij} > \cos\tfrac{\phi}{2}\big)",
               "d_ij: distance to fish j · r_a: how far it senses (attraction radius) · "
               "r_l: lateral-line range · ĥ_i: its heading · û_ij: direction to fish j · "
               "φ: field of view"),
        LiveEq("sep", "Too close? Then separate", r"d_{\min} < r_r",
               "d_min: distance to the nearest fish it senses · r_r: repulsion radius"),
        LiveEq("sum", "Steering is a sum of rules",
               r"\mathbf{F} = \mathbf{F}^{sep}+\mathbf{F}^{ali}+\mathbf{F}^{coh}"
               r"+\mathbf{F}^{flee}+\mathbf{F}^{rock}+\mathbf{F}^{food}",
               "F: the total push that steers it · each F: one rule's push (separation, "
               "alignment, cohesion, flee, rock, food)"),
        LiveEq("fear", "Fear sets the speed",
               r"v^* = v_0 + (v_b - v_0)\,f_i",
               "v*: the speed it aims for · v_0: cruise speed · v_b: burst speed · "
               "f_i: its fear, from 0 (calm) to 1 (terrified)"),
        LiveEq("hunger", "Hunger against fear",
               r"\mathbf{F}^{food} = w_h\,h_i\,\hat{\mathbf{u}}_{food},\quad "
               r"\mathbf{F}^{flee} \propto 1 - \rho\,h_i",
               "h_i: hunger, from 0 (full) to 1 (starving) · w_h: food drive · "
               "û_food: direction to the nearest food it sees · ρ: how much hunger "
               "overrides fear"),
    ]

    # ---------------------------------------------------------------- setup
    def reset(self, seed: int) -> None:
        super().reset(seed)
        n = int(self.p.n_fish)
        rng = np.random.default_rng(seed)
        self.rocks = ObstacleGrid(self.world, ROCK_CELL, self.device)
        build_rocks(self.p.scenario, self.rocks, rng)
        centers = rng.uniform((30, 20), (130, 70), size=(4, 2))
        which = rng.integers(0, 4, n)
        xy = centers[which] + rng.normal(0, 7.0, (n, 2))
        xy = np.clip(xy, (2, 2), (WIDTH - 2, HEIGHT - 2))
        for _ in range(20):                       # nobody starts inside rock
            bad = np.nonzero(~self._free(xy))[0]
            if not len(bad):
                break
            xy[bad] = rng.uniform((2, 2), (WIDTH - 2, HEIGHT - 2), (len(bad), 2))
        heading = rng.uniform(0, 2 * np.pi, 4)[which] + rng.normal(0, 0.4, n)
        vel = np.stack([np.cos(heading), np.sin(heading)], 1) * self.p.cruise
        self.hue = rng.uniform(0, 1, n).astype(np.float32)
        self.swim = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        hunger0 = rng.uniform(0.0, 0.5, n).astype(np.float32)
        self.load_state(xy, vel, hunger0)
        self.pellet_pos = np.zeros((MAX_PELLETS, 2), np.float32)
        self.pellet_amt = np.zeros(MAX_PELLETS, np.int32)
        self._upload_pellets()
        self.bites = wp.zeros(MAX_PELLETS, dtype=wp.int32, device=self.device)
        self.stats = wp.zeros(2, dtype=wp.int32, device=self.device)   # bites, risky bites
        self.caught = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.grid = wp.HashGrid(128, 128, 1, device=self.device)
        self._last_draw_t = 0.0
        self.pred_pos = np.array([-100.0, -100.0])
        self.pred_vel = np.zeros(2)
        self.pred_heading = 0.0
        self.pred_on = False
        self._last_mouse = None
        self.focus = min(self.focus, n - 1)
        self.trail = TraceMemory(180, 2)
        self._rock_img = None
        self._rock_version = -1
        self._max_alarmed = 0.0

    def load_state(self, xy, vel, hunger=None) -> None:
        """Replace every fish (tests build tiny hand-made situations with this)."""
        n = len(xy)
        pos = np.zeros((n, 3), np.float32)
        pos[:, :2] = xy
        dev = self.device
        self.pos = wp.array(pos, dtype=wp.vec3, device=dev)
        self.vel = wp.array(np.asarray(vel, np.float32), dtype=wp.vec2, device=dev)
        self.fear = wp.zeros(n, dtype=float, device=dev)
        self.mem_pos = wp.zeros(n, dtype=wp.vec2, device=dev)
        self.mem_str = wp.zeros(n, dtype=float, device=dev)
        h = np.zeros(n, np.float32) if hunger is None else np.asarray(hunger, np.float32)
        self.hunger = wp.array(h, dtype=float, device=dev)
        self._back = [wp.empty_like(a) for a in
                      (self.pos, self.vel, self.fear, self.mem_pos, self.mem_str, self.hunger)]
        self.forces = wp.zeros((n, N_FORCES), dtype=wp.vec2, device=dev)
        if len(self.hue) != n:
            self.hue = np.zeros(n, np.float32)
            self.swim = np.zeros(n, np.float32)
        self.focus = min(self.focus, n - 1)
        self._host = None

    def _free(self, xy) -> np.ndarray:
        ix = np.clip((xy[:, 0] / ROCK_CELL).astype(int), 0, self.rocks.nx - 1)
        iy = np.clip((xy[:, 1] / ROCK_CELL).astype(int), 0, self.rocks.ny - 1)
        return self.rocks.mask[iy, ix] == 0

    def _upload_pellets(self) -> None:
        self.pellet_pos_d = wp.array(self.pellet_pos, dtype=wp.vec2, device=self.device)
        self.pellet_amt_d = wp.array(self.pellet_amt, dtype=wp.int32, device=self.device)

    def _params(self) -> School:
        p = self.p
        s = School()
        s.r_rep, s.r_ori, s.r_att = p.r_rep, max(p.r_ori, p.r_rep), max(p.r_att, p.r_ori)
        s.cos_fov = float(np.cos(np.radians(p.fov) / 2))
        s.r_lat = p.r_lat
        s.w_sep, s.w_ali, s.w_coh, s.w_flee, s.w_wall = p.w_sep, p.w_ali, p.w_coh, p.w_flee, 4.0
        s.w_rock, s.r_rock, s.look = p.w_rock, 3.0, 7.0
        s.w_food, s.r_food, s.eat_r = p.w_food, 16.0, 0.9
        s.hunger_tau, s.bite, s.risk = p.hunger_tau, 0.25, p.risk
        s.n_pellets = MAX_PELLETS
        s.cruise, s.burst = p.cruise, max(p.burst, p.cruise)
        s.max_acc, s.k_speed = 30.0, 2.0
        s.r_pred, s.flee_side = p.r_pred, p.flee_side
        s.fear_tau, s.contagion, s.mem_tau = p.fear_tau, p.contagion, p.mem_tau
        s.r_catch, s.v_attack = 1.4, 40.0
        s.width, s.height, s.margin = WIDTH, HEIGHT, 12.0
        s.dt = self.dt
        s.pred_pos = wp.vec2(*self.pred_pos)
        s.pred_vel = wp.vec2(*self.pred_vel)
        s.pred_on = 1 if self.pred_on else 0
        return s

    # ----------------------------------------------------------------- tools
    def _use_tools(self, inp: InputState) -> None:
        tool = self.tool_of(inp)
        hunting = tool == "hunt" and inp.mouse is not None
        # The cursor is the predator; its smoothed motion is its direction vector.
        if hunting:
            m = np.asarray(inp.mouse, float)
            if self._last_mouse is not None:
                raw = (m - self._last_mouse) / self.dt
                self.pred_vel += (raw - self.pred_vel) * (1 - np.exp(-self.dt / 0.08))
            self._last_mouse, self.pred_pos, self.pred_on = m, m, True
        else:
            self._last_mouse, self.pred_on = None, False
            self.pred_vel *= 0.9
        if np.linalg.norm(self.pred_vel) > 3.0:
            self.pred_heading = float(np.arctan2(self.pred_vel[1], self.pred_vel[0]))

        if tool == "rock" and inp.mouse is not None and {"left", "right"} & inp.buttons:
            self.rocks.stroke(inp.mouse, 2.2, 1 if "left" in inp.buttons else 0)
        else:
            self.rocks.end_stroke()
        for button, at in inp.clicks:
            if tool == "inspect" and button == "left":
                self._pick_focus(at)
            elif tool == "food":
                self._food_click(button, np.asarray(at, float))

    def _food_click(self, button: str, at: np.ndarray) -> None:
        self.pellet_amt = self.pellet_amt_d.numpy()
        if button == "left":
            free = np.nonzero(self.pellet_amt <= 0)[0][:7]
            golden = 2.39996323
            for k, slot in enumerate(free):          # a small spiral of flakes, no RNG
                r = 0.55 * np.sqrt(k + 0.5)
                q = at + r * np.array([np.cos(k * golden), np.sin(k * golden)])
                if self.rocks.free(q):
                    self.pellet_pos[slot] = q
                    self.pellet_amt[slot] = PELLET_BITES
        elif button == "right":
            near = np.linalg.norm(self.pellet_pos - at, axis=1) < 2.5
            self.pellet_amt[near] = 0
        self._upload_pellets()

    def _pick_focus(self, at) -> None:
        xy = self.pos.numpy()[:, :2]
        self.focus = int(np.argmin(np.sum((xy - np.asarray(at)) ** 2, axis=1)))
        self.trail.clear()

    # ----------------------------------------------------------------- step
    def step(self, inp: InputState) -> None:
        self._use_tools(inp)
        self.grid.build(self.pos, float(max(self.p.r_att, 0.5)))
        b = self._back
        wp.launch(school_step, dim=len(self.pos),
                  inputs=[self.grid.id, self._params(), self.rocks.grid, self.rocks.blocked,
                          self.pellet_pos_d, self.pellet_amt_d, self.bites, self.stats,
                          self.pos, self.vel, self.fear, self.mem_pos, self.mem_str,
                          self.hunger, b[0], b[1], b[2], b[3], b[4], b[5],
                          self.forces, self.caught], device=self.device)
        wp.launch(pellets_eaten, dim=MAX_PELLETS, inputs=[self.pellet_amt_d, self.bites],
                  device=self.device)
        self.pos, b[0] = b[0], self.pos
        self.vel, b[1] = b[1], self.vel
        self.fear, b[2] = b[2], self.fear
        self.mem_pos, b[3] = b[3], self.mem_pos
        self.mem_str, b[4] = b[4], self.mem_str
        self.hunger, b[5] = b[5], self.hunger
        self._host = None

    def _fetch(self):
        if self._host is None:
            self._host = dict(pos=self.pos.numpy()[:, :2], vel=self.vel.numpy(),
                              fear=self.fear.numpy(), mem=self.mem_str.numpy(),
                              mem_pos=self.mem_pos.numpy(), forces=self.forces.numpy(),
                              hunger=self.hunger.numpy(), pellets=self.pellet_amt_d.numpy(),
                              stats=self.stats.numpy(), caught=int(self.caught.numpy()[0]))
        return self._host

    def state_arrays(self):
        h = self._fetch()
        return [h["pos"], h["vel"], h["fear"], h["mem"], h["hunger"], h["pellets"],
                self.rocks.mask]

    def _focus_view(self, h):
        """Who the focus fish perceives, host-side (the same test as the kernel)."""
        P, f = self.p, self.focus
        pos, vel = h["pos"], h["vel"]
        d = pos - pos[f]
        dist = np.linalg.norm(d, axis=1)
        u = d / np.maximum(dist, 1e-6)[:, None]
        hvec = vel[f] / max(np.linalg.norm(vel[f]), 1e-6)
        near = (dist < P.r_att) & (dist > 1e-6)
        near[f] = False
        vis = near & ((dist < P.r_lat) | (u @ hvec > np.cos(np.radians(P.fov) / 2)))
        return near, vis, dist

    # ----------------------------------------------------------------- draw
    def draw(self, s) -> None:
        h = self._fetch()
        pos, vel, fear, mem = h["pos"], h["vel"], h["fear"], h["mem"]
        f = self.focus
        spd = np.linalg.norm(vel, axis=1)
        ang = np.arctan2(vel[:, 1], vel[:, 0])
        dt_vis = max(0.0, self.t - self._last_draw_t)
        self._last_draw_t = self.t
        self.swim += dt_vis * (7.0 + 0.9 * spd)
        self.trail.push(self.t, pos[f])
        P = self.p

        s.background("water")
        s.polyline([(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)], 0.25,
                   pal.rgba("#7dd3fc", 0.35, 1.2), closed=True, additive=True)

        if self.show.grid:
            c = max(P.r_att, 0.5)
            xs = np.arange(0, WIDTH + 1e-3, c)
            ys = np.arange(0, HEIGHT + 1e-3, c)
            s.lines(np.stack([xs, np.zeros_like(xs)], 1), np.stack([xs, np.full_like(xs, HEIGHT)], 1),
                    0.06, pal.rgba(pal.SENSE, 0.18))
            s.lines(np.stack([np.zeros_like(ys), ys], 1), np.stack([np.full_like(ys, WIDTH), ys], 1),
                    0.06, pal.rgba(pal.SENSE, 0.18))
            cx, cy = np.floor(pos[f] / c)
            s.rect((cx - 1) * c, (cy - 1) * c, (cx + 2) * c, (cy + 2) * c, pal.rgba(pal.SENSE, 0.07))

        # food flakes: they bob and glint, and shrink as they are eaten
        live = np.nonzero(h["pellets"] > 0)[0]
        if len(live):
            q = self.pellet_pos[live]
            bob = 0.08 * np.stack([np.sin(self.t * 1.7 + live), np.cos(self.t * 1.3 + live)], 1)
            amt = h["pellets"][live] / PELLET_BITES
            s.glow(q + bob, 1.2, pal.rgba(FOOD_COLOR, 0.18))
            s.circles(q + bob, 0.22 + 0.18 * amt, pal.rgba(FOOD_COLOR, 1.0, 1.3))

        if self.show.memory:
            idx = np.nonzero(mem > 0.25)[0][:800]
            if len(idx):
                col = pal.rgba(pal.MEMORY, 1.0)[None].repeat(len(idx), 0)
                col[:, 3] = 0.10 * mem[idx]
                s.lines(pos[idx], h["mem_pos"][idx], 0.05, col, additive=True)

        if self.show.zones:
            hd = ang[f]
            half = np.radians(P.fov) / 2
            s.wedge(pos[f], P.r_att, hd, half, pal.rgba(pal.SENSE, 0.06))
            s.circles([pos[f]] * 3, [P.r_rep, P.r_ori, P.r_att], [
                pal.rgba(pal.REPULSE, 0.8), pal.rgba(pal.ALIGN, 0.6), pal.rgba(pal.ATTRACT, 0.5)],
                ring=0.08)
            if P.r_lat > 0:
                s.circles(pos[f], P.r_lat, pal.rgba(pal.MEMORY, 0.5), ring=0.05)

        if self.show.links:
            _, vis, dist = self._focus_view(h)
            idx = np.nonzero(vis)[0]
            zone = np.where(dist[idx] < P.r_rep, 0, np.where(dist[idx] < P.r_ori, 1, 2))
            cols = np.stack([pal.rgba(pal.REPULSE, 0.85), pal.rgba(pal.ALIGN, 0.45),
                             pal.rgba(pal.ATTRACT, 0.22)])[zone]
            s.lines(np.broadcast_to(pos[f], (len(idx), 2)), pos[idx], 0.06, cols)
            self._nbr = np.bincount(zone, minlength=3)

        if self.show.trail and len(self.trail) > 2:
            _, tr = self.trail.ordered()
            a = np.linspace(0, 0.6, len(tr) - 1)
            col = np.tile(pal.rgba("#e0f2fe"), (len(tr) - 1, 1))
            col[:, 3] = a
            s.polyline(tr, 0.12, col, additive=True)

        # the fish themselves
        base = pal.lerp_colors("#4f9cc0", "#8fbdb4", self.hue)
        if self.show.alarm:
            col = pal.lerp_colors("#5eead4", "#fb7185", fear)
            col[:, :3] *= (1 + 1.2 * fear)[:, None]
            glow = np.nonzero(fear > 0.35)[0]
            if len(glow):
                s.glow(pos[glow], 1.6, pal.rgba(pal.ROSE, 0.10))
        elif self.show.hunger:
            col = pal.lerp_colors("#60a5fa", "#fbbf24", h["hunger"])
            col[:, :3] *= (1 + 0.5 * h["hunger"])[:, None]
        else:
            # startled fish flash silver: the real "flash expansion" glint
            col = base.copy()
            col[:, :3] *= (1 + 0.35 * fear ** 2)[:, None]
        size = np.stack([1.6 * (0.92 + 0.16 * self.hue), 0.56 * (0.92 + 0.16 * self.hue)], 1)
        s.sprites("fish", pos, ang, size, col, self.swim)

        # rock sits above the water column, over the fish that swim beneath its edge
        if self._rock_version != self.rocks.version:
            self._rock_img = self.rocks.rock_image((0.17, 0.21, 0.24), self.seed)
            self._rock_version = self.rocks.version
        if self.rocks.mask.any():
            s.image(self._rock_img, self.world, mode="mask")

        # focus fish: its looks ahead, then its steering forces
        s.circles(pos[f], 1.4, pal.rgba("#ffffff", 0.75, 1.1), ring=0.09)
        if self.show.look:
            hv = vel[f] / max(np.linalg.norm(vel[f]), 1e-6)
            for k, a in enumerate((LOOK_SPREAD, 0.0, -LOOK_SPREAD)):
                u = np.array([hv[0] * np.cos(a) - hv[1] * np.sin(a),
                              hv[0] * np.sin(a) + hv[1] * np.cos(a)])
                c = self._clearance(pos[f], u, 7.0)
                hit = c < 7.0
                col = pal.rgba(pal.REPULSE if hit else pal.SENSE, 0.75 if k == 1 else 0.45)
                s.lines(pos[f], pos[f] + u * c, 0.07, col, additive=True)
                if hit:
                    s.circles(pos[f] + u * c, 0.35, pal.rgba(pal.REPULSE, 0.9, 1.4))
        if self.show.forces:
            F = h["forces"][f]
            scale = 2.5
            for k, (_, c) in enumerate(FORCE_STYLE):
                s.arrows(pos[f], F[k] * scale, 0.14, pal.rgba(c, 0.95, 1.3))
            s.arrows(pos[f], F.sum(0) * scale, 0.2, pal.rgba("#ffffff", 1.0, 1.5))
        if self.show.memory and mem[f] > 0.02:
            s.circles(h["mem_pos"][f], 2.0, pal.rgba(pal.MEMORY, 0.8 * mem[f], 1.4), ring=0.15)

        # the predator (your cursor, while you hold the Hunt tool)
        if self.pred_on:
            pp = self.pred_pos
            threat = min(np.linalg.norm(self.pred_vel) / 40.0, 1.0)
            s.circles(pp, P.r_pred, pal.rgba(pal.CORAL, 0.10 + 0.25 * threat), ring=0.15)
            s.glow(pp, 6.0, pal.rgba(pal.CORAL, 0.10 + 0.25 * threat))
            s.sprites("predator", pp, self.pred_heading, (7.0, 2.3), pal.rgba("#475569"),
                      self.t * 5.0)
            if np.linalg.norm(self.pred_vel) > 1.0:
                s.arrows(pp, self.pred_vel * 0.12, 0.18, pal.rgba(pal.CORAL, 0.9, 1.4))

    def _clearance(self, p, u, max_dist) -> float:
        """Host mirror of obstacle_clearance, for the Rock sense overlay."""
        step = 0.5 * ROCK_CELL
        for k in range(1, int(max_dist / step) + 1):
            if not self.rocks.free(p + u * k * step):
                return k * step
        return max_dist

    # ------------------------------------------------------------------ hud
    def hud(self) -> list[HudItem]:
        h = self._fetch()
        fear = h["fear"]
        alarmed = float(np.mean(fear > 0.3))
        self._max_alarmed = max(self._max_alarmed, alarmed)
        items = [
            HudItem("fish", f"{len(fear):,}"),
            HudItem("alarmed", f"{100 * alarmed:.0f}%", accent=True),
            HudItem("caught", f"{h['caught']}"),
            HudItem("hungry", f"{100 * np.mean(h['hunger'] > 0.5):.0f}%"),
            HudItem("predator speed", f"{np.linalg.norm(self.pred_vel):.0f} u/s"),
        ]
        if (h["pellets"] > 0).any():
            items.append(HudItem("food bites left", f"{int(h['pellets'].sum())}"))
        nb = getattr(self, "_nbr", None)
        if nb is not None and self.show.links:
            items.append(HudItem("focus sees", f"{nb[0]} close · {nb[1]} align · {nb[2]} far"))
        return items

    # ------------------------------------------------------------- live math
    def live_math(self) -> dict[str, LiveValue]:
        h = self._fetch()
        P, f = self.p, self.focus
        near, vis, dist = self._focus_view(h)
        n_near, n_vis = int(near.sum()), int(vis.sum())
        out = {"sense": LiveValue(
            f"sees {n_vis} of {n_near} fish within r_a = {fmt(P.r_att, 1)}",
            None, f"{n_near - n_vis} hidden in the blind spot" if n_near > n_vis
            else "nobody hidden right now")}
        dmin = float(dist[vis].min()) if n_vis else float("inf")
        close = dmin < P.r_rep
        out["sep"] = LiveValue(f"d_min = {fmt(dmin)} {'<' if close else '≥'} r_r = "
                               f"{fmt(P.r_rep)}", close,
                               "too close: separation pushes it away" if close
                               else "enough room: no separation force")
        F = h["forces"][f]
        mags = np.linalg.norm(F, axis=1)
        names = ["sep", "ali", "coh", "flee", "tank", "rock", "food"]
        shown = [f"|{n}| {fmt(m)}" for n, m in zip(names, mags, strict=True) if m > 1e-3]
        top = names[int(np.argmax(mags))] if mags.max() > 1e-3 else ""
        full = {"sep": "separation", "ali": "alignment", "coh": "cohesion", "flee": "flee",
                "tank": "the tank wall", "rock": "rock avoidance", "food": "food"}
        out["sum"] = LiveValue("  ".join(shown) or "all rules quiet: just cruising",
                               None, f"strongest: {full[top]}" if top else "")
        fr = float(h["fear"][f])
        vt = P.cruise + (max(P.burst, P.cruise) - P.cruise) * fr
        out["fear"] = LiveValue(f"v* = {fmt(P.cruise, 1)} + {fmt(P.burst - P.cruise, 1)}·"
                                f"{fmt(fr)} = {fmt(vt, 1)} u/s", fr > 0.3,
                                "alarmed: bursting away" if fr > 0.3 else "calm: cruising")
        hg = float(h["hunger"][f])
        out["hunger"] = LiveValue(f"h = {fmt(hg)}:  food pull {fmt(P.w_food * hg)},  "
                                  f"flee ×{fmt(1 - P.risk * hg)}", hg > 0.5,
                                  "hungry: will take risks for food" if hg > 0.5
                                  else "fed: safety first")
        return out

    # ------------------------------------------------------------ experiments
    def exp_scatter(self) -> bool:
        return self._max_alarmed >= 0.5

    def exp_catch(self) -> bool:
        return self._fetch()["caught"] >= 5

    def exp_feed(self) -> bool:
        return int(self._fetch()["stats"][0]) >= 30

    def exp_risk(self) -> bool:
        return int(self._fetch()["stats"][1]) >= 5

    def exp_reef(self) -> bool:
        return self.rocks.painted >= 80
