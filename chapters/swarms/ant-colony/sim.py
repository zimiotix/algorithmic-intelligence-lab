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

from ailab.core import HudItem, InputState, Overlay, Param, Simulation
from ailab.core.memory import (
    FieldMemory,
    Grid2D,
    field_deposit,
    field_sample3,
    grid_cell,
    grid_inside,
)
from ailab.render import palette as pal

WIDTH, HEIGHT, CELL = 160.0, 90.0, 0.5
HOME, FOOD = 0, 1
NO_CLAIM = 2**31 - 1
SCENARIOS = ("Open field", "Double bridge", "Maze")


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
    homing: float
    dt: float
    seed: int


@wp.func
def is_blocked(blocked: wp.array2d(dtype=wp.uint8), g: Grid2D, p: wp.vec2) -> bool:
    c = grid_cell(g, p)
    if not grid_inside(g, c):
        return True
    return blocked[c[1], c[0]] != wp.uint8(0)


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
                        s += 3.0                               # food itself
    else:
        s = wp.log(1.0 + field_sample3(value, 0, g, q))     # follow the home trail
        dn = wp.length(q - P.nest)
        if dn < P.scent_r:
            s += 6.0 * (1.0 - dn / P.scent_r)                  # the nest smells too
    return s


@wp.kernel
def ant_step(P: Colony, g: Grid2D, pos: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
             carry: wp.array(dtype=int), timer: wp.array(dtype=float),
             value: wp.array3d(dtype=float), deposit: wp.array3d(dtype=wp.int32),
             food: wp.array2d(dtype=int), blocked: wp.array2d(dtype=wp.uint8),
             claims: wp.array2d(dtype=int), delivered: wp.array(dtype=int)):
    i = wp.tid()
    p = pos[i]
    a = ang[i]
    c = carry[i]
    tm = timer[i]
    rng = wp.rand_init(P.seed, i)

    # 1. sense: three sensors ahead-left, ahead, ahead-right
    sa = P.sensor_angle
    sd = P.sensor_dist
    s_l = smell(P, g, value, food, p + wp.vec2(wp.cos(a + sa), wp.sin(a + sa)) * sd, c)
    s_f = smell(P, g, value, food, p + wp.vec2(wp.cos(a), wp.sin(a)) * sd, c)
    s_r = smell(P, g, value, food, p + wp.vec2(wp.cos(a - sa), wp.sin(a - sa)) * sd, c)

    # 2. decide: turn toward the strongest smell, plus a little random wandering
    turn = float(0.0)
    if s_f < s_l or s_f < s_r:
        if s_l > s_r:
            turn = 1.0
        elif s_r > s_l:
            turn = -1.0
    # path integration: loaded ants also feel the direction home (breaks "ant mills")
    if c == 1:
        to_nest = P.nest - p
        turn = turn + P.homing * wp.sin(wp.atan2(to_nest[1], to_nest[0]) - a)
    a = a + turn * P.turn_rate * P.dt + (wp.randf(rng) - 0.5) * 2.0 * P.wander * wp.sqrt(P.dt)

    # 3. act: move, or turn around at walls
    step = wp.vec2(wp.cos(a), wp.sin(a)) * (P.speed * P.dt)
    if is_blocked(blocked, g, p + step):
        a = a + 3.14159265 + (wp.randf(rng) - 0.5) * 1.2
    else:
        p = p + step

    # 4. mark: strength fades with time since the ant left home / found food
    ch = int(0)
    if c == 1:
        ch = 1
    field_deposit(deposit, ch, g, p, P.deposit * wp.exp(-tm / P.trail_tau) * P.dt)
    tm = tm + P.dt

    if wp.length(p - P.nest) < P.nest_r:
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


def _rock_texture(nx, ny, seed):
    """Base rock colour; the mask shader adds the detailed noise and lighting."""
    rng = np.random.default_rng(seed + 99)
    fine = rng.random((ny, nx))
    t = 0.92 + 0.08 * fine
    rgb = np.stack([0.19 * t, 0.16 * t, 0.14 * t], -1)
    return rgb.astype(np.float32)


class Sim(Simulation):
    world = (0.0, 0.0, WIDTH, HEIGHT)
    background = "soil"
    PARAMS = [
        Param("scenario", "Scenario", "Open field", choices=SCENARIOS, restart=True,
              help="Double bridge is the classic experiment: two routes, one shorter."),
        Param("n_ants", "Ants", 1500, 50, 20000, 50, "", "N", restart=True, cpu_default=600),
        Param("speed", "Walking speed", 10.0, 2.0, 25.0, 0.5, "", "v", "u/s"),
        Param("sensor_angle", "Sensor angle", 35.0, 5.0, 90.0, 1.0,
              "Angle between the front sensor and the side sensors.", "\\theta_s", "deg"),
        Param("sensor_dist", "Sensor distance", 2.5, 0.5, 8.0, 0.1,
              "How far ahead the sensors reach.", "d_s"),
        Param("turn_rate", "Turn rate", 8.0, 0.5, 20.0, 0.5, "", "\\omega", "rad/s"),
        Param("wander", "Wander", 1.5, 0.0, 6.0, 0.1,
              "Random turning. Too little: no exploring. Too much: no trail following.",
              "\\sigma"),
        Param("homing", "Path integration", 0.5, 0.0, 2.0, 0.05,
              "Loaded ants also steer toward home by dead reckoning. 0 = pheromone only "
              "(try it: ants can get stuck circling in an 'ant mill').", "h"),
        Param("deposit", "Pheromone per second", 1.0, 0.0, 5.0, 0.05, "", "q"),
        Param("trail_tau", "Ant memory", 20.0, 1.0, 120.0, 1.0,
              "Marks weaken with time since the ant left home or found food.", "\\tau_a", "s"),
        Param("evap_tau", "Evaporation time", 40.0, 2.0, 300.0, 1.0,
              "How long pheromone lasts in the world.", "\\tau_e", "s"),
        Param("diffusion", "Diffusion", 1.0, 0.0, 12.0, 0.1,
              "How fast pheromone spreads to neighbouring cells.", "D", "cells^2/s"),
    ]
    OVERLAYS = [
        Overlay("home", "Home trail", True, "Laid by searching ants: points back to the nest."),
        Overlay("food_trail", "Food trail", True, "Laid by ants carrying food."),
        Overlay("sensors", "Focus ant sensors", True, "What the focus ant smells."),
        Overlay("memory", "Ant memory", False, "Tint ants by time since they left home/food."),
        Overlay("ants", "Ants", True),
    ]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        rng = np.random.default_rng(seed)
        dev = self.device
        self.field = FieldMemory(self.world, CELL, channels=2, device=dev)
        nx, ny = self.field.nx, self.field.ny
        self.blocked_np, food_np, self.nest = build_world(self.p.scenario, rng, nx, ny)
        self.food_total = int(food_np.sum())
        self.food = wp.array(food_np, dtype=int, device=dev)
        self.field.set_blocked(self.blocked_np)
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
        self.leg_phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.rock_rgb = _rock_texture(nx, ny, seed)
        self.seed_jitter = rng.uniform(-0.3, 0.3, (ny, nx, 2)).astype(np.float32)
        self._rock_img = None
        self._paint_prev = None
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
        c.nest_r, c.scent_r, c.homing = 3.0, 10.0, p.homing
        c.dt = self.dt
        c.seed = int((self.seed * 1_000_003 + self.steps) % (2**31 - 1))
        return c

    # ------------------------------------------------------------ interaction
    def _paint(self, inp: InputState) -> None:
        m = inp.mouse
        painting = (m is not None and ({"left", "right"} & inp.buttons)
                    and not ({"Shift", "Ctrl"} & inp.keys))
        for button, at in inp.clicks:
            if button == "left" and "Ctrl" in inp.keys:
                xy = self.pos.numpy()
                self.focus = int(np.argmin(np.sum((xy - np.asarray(at)) ** 2, axis=1)))
            elif button == "left" and "Shift" in inp.keys:
                food = self.food.numpy()
                X, Y = _centers(self.field.nx, self.field.ny)
                disk = _capsule(X, Y, at, at, 3.0) & (self.blocked_np == 0)
                food[disk] += 8
                self.food_total += int(8 * disk.sum())
                self.food = wp.array(food, dtype=int, device=self.device)
        if not painting:
            self._paint_prev = None
            return
        a = self._paint_prev if self._paint_prev is not None else m
        X, Y = _centers(self.field.nx, self.field.ny)
        brush = _capsule(X, Y, a, m, 1.6)
        brush &= ~_capsule(X, Y, self.nest, self.nest, 5.0)     # never bury the nest
        if "left" in inp.buttons:
            self.blocked_np[brush] = 1
        else:
            self.blocked_np[brush] = 0
            self.blocked_np[0, :] = self.blocked_np[-1, :] = 1
            self.blocked_np[:, 0] = self.blocked_np[:, -1] = 1
        self.field.set_blocked(self.blocked_np)
        self._rock_img = None
        self._paint_prev = m

    def step(self, inp: InputState) -> None:
        self._paint(inp)
        dev, g, f = self.device, self.field.grid, self.field
        self.claims.fill_(NO_CLAIM)
        wp.launch(ant_step, dim=len(self.pos),
                  inputs=[self._colony(), g, self.pos, self.ang, self.carry, self.timer,
                          f.value, f.deposit, self.food, f.blocked, self.claims,
                          self.delivered], device=dev)
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
                              delivered=int(self.delivered.numpy()[0]))
        return self._host

    def state_arrays(self):
        h = self._fetch()
        return [h["pos"], h["ang"], h["carry"], h["field"], h["food"]]

    # ------------------------------------------------------------------ draw
    def draw(self, s) -> None:
        h = self._fetch()
        pos, ang, carry = h["pos"], h["ang"], h["carry"]
        field = h["field"]
        s.background("soil")
        b = self.world

        if self._rock_img is None:
            img = np.zeros(self.blocked_np.shape + (4,), np.float32)
            img[..., :3] = self.rock_rgb
            img[..., 3] = self.blocked_np
            self._rock_img = img
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
            s.sprites("ant", pos, ang, (1.5, 0.8), col, self.leg_phase)
            idx = np.nonzero(carry == 1)[0]
            if len(idx):
                tip = pos[idx] + np.stack([np.cos(ang[idx]), np.sin(ang[idx])], 1) * 0.75
                s.circles(tip, 0.28, pal.rgba("#86efac", 1.0, 1.6))

        f = self.focus
        if self.show.sensors and f < len(pos):
            self._draw_sensors(s, pos[f], ang[f], carry[f], field, food)
            s.circles(pos[f], 1.6, pal.rgba("#ffffff", 0.9, 1.5), ring=0.12)

    def _draw_sensors(self, s, p, a, c, field, food) -> None:
        P = self.p
        sa, sd = np.radians(P.sensor_angle), P.sensor_dist
        pts = [p + sd * np.array([np.cos(a + o), np.sin(a + o)]) for o in (sa, 0.0, -sa)]
        vals = []
        ch = FOOD if c == 0 else HOME
        for q in pts:
            ix, iy = int(q[0] / CELL), int(q[1] / CELL)
            y0, y1 = max(iy - 1, 0), min(iy + 2, field.shape[1])
            x0, x1 = max(ix - 1, 0), min(ix + 2, field.shape[2])
            v = np.log1p(field[ch, y0:y1, x0:x1].sum())      # same rule as the kernel
            if c == 0:
                v += 3.0 * (food[y0:y1, x0:x1] > 0).sum()
            else:
                dn = np.hypot(*(q - self.nest))
                v += 6.0 * max(0.0, 1 - dn / 10.0)
            vals.append(v)
        vals = np.array(vals)
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
            HudItem("food left", f"{left:,} / {self.food_total:,}"),
            HudItem("time", f"{self.t:.0f} s"),
        ]
