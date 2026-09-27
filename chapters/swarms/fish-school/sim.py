"""Fish school under predation.

Every fish runs the same local rules (Reynolds boids with Couzin's zones):
  repulsion zone   -> move away       (don't collide)
  orientation zone -> match heading   (swim together)
  attraction zone  -> move closer     (don't get left behind)
plus senses (vision cone with a blind spot, a 360 degree lateral line), fear that spreads
between neighbours, and a *working memory* of where the predator was last seen.

The GPU kernel reads the previous state and writes a new buffer (double buffering),
so no fish ever sees a half-updated neighbour, and the run is deterministic.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ailab.core import HudItem, InputState, Overlay, Param, Simulation
from ailab.core.memory import TraceMemory, forget
from ailab.render import palette as pal

WIDTH, HEIGHT = 160.0, 90.0
N_FORCES = 5  # separation, alignment, cohesion, flee, wall


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


@wp.kernel
def school_step(grid: wp.uint64, P: School,
                pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec2),
                fear: wp.array(dtype=float), mem_pos: wp.array(dtype=wp.vec2),
                mem_str: wp.array(dtype=float),
                pos_out: wp.array(dtype=wp.vec3), vel_out: wp.array(dtype=wp.vec2),
                fear_out: wp.array(dtype=float), mem_pos_out: wp.array(dtype=wp.vec2),
                mem_str_out: wp.array(dtype=float),
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
        f_flee = wp.normalize(away + side * P.flee_side) * (urgency * P.w_flee)

    # ---- 4. walls of the tank
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

    # ---- 5. integrate
    acc = (f_sep + f_ali + f_coh + f_flee + f_wall) * P.max_acc
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
    if got_caught == 1:          # respawn on the far side of the tank
        wp.atomic_add(caught, 0, 1)
        x = P.pred_pos[0] + P.width * 0.5
        if x > P.width:
            x = x - P.width
        y = P.pred_pos[1] + P.height * 0.5
        if y > P.height:
            y = y - P.height
        x = wp.clamp(x + (float(i % 13) - 6.0) * 0.6, 1.0, P.width - 1.0)
        y = wp.clamp(y + (float(i % 7) - 3.0) * 0.6, 1.0, P.height - 1.0)
        fr = 0.0
        ms = 0.0

    pos_out[i] = wp.vec3(x, y, 0.0)
    vel_out[i] = wp.vec2(vx, vy)
    fear_out[i] = fr
    mem_pos_out[i] = mp
    mem_str_out[i] = ms
    forces[i, 0] = f_sep
    forces[i, 1] = f_ali
    forces[i, 2] = f_coh
    forces[i, 3] = f_flee
    forces[i, 4] = f_wall


FORCE_STYLE = [("separation", pal.REPULSE), ("alignment", pal.ALIGN),
               ("cohesion", pal.ATTRACT), ("flee", pal.ROSE), ("wall", pal.SKY)]


class Sim(Simulation):
    world = (0.0, 0.0, WIDTH, HEIGHT)
    background = "water"
    PARAMS = [
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
        Param("w_sep", "Separation weight", 3.0, 0.0, 6.0, 0.1, "", "w_s"),
        Param("w_ali", "Alignment weight", 1.4, 0.0, 6.0, 0.1, "", "w_a"),
        Param("w_coh", "Cohesion weight", 0.5, 0.0, 6.0, 0.1, "", "w_c"),
        Param("w_flee", "Flee weight", 4.0, 0.0, 10.0, 0.1, "", "w_f"),
        Param("cruise", "Cruise speed", 8.0, 1.0, 20.0, 0.5, "", "v_0", "u/s"),
        Param("burst", "Burst speed", 24.0, 5.0, 45.0, 0.5, "Speed when terrified.", "v_b"),
        Param("r_pred", "Predator detection range", 18.0, 4.0, 40.0, 0.5, "", "r_p"),
        Param("flee_side", "Sideways escape", 0.8, 0.0, 3.0, 0.05,
              "How much fish dodge sideways off the predator's path (fountain effect).", "k"),
        Param("contagion", "Alarm contagion", 0.85, 0.0, 1.0, 0.01,
              "Fraction of a neighbour's fear a fish copies. Near 1: panic spreads far.", "c"),
        Param("fear_tau", "Fear fades over", 1.5, 0.1, 10.0, 0.1, "", "\\tau_f", "s"),
        Param("mem_tau", "Memory fades over", 2.0, 0.1, 15.0, 0.1,
              "How long a fish keeps fleeing from where it last saw the predator.", "\\tau_m",
              "s"),
    ]
    OVERLAYS = [
        Overlay("zones", "Perception zones", True, "The focus fish's three zones and vision cone."),
        Overlay("links", "Neighbour links", True, "Who the focus fish is reacting to."),
        Overlay("forces", "Steering forces", True, "Each rule as an arrow; they add up."),
        Overlay("alarm", "Alarm level", False, "Colour every fish by fear."),
        Overlay("memory", "Predator memory", False, "Where scared fish remember the predator."),
        Overlay("grid", "Spatial hash grid", False, "How the GPU finds neighbours fast."),
        Overlay("trail", "Focus trail", True, "The recent path of the focus fish."),
    ]

    # ---------------------------------------------------------------- setup
    def reset(self, seed: int) -> None:
        super().reset(seed)
        n = int(self.p.n_fish)
        rng = np.random.default_rng(seed)
        centers = rng.uniform((30, 20), (130, 70), size=(4, 2))
        which = rng.integers(0, 4, n)
        xy = centers[which] + rng.normal(0, 7.0, (n, 2))
        xy = np.clip(xy, (2, 2), (WIDTH - 2, HEIGHT - 2))
        heading = rng.uniform(0, 2 * np.pi, 4)[which] + rng.normal(0, 0.4, n)
        vel = np.stack([np.cos(heading), np.sin(heading)], 1) * self.p.cruise
        pos = np.zeros((n, 3), np.float32)
        pos[:, :2] = xy
        dev = self.device
        self.pos = wp.array(pos, dtype=wp.vec3, device=dev)
        self.vel = wp.array(vel.astype(np.float32), dtype=wp.vec2, device=dev)
        self.fear = wp.zeros(n, dtype=float, device=dev)
        self.mem_pos = wp.zeros(n, dtype=wp.vec2, device=dev)
        self.mem_str = wp.zeros(n, dtype=float, device=dev)
        self._back = [wp.empty_like(a) for a in
                      (self.pos, self.vel, self.fear, self.mem_pos, self.mem_str)]
        self.forces = wp.zeros((n, N_FORCES), dtype=wp.vec2, device=dev)
        self.caught = wp.zeros(1, dtype=wp.int32, device=dev)
        self.grid = wp.HashGrid(128, 128, 1, device=dev)
        # host-side, visual only
        self.hue = rng.uniform(0, 1, n).astype(np.float32)
        self.swim = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self._last_draw_t = 0.0
        self.pred_pos = np.array([-100.0, -100.0])
        self.pred_vel = np.zeros(2)
        self.pred_heading = 0.0
        self.pred_on = False
        self._last_mouse = None
        self.focus = min(self.focus, n - 1)
        self.trail = TraceMemory(180, 2)
        self._host = None

    def _params(self) -> School:
        p = self.p
        s = School()
        s.r_rep, s.r_ori, s.r_att = p.r_rep, max(p.r_ori, p.r_rep), max(p.r_att, p.r_ori)
        s.cos_fov = float(np.cos(np.radians(p.fov) / 2))
        s.r_lat = p.r_lat
        s.w_sep, s.w_ali, s.w_coh, s.w_flee, s.w_wall = p.w_sep, p.w_ali, p.w_coh, p.w_flee, 4.0
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

    # ----------------------------------------------------------------- step
    def step(self, inp: InputState) -> None:
        # The cursor is the predator; its smoothed motion is its direction vector.
        if inp.mouse is not None:
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
        for button, at in inp.clicks:
            if button == "left":
                self._pick_focus(at)

        self.grid.build(self.pos, float(max(self.p.r_att, 0.5)))
        b = self._back
        wp.launch(school_step, dim=len(self.pos),
                  inputs=[self.grid.id, self._params(), self.pos, self.vel, self.fear,
                          self.mem_pos, self.mem_str, b[0], b[1], b[2], b[3], b[4],
                          self.forces, self.caught], device=self.device)
        self.pos, b[0] = b[0], self.pos
        self.vel, b[1] = b[1], self.vel
        self.fear, b[2] = b[2], self.fear
        self.mem_pos, b[3] = b[3], self.mem_pos
        self.mem_str, b[4] = b[4], self.mem_str
        self._host = None

    def _pick_focus(self, at) -> None:
        xy = self.pos.numpy()[:, :2]
        self.focus = int(np.argmin(np.sum((xy - np.asarray(at)) ** 2, axis=1)))
        self.trail.clear()

    def _fetch(self):
        if self._host is None:
            self._host = dict(pos=self.pos.numpy()[:, :2], vel=self.vel.numpy(),
                              fear=self.fear.numpy(), mem=self.mem_str.numpy(),
                              mem_pos=self.mem_pos.numpy(), forces=self.forces.numpy())
        return self._host

    def state_arrays(self):
        h = self._fetch()
        return [h["pos"], h["vel"], h["fear"], h["mem"]]

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
            d = pos - pos[f]
            dist = np.linalg.norm(d, axis=1)
            u = d / np.maximum(dist, 1e-6)[:, None]
            hvec = vel[f] / max(np.linalg.norm(vel[f]), 1e-6)
            vis = (dist < P.r_att) & (dist > 1e-6) & (
                (dist < P.r_lat) | (u @ hvec > np.cos(np.radians(P.fov) / 2)))
            vis[f] = False
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
        else:
            # startled fish flash silver: the real "flash expansion" glint
            col = base.copy()
            col[:, :3] *= (1 + 0.35 * fear ** 2)[:, None]
        size = np.stack([1.6 * (0.92 + 0.16 * self.hue), 0.56 * (0.92 + 0.16 * self.hue)], 1)
        s.sprites("fish", pos, ang, size, col, self.swim)

        # focus fish + its steering forces
        s.circles(pos[f], 1.4, pal.rgba("#ffffff", 0.75, 1.1), ring=0.09)
        if self.show.forces:
            F = h["forces"][f]
            scale = 2.5
            for k, (_, c) in enumerate(FORCE_STYLE):
                s.arrows(pos[f], F[k] * scale, 0.14, pal.rgba(c, 0.95, 1.3))
            s.arrows(pos[f], F.sum(0) * scale, 0.2, pal.rgba("#ffffff", 1.0, 1.5))
        if self.show.memory and mem[f] > 0.02:
            s.circles(h["mem_pos"][f], 2.0, pal.rgba(pal.MEMORY, 0.8 * mem[f], 1.4), ring=0.15)

        # the predator (your cursor)
        if self.pred_on:
            pp = self.pred_pos
            threat = min(np.linalg.norm(self.pred_vel) / 40.0, 1.0)
            s.circles(pp, P.r_pred, pal.rgba(pal.CORAL, 0.10 + 0.25 * threat), ring=0.15)
            s.glow(pp, 6.0, pal.rgba(pal.CORAL, 0.10 + 0.25 * threat))
            s.sprites("predator", pp, self.pred_heading, (7.0, 2.3), pal.rgba("#475569"),
                      self.t * 5.0)
            if np.linalg.norm(self.pred_vel) > 1.0:
                s.arrows(pp, self.pred_vel * 0.12, 0.18, pal.rgba(pal.CORAL, 0.9, 1.4))

    # ------------------------------------------------------------------ hud
    def hud(self) -> list[HudItem]:
        h = self._fetch()
        fear = h["fear"]
        items = [
            HudItem("fish", f"{len(fear):,}"),
            HudItem("alarmed", f"{100 * np.mean(fear > 0.3):.0f}%", accent=True),
            HudItem("caught", f"{int(self.caught.numpy()[0])}"),
            HudItem("predator speed", f"{np.linalg.norm(self.pred_vel):.0f} u/s"),
        ]
        nb = getattr(self, "_nbr", None)
        if nb is not None and self.show.links:
            items.append(HudItem("focus sees", f"{nb[0]} close · {nb[1]} align · {nb[2]} far"))
        return items
