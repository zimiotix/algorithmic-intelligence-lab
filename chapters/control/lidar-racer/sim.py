"""Lidar racer: an autonomous car that drives only from what its lidar sees.

Pipeline, every step (1/60 s):
  1. SENSE    cast N rays; each returns the distance to the first wall or cone.
  2. PLAN     Follow-the-Gap: find the widest free gap, aim into it.
  3. STEER    pure pursuit turns the aim point into a steering angle.
  4. SPEED    go slower when turning hard, and never faster than you can stop
              in the free distance ahead (v^2 = 2 a d).
  5. MOVE     kinematic bicycle model.
Memory: the car keeps an occupancy map of past lidar hits (environmental memory) and a
trail of where it has been (episodic trace). The controller doesn't need them; they
show what "a map" is, and the next chapters (mapping, planning) build on them.

Pure numpy: this is light enough for the CPU, even with hundreds of rays.
"""

from __future__ import annotations

import numpy as np

from ailab.core import HudItem, InputState, Overlay, Param, Simulation
from ailab.core.memory import FieldMemory, TraceMemory
from ailab.render import palette as pal

WIDTH, HEIGHT = 160.0, 90.0
CAR_L, CAR_W, WHEELBASE = 3.2, 1.5, 2.0
HALF_TRACK = 4.6
CONE_R = 0.55


def _arc(cx, cy, r, a0, a1):
    n = max(8, int(abs(np.radians(a1 - a0)) * r / 0.2))
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _hairpins() -> np.ndarray:
    """Straights joined by true circular arcs, so every hairpin has a known radius."""
    parts = [
        np.array([(24, 12), (128, 12)]), _arc(128, 26, 14, -90, 90),
        np.array([(128, 40), (52, 40)]), _arc(52, 52, 12, 270, 90),
        np.array([(52, 64), (128, 64)]), _arc(128, 74, 10, -90, 90),
        np.array([(128, 84), (24, 84)]), _arc(24, 72, 12, 90, 180),
        np.array([(12, 72), (12, 24)]), _arc(24, 24, 12, 180, 270),
    ]
    return np.vstack(parts)[:-1]


TRACKS = {
    # control points -> Catmull-Rom spline; arrays -> used as drawn
    "Grand Prix": [(40, 10), (90, 10), (134, 11), (149, 22), (151, 40), (144, 56), (128, 61),
                   (112, 56), (98, 62), (86, 73), (60, 80), (30, 79), (14, 64), (11, 38),
                   (18, 18)],
    "Hairpins": _hairpins,
    "Oval": [(80 + 64 * np.cos(a), 45 + 32 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 13)[:-1]],
}


def track_centerline(name: str) -> np.ndarray:
    spec = TRACKS[name]
    path = spec() if callable(spec) else catmull_rom_closed(spec)
    return resample_closed(path, 0.5)


def min_turn_radius(c: np.ndarray) -> float:
    """Smallest radius of curvature along a closed centreline (3-point circle fit)."""
    a, b, d = np.roll(c, 1, 0), c, np.roll(c, -1, 0)
    ab, bd, ad = (np.linalg.norm(b - a, axis=1), np.linalg.norm(d - b, axis=1),
                  np.linalg.norm(d - a, axis=1))
    cross = np.abs((b - a)[:, 0] * (d - a)[:, 1] - (b - a)[:, 1] * (d - a)[:, 0])
    with np.errstate(divide="ignore"):
        r = ab * bd * ad / (2 * cross)
    return float(np.min(r))


# ---------------------------------------------------------------- geometry
def catmull_rom_closed(pts, per_seg: int = 40) -> np.ndarray:
    P = np.asarray(pts, float)
    n = len(P)
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    return np.vstack(out)


def resample_closed(path: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    samples = np.arange(0, s[-1], spacing)
    return np.stack([np.interp(samples, s, closed[:, 0]), np.interp(samples, s, closed[:, 1])], 1)


def raycast(origin, dirs, seg_a, seg_b, circles, radius, max_range) -> np.ndarray:
    """Distance along each ray to the first segment or circle (vectorised).

    Ray o + t d meets segment a + u e when o + t d = a + u e: two linear equations,
    solved with 2D cross products (Cramer's rule)."""
    t_best = np.full(len(dirs), max_range)
    if len(seg_a):
        e = seg_b - seg_a
        w = seg_a - origin
        denom = dirs[:, 0:1] * e[None, :, 1] - dirs[:, 1:2] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
            u = (w[None, :, 0] * dirs[:, 1:2] - w[None, :, 1] * dirs[:, 0:1]) / denom
        ok = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t_best = np.minimum(t_best, np.where(ok, t, np.inf).min(axis=1))
    if len(circles):
        oc = origin - circles                                # (M, 2)
        b = dirs @ oc.T                                      # (K, M)
        c = np.sum(oc * oc, axis=1)[None, :] - radius**2
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        ok = (disc >= 0) & (t > 0)
        t_best = np.minimum(t_best, np.where(ok, t, np.inf).min(axis=1))
    return t_best


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True values."""
    best_s, best_n, s = 0, 0, None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s > best_n:
                best_s, best_n = s, i - s
            s = None
    return best_s, best_n


def follow_the_gap(ranges, angles, max_range, bubble, threshold, aim):
    """Returns a dict describing every intermediate decision (for the overlays)."""
    r = np.minimum(ranges, max_range)
    rs = np.convolve(np.pad(r, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    i_min = int(np.argmin(rs))
    half = np.arctan2(bubble, max(rs[i_min], 1e-3))
    in_bubble = np.abs(angles - angles[i_min]) < half
    free = (rs > threshold) & ~in_bubble
    start, length = longest_run(free)
    if length == 0:
        idx = int(np.argmax(np.where(in_bubble, 0, rs)))
        start, length = idx, 1
    gap = slice(start, start + length)
    deepest = start + int(np.argmax(rs[gap]))
    centre = start + length // 2
    idx = {"Deepest point": deepest, "Gap centre": centre}.get(aim, (deepest + centre) // 2)
    return dict(smoothed=rs, i_min=i_min, bubble=in_bubble, free=free, gap=(start, length),
                target=idx, alpha=float(angles[idx]))


def pure_pursuit(alpha: float, target_dist: float, v: float, gain: float, ld_min: float):
    """Steering that puts the rear axle on a circle through the aim point.

    Look ahead further at higher speed (L_d = k v + L_min), but never past the target."""
    ld = float(np.clip(gain * v + ld_min, ld_min, max(target_dist, ld_min)))
    return float(np.arctan2(2 * WHEELBASE * np.sin(alpha), ld)), ld


class Sim(Simulation):
    world = (0.0, 0.0, WIDTH, HEIGHT)
    background = "grass"
    seeded = False          # no randomness unless sensor noise is turned up
    PARAMS = [
        Param("track", "Track", "Grand Prix", choices=tuple(TRACKS), restart=True),
        Param("n_rays", "Lidar rays", 61, 5, 361, 2, "More rays: finer view, more work.", "N"),
        Param("fov", "Lidar field of view", 240.0, 60.0, 360.0, 5.0, "", "\\Phi", "deg"),
        Param("range", "Lidar range", 30.0, 5.0, 80.0, 1.0, "", "R_{max}"),
        Param("noise", "Sensor noise", 0.0, 0.0, 1.5, 0.05,
              "Gaussian noise on each range (seeded, so still deterministic).", "\\sigma"),
        Param("bubble", "Safety bubble", 1.6, 0.0, 6.0, 0.1,
              "Rays around the closest obstacle are treated as blocked.", "b"),
        Param("threshold", "Gap threshold", 6.0, 1.0, 30.0, 0.5,
              "A ray counts as free if it sees farther than this.", "r_{gap}"),
        Param("aim", "Aim point", "Blend", choices=("Blend", "Deepest point", "Gap centre")),
        Param("lookahead_gain", "Lookahead gain", 0.3, 0.0, 1.5, 0.05,
              "Lookahead distance grows with speed: L_d = k v + L_min.", "k", "s"),
        Param("lookahead_min", "Min lookahead", 3.0, 1.0, 20.0, 0.5, "", "L_{min}"),
        Param("v_max", "Top speed", 24.0, 3.0, 45.0, 0.5, "", "v_{max}", "u/s"),
        Param("a_brake", "Braking", 18.0, 2.0, 40.0, 0.5, "Max deceleration.", "a_b", "u/s^2"),
        Param("k_turn", "Slow down in turns", 2.0, 0.0, 8.0, 0.1, "", "k_\\delta"),
        Param("steer_rate", "Steering speed", 3.0, 0.5, 10.0, 0.1, "", "\\dot\\delta_{max}",
              "rad/s"),
        Param("map_tau", "Map memory", 60.0, 2.0, 600.0, 1.0,
              "How long the occupancy map remembers a wall it no longer sees.", "\\tau_m", "s"),
    ]
    OVERLAYS = [
        Overlay("rays", "Lidar rays", True, "Every ray, fading with distance."),
        Overlay("hits", "Hit points", True),
        Overlay("bubble", "Safety bubble", True, "Rays masked around the nearest obstacle."),
        Overlay("gap", "Chosen gap & target", True),
        Overlay("steer", "Pure pursuit geometry", False, "Lookahead circle and target line."),
        Overlay("brake", "Braking envelope", True, "Stopping distance v^2/2a vs free space."),
        Overlay("predict", "Predicted path", True, "Where the car goes if nothing changes."),
        Overlay("scan", "Scan plot", True, "The lidar scan as a bar chart."),
        Overlay("map", "Occupancy memory", False, "Everything the lidar has hit recently."),
        Overlay("trail", "Speed trail", True),
    ]

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.rng = np.random.default_rng(seed)
        c = track_centerline(self.p.track)
        tan = np.roll(c, -1, 0) - np.roll(c, 1, 0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        self.center, self.tangent = c, tan
        self.left, self.right = c + nrm * HALF_TRACK, c - nrm * HALF_TRACK
        L, R = self.left, self.right
        self.seg_a = np.vstack([L, R])
        self.seg_b = np.vstack([np.roll(L, -1, 0), np.roll(R, -1, 0)])
        self.seg_mid = (self.seg_a + self.seg_b) / 2
        self._build_meshes()
        self.cones = np.zeros((0, 2))
        self.x, self.y = c[0]
        self.theta = float(np.arctan2(tan[0, 1], tan[0, 0]))
        self.v, self.delta, self.accel, self.v_target = 0.0, 0.0, 0.0, 0.0
        self.progress, self.lap, self.crashes = 0, 0, 0
        self.lap_start, self.last_lap, self.best_lap = 0.0, None, None
        self.crash_flash = 0.0
        self.map = FieldMemory(self.world, 0.5, channels=1, tau=self.p.map_tau,
                               device=self.device)
        self.trail = TraceMemory(600, 3)
        self._sense_and_plan()

    def _build_meshes(self) -> None:
        L, R = self.left, self.right
        L2, R2 = np.roll(L, -1, 0), np.roll(R, -1, 0)
        self.track_mesh = np.stack([L, R, L2, R, R2, L2], 1).reshape(-1, 2)
        curbs, colors = [], []
        for edge, inward in ((L, -1), (R, 1)):
            nrm = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], 1) * inward * 0.7
            for k, i in enumerate(range(0, len(edge), 4)):
                j = min(i + 4, len(edge) - 1) if i + 4 < len(edge) else 0
                a, b = edge[i], edge[j]
                a2, b2 = a + nrm[i], b + nrm[j]
                curbs.append([a, b, b2, a, b2, a2])
                colors += [pal.rgba("#dc2626" if k % 2 else "#f1f5f9", 0.9)] * 6
        self.curb_mesh = np.array(curbs).reshape(-1, 2)
        self.curb_colors = np.array(colors)
        # chequered start/finish line
        tiles, tcol = [], []
        n_across = 8
        a, b = self.left[0], self.right[0]
        t = self.tangent[0] * 0.6
        for i in range(n_across):
            p0 = a + (b - a) * i / n_across
            p1 = a + (b - a) * (i + 1) / n_across
            for row in range(2):
                o = t * row
                tiles.append([p0 + o, p1 + o, p1 + o + t, p0 + o, p1 + o + t, p0 + o + t])
                tcol += [pal.rgba("#f8fafc" if (i + row) % 2 else "#0f172a")] * 6
        self.start_mesh = np.array(tiles).reshape(-1, 2)
        self.start_colors = np.array(tcol)

    # -------------------------------------------------------------- the brain
    @property
    def sensor(self) -> np.ndarray:
        return np.array([self.x, self.y]) + 0.8 * np.array([np.cos(self.theta),
                                                            np.sin(self.theta)])

    def _sense_and_plan(self) -> None:
        P = self.p
        n = int(P.n_rays) | 1                         # odd: one ray straight ahead
        self.angles = np.radians(np.linspace(-P.fov / 2, P.fov / 2, n))
        world_ang = self.theta + self.angles
        dirs = np.stack([np.cos(world_ang), np.sin(world_ang)], 1)
        o = self.sensor
        near = np.linalg.norm(self.seg_mid - o, axis=1) < P.range + 1.0
        r = raycast(o, dirs, self.seg_a[near], self.seg_b[near], self.cones, CONE_R, P.range)
        if P.noise > 0:
            r = np.clip(r + self.rng.normal(0, P.noise, n), 0.05, P.range)
        self.ranges, self.dirs = r, dirs
        plan = follow_the_gap(r, self.angles, P.range, P.bubble, P.threshold, P.aim)
        plan["delta"], plan["lookahead"] = pure_pursuit(
            plan["alpha"], float(plan["smoothed"][plan["target"]]), self.v, P.lookahead_gain,
            P.lookahead_min)
        self.plan = plan
        ahead = np.abs(self.angles) < np.radians(10)
        self.front = float(r[ahead].min()) if ahead.any() else float(r.min())

    def step(self, inp: InputState) -> None:
        P, dt = self.p, self.dt
        self._interact(inp)
        self._sense_and_plan()
        plan = self.plan
        # speed: slower in turns, and able to stop within the free distance ahead
        v_turn = P.v_max / (1 + P.k_turn * abs(plan["delta"]))
        v_stop = np.sqrt(max(0.0, 2 * P.a_brake * (self.front - 1.5)))
        self.v_target = min(P.v_max, v_turn, v_stop)
        self.accel = float(np.clip(3.0 * (self.v_target - self.v), -P.a_brake, 12.0))
        # steering actuator can only turn so fast
        d = np.clip(plan["delta"] - self.delta, -P.steer_rate * dt, P.steer_rate * dt)
        self.delta = float(np.clip(self.delta + d, -0.5, 0.5))
        # disturbances from the keyboard: push the car and watch it recover
        if "Left" in inp.keys:
            self.theta += 1.6 * dt
        if "Right" in inp.keys:
            self.theta -= 1.6 * dt
        if "Up" in inp.keys:
            self.v += 10 * dt
        if "Down" in inp.keys:
            self.v = max(0.0, self.v - 15 * dt)
        # kinematic bicycle model
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / WHEELBASE * np.tan(self.delta) * dt
        self.v = max(0.0, self.v + self.accel * dt)
        self._collide_and_score()
        hits = self.sensor + self.dirs * self.ranges[:, None]
        self.map.deposit_points(hits[self.ranges < P.range - 1e-3], 0.25)
        self.map.update(dt)
        self.trail.push(self.t, (self.x, self.y, self.v))
        self.crash_flash = max(0.0, self.crash_flash - dt)

    def _interact(self, inp: InputState) -> None:
        for button, at in inp.clicks:
            at = np.asarray(at, float)
            if button == "left" and np.hypot(at[0] - self.x, at[1] - self.y) > 4:
                self.cones = np.vstack([self.cones, at[None]])
            elif button == "right" and len(self.cones):
                d = np.linalg.norm(self.cones - at, axis=1)
                if d.min() < 3:
                    self.cones = np.delete(self.cones, int(np.argmin(d)), 0)
        if "C" in inp.key_presses:
            self.cones = np.zeros((0, 2))
        if self.map.tau[0] != self.p.map_tau:
            self.map.tau[:] = self.p.map_tau

    def _collide_and_score(self) -> None:
        p = np.array([self.x, self.y])
        n = len(self.center)
        window = (self.progress + np.arange(-20, 60)) % n
        dist = np.linalg.norm(self.center[window] - p, axis=1)
        k = int(window[np.argmin(dist)])
        if self.progress > n * 0.8 and k < n * 0.2:
            self.lap += 1
            self.last_lap = self.t - self.lap_start
            self.best_lap = self.last_lap if self.best_lap is None else min(self.best_lap,
                                                                            self.last_lap)
            self.lap_start = self.t
        self.progress = k
        off_track = dist.min() > HALF_TRACK - CAR_W * 0.45
        hit_cone = len(self.cones) and np.min(np.linalg.norm(self.cones - p, axis=1)) < \
            CONE_R + CAR_W * 0.5
        if off_track or hit_cone:
            self.crashes += 1
            self.crash_flash = 0.6
            j = (k + 6) % n
            if len(self.cones):     # don't respawn inside a cone
                for _ in range(20):
                    if np.min(np.linalg.norm(self.cones - self.center[j], axis=1)) > 3:
                        break
                    j = (j + 4) % n
            self.x, self.y = self.center[j]
            self.theta = float(np.arctan2(self.tangent[j, 1], self.tangent[j, 0]))
            self.v, self.delta = 0.0, 0.0
            self.progress = j

    def state_arrays(self):
        return [np.array([self.x, self.y, self.theta, self.v, self.delta]), self.ranges]

    # ------------------------------------------------------------------ draw
    def draw(self, s) -> None:
        P, plan = self.p, self.plan
        s.background("grass")
        s.mesh(self.track_mesh + np.array([0.35, -0.35]), pal.rgba("#000000", 0.35))
        s.mesh(self.track_mesh, pal.rgba("#2b2f36"), style="asphalt")
        s.mesh(self.curb_mesh, self.curb_colors)
        s.polyline(self.left, 0.12, pal.rgba("#e2e8f0", 0.7), closed=True)
        s.polyline(self.right, 0.12, pal.rgba("#e2e8f0", 0.7), closed=True)
        s.mesh(self.start_mesh, self.start_colors)

        if self.show.map:
            v = self.map.numpy()[0]
            img = np.zeros(v.shape + (4,), np.float32)
            a = 1 - np.exp(-0.6 * v)
            img[..., 0], img[..., 1], img[..., 2] = 0.8 * a, 0.5 * a, 1.6 * a
            img[..., 3] = a
            s.image(img, self.world, additive=True)

        if self.show.trail and len(self.trail) > 2:
            _, tr = self.trail.ordered()
            col = pal.ramp([(0, "#38bdf8"), (0.5, "#facc15"), (1, "#f43f5e")],
                           tr[:-1, 2] / max(P.v_max, 1e-3))
            col[:, 3] = np.linspace(0.0, 0.8, len(tr) - 1)
            s.polyline(tr[:, :2], 0.22, col)

        if len(self.cones):
            s.circles(self.cones + np.array([0.15, -0.15]), CONE_R * 1.1, pal.rgba("#000", 0.4))
            s.sprites("cone", self.cones, 0.0, (CONE_R * 2.2, CONE_R * 2.2),
                      pal.rgba("#f97316"))

        o = self.sensor
        hits = o + self.dirs * self.ranges[:, None]
        near = 1 - np.clip(self.ranges / P.range, 0, 1)
        if self.show.rays:
            col = pal.ramp([(0, "#22d3ee"), (0.6, "#facc15"), (1, "#f43f5e")], near)
            col[:, 3] = 0.10 + 0.25 * near
            s.lines(np.broadcast_to(o, hits.shape), hits, 0.05, col, additive=True)
        if self.show.bubble and plan["bubble"].any():
            b = plan["bubble"]
            s.lines(np.broadcast_to(o, hits[b].shape), hits[b], 0.08,
                    pal.rgba(pal.CORAL, 0.45, 1.3), additive=True)
            s.glow(hits[plan["i_min"]], 1.8, pal.rgba(pal.CORAL, 0.5))
        if self.show.gap:
            st, ln = plan["gap"]
            a0 = self.theta + self.angles[st]
            a1 = self.theta + self.angles[min(st + ln - 1, len(self.angles) - 1)]
            reach = min(P.range, float(np.median(self.ranges[st:st + ln])))
            s.wedge(o, reach, (a0 + a1) / 2, max((a1 - a0) / 2, 0.01),
                    pal.rgba(pal.ATTRACT, 0.10), additive=True)
            tgt = hits[plan["target"]]
            s.lines(o, tgt, 0.12, pal.rgba(pal.ATTRACT, 0.9, 1.4), additive=True)
            s.circles(tgt, 0.7, pal.rgba(pal.ATTRACT, 1.0, 1.6), ring=0.18)
            s.glow(tgt, 1.6, pal.rgba(pal.ATTRACT, 0.35))
        if self.show.hits:
            s.circles(hits[self.ranges < P.range - 1e-3], 0.16,
                      pal.rgba("#fde68a", 0.85, 1.5), soft=0.25, additive=True)
        if self.show.predict:
            s.polyline(self._predict(), 0.14, pal.rgba("#e0f2fe", 0.45), additive=True)
        if self.show.steer:
            s.circles(o, plan["lookahead"], pal.rgba(pal.SENSE, 0.5), ring=0.08)
        if self.show.brake:
            h = np.array([np.cos(self.theta), np.sin(self.theta)])
            stop = self.v**2 / (2 * P.a_brake)
            s.lines(o, o + h * stop, 0.35, pal.rgba(pal.AMBER, 0.55, 1.2), additive=True)
            s.lines(o + h * self.front - np.array([-h[1], h[0]]) * 1.2,
                    o + h * self.front + np.array([-h[1], h[0]]) * 1.2, 0.18,
                    pal.rgba(pal.CORAL, 0.8, 1.3))

        # the car
        hd = np.array([np.cos(self.theta), np.sin(self.theta)])
        side = np.array([-hd[1], hd[0]])
        pos = np.array([self.x, self.y]) + hd * (CAR_L * 0.25)
        front = pos + hd * CAR_L * 0.5
        s.wedge(front, 9.0, self.theta, 0.35, pal.rgba("#fef9c3", 0.05), additive=True)
        s.circles(pos + np.array([0.3, -0.3]), 1.2, pal.rgba("#000", 0.35), soft=0.6)
        s.sprites("car", pos, self.theta, (CAR_L, CAR_W), pal.rgba("#e11d48"), self.delta)
        s.glow([front + side * 0.5, front - side * 0.5], 0.7, pal.rgba("#fef3c7", 0.9, 1.8))
        if self.accel < -3:
            back = pos - hd * CAR_L * 0.5
            s.glow([back + side * 0.5, back - side * 0.5], 0.8, pal.rgba("#ef4444", 1.0, 2.5))
        if self.crash_flash > 0:
            s.glow(pos, 6.0 * self.crash_flash + 1, pal.rgba(pal.CORAL, self.crash_flash))

        if self.show.scan:
            self._draw_scan(s)

    def _predict(self, seconds: float = 1.5) -> np.ndarray:
        x, y, th = self.x, self.y, self.theta
        pts = [(x, y)]
        for _ in range(int(seconds / 0.05)):
            x += self.v * np.cos(th) * 0.05
            y += self.v * np.sin(th) * 0.05
            th += self.v / WHEELBASE * np.tan(self.delta) * 0.05
            pts.append((x, y))
        return np.array(pts)

    def _draw_scan(self, s) -> None:
        """Screen-space bar chart: range per ray, left = car's right side."""
        w, h = s.view
        bw, bh = min(420.0, w * 0.32), 120.0
        x0, y1 = w - bw - 20, h - 20
        y0 = y1 - bh
        s.rect(x0 - 10, y0 - 10, x0 + bw + 10, y1 + 10, pal.rgba("#0b1220", 0.72), screen=True)
        n = len(self.ranges)
        xs = x0 + (np.arange(n) + 0.5) * bw / n
        r = self.plan["smoothed"] / self.p.range
        col = np.tile(pal.rgba(pal.SENSE, 0.55), (n, 1))
        st, ln = self.plan["gap"]
        col[st:st + ln] = pal.rgba(pal.ATTRACT, 0.85)
        col[self.plan["bubble"]] = pal.rgba(pal.CORAL, 0.85)
        col[self.plan["target"]] = pal.rgba("#ffffff", 1.0, 1.6)
        bar = max(1.0, bw / n * 0.7)
        s.lines(np.stack([xs, np.full(n, y1)], 1), np.stack([xs, y1 - r * bh], 1), bar, col,
                screen=True)
        ty = y1 - self.p.threshold / self.p.range * bh
        s.lines((x0, ty), (x0 + bw, ty), 1.0, pal.rgba(pal.AMBER, 0.8), screen=True)

    def hud(self) -> list[HudItem]:
        fmt = lambda t: "–" if t is None else f"{t:.2f} s"  # noqa: E731
        st, ln = self.plan["gap"]
        gap_deg = np.degrees(self.angles[min(st + ln - 1, len(self.angles) - 1)]
                             - self.angles[st])
        return [
            HudItem("speed", f"{self.v * 3.6:5.0f} km/h", accent=True),
            HudItem("steering", f"{np.degrees(self.delta):+5.1f}°"),
            HudItem("gap", f"{gap_deg:.0f}° wide"),
            HudItem("lap", f"{self.lap}  ·  {self.t - self.lap_start:.1f} s"),
            HudItem("last / best", f"{fmt(self.last_lap)} / {fmt(self.best_lap)}"),
            HudItem("crashes", f"{self.crashes}"),
        ]
