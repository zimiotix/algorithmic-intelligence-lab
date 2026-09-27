"""The sandbox kit: things learners build into a world, shared by every chapter.

``ObstacleGrid`` is a paintable grid of blocked cells (rocks, walls, reefs). The host keeps
a numpy mask the Lab tools paint into; the device copy feeds Warp kernels through three
functions every agent-based chapter can use:

  obstacle_at(blocked, g, p)            is p inside rock (or outside the world)?
  obstacle_push(blocked, g, p, r)       repulsion from rock within r (a "pressure" sense)
  obstacle_clearance(blocked, g, p, u, L)   free distance ahead along u, up to L (a "look")

Painting happens between steps on the host, from the learner's input, so runs stay
deterministic: same seed + same strokes = same world.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .memory import Grid2D, grid_cell, grid_inside


@wp.func
def obstacle_at(blocked: wp.array2d(dtype=wp.uint8), g: Grid2D, p: wp.vec2) -> bool:
    c = grid_cell(g, p)
    if not grid_inside(g, c):
        return True
    return blocked[c[1], c[0]] != wp.uint8(0)


@wp.func
def obstacle_push(blocked: wp.array2d(dtype=wp.uint8), g: Grid2D, p: wp.vec2,
                  radius: float) -> wp.vec2:
    """Sum of pushes away from every blocked cell within ``radius``, each weighted
    (1 - d/r)^2: gentle far away, strong up close. Cells outside the world don't push
    (the chapter's own tank walls handle the border)."""
    c = grid_cell(g, p)
    cell = 1.0 / g.inv_cell
    k = int(wp.ceil(radius * g.inv_cell))
    push = wp.vec2(0.0, 0.0)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            q = wp.vec2i(c[0] + dx, c[1] + dy)
            if grid_inside(g, q):
                if blocked[q[1], q[0]] != wp.uint8(0):
                    center = g.origin + wp.vec2((float(q[0]) + 0.5) * cell,
                                                (float(q[1]) + 0.5) * cell)
                    d = p - center
                    dist = wp.length(d)
                    if dist < radius and dist > 1.0e-5:
                        w = 1.0 - dist / radius
                        push = push + d / dist * (w * w)
    return push


@wp.func
def obstacle_clearance(blocked: wp.array2d(dtype=wp.uint8), g: Grid2D, p: wp.vec2,
                       u: wp.vec2, max_dist: float) -> float:
    """March along direction u in half-cell steps; distance to the first rock (or max)."""
    step = 0.5 / g.inv_cell
    n = int(max_dist / step)
    for s in range(1, n + 1):
        d = float(s) * step
        if obstacle_at(blocked, g, p + u * d):
            return d
    return max_dist


class ObstacleGrid:
    """A paintable blocked-cell grid over the world bounds (x0, y0, x1, y1)."""

    def __init__(self, bounds, cell: float, device: str = "cpu", border: bool = False):
        x0, y0, x1, y1 = bounds
        self.border = border      # keep a solid rim of rock around the world
        self.bounds = bounds
        self.cell = cell
        self.nx = int(np.ceil((x1 - x0) / cell))
        self.ny = int(np.ceil((y1 - y0) / cell))
        self.device = device
        g = Grid2D()
        g.origin = wp.vec2(x0, y0)
        g.inv_cell = 1.0 / cell
        g.nx, g.ny = self.nx, self.ny
        self.grid = g
        self.mask = np.zeros((self.ny, self.nx), np.uint8)
        xs = x0 + (np.arange(self.nx) + 0.5) * cell
        ys = y0 + (np.arange(self.ny) + 0.5) * cell
        self.X, self.Y = np.meshgrid(xs, ys)
        self.blocked = wp.zeros((self.ny, self.nx), dtype=wp.uint8, device=device)
        self.version = 0          # bumps on every change (for cached drawings)
        self.painted = 0          # cells the learner has turned into rock
        self._prev = None         # last point of the stroke in progress
        self._protect: list[tuple[np.ndarray, float]] = []

    # ------------------------------------------------------------------ shapes
    def capsule(self, a, b, r: float) -> np.ndarray:
        """Cells within r of the segment a-b."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ba = b - a
        t = np.clip(((self.X - a[0]) * ba[0] + (self.Y - a[1]) * ba[1]) / max(ba @ ba, 1e-9),
                    0, 1)
        return (self.X - a[0] - ba[0] * t) ** 2 + (self.Y - a[1] - ba[1] * t) ** 2 < r * r

    def protect(self, center, radius: float) -> None:
        """Never paint rock here (a nest, a spawn point)."""
        self._protect.append((np.asarray(center, float), float(radius)))

    def set(self, mask: np.ndarray) -> None:
        self.mask = mask.astype(np.uint8)
        self._rim()
        self.sync()

    def _rim(self) -> None:
        if self.border:
            self.mask[0, :] = self.mask[-1, :] = 1
            self.mask[:, 0] = self.mask[:, -1] = 1

    def sync(self) -> None:
        self.blocked = wp.array(self.mask, dtype=wp.uint8, device=self.device)
        self.version += 1

    def free(self, p) -> bool:
        ix, iy = int((p[0] - self.bounds[0]) / self.cell), int((p[1] - self.bounds[1]) / self.cell)
        return 0 <= ix < self.nx and 0 <= iy < self.ny and self.mask[iy, ix] == 0

    # ---------------------------------------------------------------- painting
    def stroke(self, point, radius: float, value: int) -> bool:
        """Continue a brush stroke to ``point`` (value 1 = rock, 0 = erase).
        Returns True when the mask changed."""
        a = self._prev if self._prev is not None else point
        self._prev = np.asarray(point, float)
        brush = self.capsule(a, point, radius)
        for c, r in self._protect:
            brush &= ~self.capsule(c, c, r)
        before = int(self.mask.sum())
        self.mask[brush] = value
        self._rim()
        changed = int(self.mask.sum()) - before
        if changed:
            self.painted = max(0, self.painted + changed)
            self.sync()
        return changed != 0

    def end_stroke(self) -> None:
        self._prev = None

    # ------------------------------------------------------------------ drawing
    def rock_image(self, base=(0.19, 0.16, 0.14), seed: int = 0) -> np.ndarray:
        """RGBA image for ``Scene.image(..., mode="mask")``: alpha = rock."""
        rng = np.random.default_rng(seed + 99)
        t = 0.92 + 0.08 * rng.random((self.ny, self.nx))
        img = np.zeros((self.ny, self.nx, 4), np.float32)
        img[..., :3] = np.stack([base[0] * t, base[1] * t, base[2] * t], -1)
        img[..., 3] = self.mask
        return img
