"""Memory systems for algorithms that need to remember.

Three kinds, and chapters say which one they use:

1. **FieldMemory** (environmental memory / stigmergy)
   Memory written *into the world*: a grid that agents deposit into and that
   evaporates and diffuses over time. Ant pheromones, a robot's occupancy map.
   Runs in Warp on CUDA or CPU. Deposits are integer fixed-point atomics, so the result
   does not depend on GPU thread order (float atomics would break determinism).

2. **Working memory** (a decaying belief held by one agent)
   e.g. "where I last saw the predator", with a strength that fades:
   s(t + dt) = s(t) * exp(-dt / tau). Use ``forget`` inside kernels, ``forget_np`` on host.

3. **TraceMemory** (episodic / short-term trace)
   A ring buffer of recent observations with timestamps: trails, recent lidar scans,
   the last N decisions. Host-side numpy.
"""

from __future__ import annotations

import numpy as np
import warp as wp

FIXED_POINT = 4096.0  # deposit resolution: 1/4096 of a unit


# ------------------------------------------------------------ working memory
@wp.func
def forget(strength: float, dt: float, tau: float) -> float:
    """Exponential forgetting: after tau seconds only 1/e of the memory remains."""
    return strength * wp.exp(-dt / tau)


def forget_np(strength, dt: float, tau: float):
    return strength * np.exp(-dt / tau)


# ---------------------------------------------------------- environmental memory
@wp.struct
class Grid2D:
    """Maps world coordinates onto a grid of cells."""

    origin: wp.vec2
    inv_cell: float
    nx: int
    ny: int


@wp.func
def grid_cell(g: Grid2D, p: wp.vec2) -> wp.vec2i:
    """Cell index (ix, iy) for world point p; may be outside the grid."""
    q = (p - g.origin) * g.inv_cell
    return wp.vec2i(int(wp.floor(q[0])), int(wp.floor(q[1])))


@wp.func
def grid_inside(g: Grid2D, c: wp.vec2i) -> bool:
    return c[0] >= 0 and c[1] >= 0 and c[0] < g.nx and c[1] < g.ny


@wp.func
def field_sample3(value: wp.array3d(dtype=float), ch: int, g: Grid2D, p: wp.vec2) -> float:
    """Sum of a 3x3 neighbourhood around p: what a sensor with a small footprint smells."""
    c = grid_cell(g, p)
    s = float(0.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            q = wp.vec2i(c[0] + dx, c[1] + dy)
            if grid_inside(g, q):
                s += value[ch, q[1], q[0]]
    return s


@wp.func
def field_deposit(deposit: wp.array3d(dtype=wp.int32), ch: int, g: Grid2D, p: wp.vec2,
                  amount: float):
    """Add `amount` at p. Integer atomics => identical result in any thread order."""
    c = grid_cell(g, p)
    if grid_inside(g, c):
        wp.atomic_add(deposit, ch, c[1], c[0], wp.int32(amount * FIXED_POINT))


@wp.kernel
def _field_update(value: wp.array3d(dtype=float), deposit: wp.array3d(dtype=wp.int32),
                  blocked: wp.array2d(dtype=wp.uint8), decay: wp.array(dtype=float),
                  diffusion: wp.array(dtype=float), out: wp.array3d(dtype=float)):
    ch, y, x = wp.tid()
    ny = value.shape[1]
    nx = value.shape[2]
    if blocked[y, x] != wp.uint8(0):
        out[ch, y, x] = 0.0
        deposit[ch, y, x] = 0
        return
    v = value[ch, y, x]
    # 5-point Laplacian with no-flux walls: a blocked/outside neighbour mirrors v.
    lap = float(0.0)
    for k in range(4):
        dx = 0
        dy = 0
        if k == 0:
            dx = 1
        elif k == 1:
            dx = -1
        elif k == 2:
            dy = 1
        else:
            dy = -1
        xx = x + dx
        yy = y + dy
        n = v
        if xx >= 0 and yy >= 0 and xx < nx and yy < ny:
            if blocked[yy, xx] == wp.uint8(0):
                n = value[ch, yy, xx]
        lap += n - v
    v = (v + diffusion[ch] * lap) * decay[ch]
    v += float(deposit[ch, y, x]) / FIXED_POINT
    deposit[ch, y, x] = 0
    out[ch, y, x] = v


@wp.kernel
def _deposit_points(points: wp.array(dtype=wp.vec2), amount: float, ch: int, g: Grid2D,
                    deposit: wp.array3d(dtype=wp.int32)):
    i = wp.tid()
    field_deposit(deposit, ch, g, points[i], amount)


class FieldMemory:
    """A multi-channel grid memory with evaporation and diffusion.

    Per step and channel:  v <- (v + D * lap(v)) * exp(-dt / tau) + deposits
    """

    def __init__(self, bounds: tuple[float, float, float, float], cell: float,
                 channels: int = 1, tau: float | list[float] = 30.0,
                 diffusion: float | list[float] = 0.0, device: str = "cpu"):
        x0, y0, x1, y1 = bounds
        self.bounds = bounds
        self.cell = cell
        self.nx = int(np.ceil((x1 - x0) / cell))
        self.ny = int(np.ceil((y1 - y0) / cell))
        self.channels = channels
        self.device = device
        g = Grid2D()
        g.origin = wp.vec2(x0, y0)
        g.inv_cell = 1.0 / cell
        g.nx, g.ny = self.nx, self.ny
        self.grid = g
        shape = (channels, self.ny, self.nx)
        self.value = wp.zeros(shape, dtype=float, device=device)
        self._back = wp.zeros(shape, dtype=float, device=device)
        self.deposit = wp.zeros(shape, dtype=wp.int32, device=device)
        self.blocked = wp.zeros((self.ny, self.nx), dtype=wp.uint8, device=device)
        self.tau = np.broadcast_to(np.asarray(tau, np.float32), (channels,)).copy()
        self.diffusion = np.broadcast_to(np.asarray(diffusion, np.float32), (channels,)).copy()

    def set_blocked(self, mask: np.ndarray) -> None:
        self.blocked = wp.array(mask.astype(np.uint8), dtype=wp.uint8, device=self.device)

    def clear(self) -> None:
        self.value.zero_()
        self.deposit.zero_()

    def update(self, dt: float) -> None:
        decay = wp.array(np.exp(-dt / np.maximum(self.tau, 1e-6)).astype(np.float32),
                         dtype=float, device=self.device)
        # Explicit diffusion is stable while D*dt <= 1/4 (cells^2).
        diff = wp.array(np.minimum(self.diffusion * dt, 0.24).astype(np.float32),
                        dtype=float, device=self.device)
        wp.launch(_field_update, dim=(self.channels, self.ny, self.nx),
                  inputs=[self.value, self.deposit, self.blocked, decay, diff, self._back],
                  device=self.device)
        self.value, self._back = self._back, self.value

    def deposit_points(self, points: np.ndarray, amount: float, ch: int = 0) -> None:
        if len(points) == 0:
            return
        pts = wp.array(np.asarray(points, np.float32), dtype=wp.vec2, device=self.device)
        wp.launch(_deposit_points, dim=len(points),
                  inputs=[pts, float(amount), ch, self.grid, self.deposit], device=self.device)

    def numpy(self) -> np.ndarray:
        return self.value.numpy()


# ------------------------------------------------------------- episodic memory
class TraceMemory:
    """Fixed-capacity ring buffer of (time, vector) observations."""

    def __init__(self, capacity: int, dim: int):
        self.capacity = capacity
        self.times = np.zeros(capacity, np.float64)
        self.values = np.zeros((capacity, dim), np.float32)
        self.count = 0
        self.head = 0

    def clear(self) -> None:
        self.count = 0
        self.head = 0

    def push(self, t: float, value) -> None:
        self.times[self.head] = t
        self.values[self.head] = value
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def ordered(self) -> tuple[np.ndarray, np.ndarray]:
        """Oldest -> newest."""
        if self.count < self.capacity:
            return self.times[: self.count], self.values[: self.count]
        idx = (np.arange(self.capacity) + self.head) % self.capacity
        return self.times[idx], self.values[idx]

    def __len__(self) -> int:
        return self.count
