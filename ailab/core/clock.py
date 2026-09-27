"""Fixed-timestep clock: real time in, a whole number of simulation steps out."""

from __future__ import annotations


class Clock:
    def __init__(self, dt: float = 1 / 60, max_steps: int = 8):
        self.dt = dt
        self.max_steps = max_steps
        self.speed = 1.0
        self.paused = False
        self._acc = 0.0
        self._single = 0

    def request_step(self, n: int = 1) -> None:
        self._single += n

    def reset(self) -> None:
        self._acc = 0.0
        self._single = 0

    def tick(self, real_dt: float) -> int:
        if self.paused:
            n, self._single = self._single, 0
            return n
        self._acc += min(real_dt, 0.25) * self.speed
        n = int(self._acc / self.dt)
        self._acc -= n * self.dt
        if n > self.max_steps:        # too slow to keep up: drop time, never spiral
            n, self._acc = self.max_steps, 0.0
        return n
