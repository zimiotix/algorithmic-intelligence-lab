"""World <-> screen mapping with zoom and pan. Screen = logical pixels, y down."""

from __future__ import annotations


class Camera:
    def __init__(self, world=(0.0, 0.0, 160.0, 90.0), pad: float = 0.03):
        self.pad = pad
        self.set_world(world)

    def set_world(self, world) -> None:
        self.world = tuple(float(v) for v in world)
        self.reset()

    def reset(self) -> None:
        x0, y0, x1, y1 = self.world
        self.zoom = 1.0
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)

    def ppu(self, w: float, h: float) -> float:
        """Logical pixels per world unit."""
        x0, y0, x1, y1 = self.world
        return min(w / (x1 - x0), h / (y1 - y0)) * (1 - self.pad) * self.zoom

    def uniforms(self, w: float, h: float) -> tuple[tuple[float, float], tuple[float, float]]:
        s = self.ppu(w, h)
        sx, sy = 2 * s / w, 2 * s / h
        return (sx, sy), (-self.center[0] * sx, -self.center[1] * sy)

    def to_world(self, px: float, py: float, w: float, h: float) -> tuple[float, float]:
        s = self.ppu(w, h)
        return self.center[0] + (px - w / 2) / s, self.center[1] - (py - h / 2) / s

    def zoom_at(self, factor: float, px: float, py: float, w: float, h: float) -> None:
        wx, wy = self.to_world(px, py, w, h)
        self.zoom = min(16.0, max(0.5, self.zoom * factor))
        s = self.ppu(w, h)
        self.center = (wx - (px - w / 2) / s, wy + (py - h / 2) / s)

    def pan(self, dx: float, dy: float, w: float, h: float) -> None:
        s = self.ppu(w, h)
        self.center = (self.center[0] - dx / s, self.center[1] + dy / s)
