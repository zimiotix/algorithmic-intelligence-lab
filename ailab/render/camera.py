"""World <-> screen mapping with zoom, pan and rotation. Screen = logical pixels, y down.

``rotation`` turns the view (radians, counter-clockwise): a chase camera sets it to
pi/2 - heading so that the followed agent always points up the screen.
"""

from __future__ import annotations

import math


def _rot(x: float, y: float, a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


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
        self.rotation = 0.0
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)

    def ppu(self, w: float, h: float) -> float:
        """Logical pixels per world unit."""
        x0, y0, x1, y1 = self.world
        return min(w / (x1 - x0), h / (y1 - y0)) * (1 - self.pad) * self.zoom

    def uniforms(self, w: float, h: float):
        """(scale, offset, (cos, sin)) for clip = rot(p) * scale + offset."""
        s = self.ppu(w, h)
        sx, sy = 2 * s / w, 2 * s / h
        cx, cy = _rot(*self.center, self.rotation)
        return (sx, sy), (-cx * sx, -cy * sy), (math.cos(self.rotation),
                                               math.sin(self.rotation))

    def _view_to_world(self, vx: float, vy: float) -> tuple[float, float]:
        dx, dy = _rot(vx, vy, -self.rotation)
        return self.center[0] + dx, self.center[1] + dy

    def to_world(self, px: float, py: float, w: float, h: float) -> tuple[float, float]:
        s = self.ppu(w, h)
        return self._view_to_world((px - w / 2) / s, -(py - h / 2) / s)

    def zoom_at(self, factor: float, px: float, py: float, w: float, h: float) -> None:
        wx, wy = self.to_world(px, py, w, h)
        self.zoom = min(16.0, max(0.5, self.zoom * factor))
        s = self.ppu(w, h)
        dx, dy = _rot((px - w / 2) / s, -(py - h / 2) / s, -self.rotation)
        self.center = (wx - dx, wy - dy)

    def pan(self, dx: float, dy: float, w: float, h: float) -> None:
        s = self.ppu(w, h)
        mx, my = _rot(-dx / s, dy / s, -self.rotation)
        self.center = (self.center[0] + mx, self.center[1] + my)

    def follow(self, x: float, y: float, heading: float, lead: float, k: float) -> None:
        """Ease toward a chase view of (x, y) facing ``heading``: the agent sits ``lead``
        world units below the centre, pointing up. ``k`` in [0, 1] is the easing step."""
        target_rot = math.pi / 2 - heading
        d = (target_rot - self.rotation + math.pi) % (2 * math.pi) - math.pi
        self.rotation += d * k
        cx, cy = x + math.cos(heading) * lead, y + math.sin(heading) * lead
        self.center = (self.center[0] + (cx - self.center[0]) * k,
                       self.center[1] + (cy - self.center[1]) * k)
