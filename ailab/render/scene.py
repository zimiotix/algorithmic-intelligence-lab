"""The drawing API chapters use. Pure numpy: it records commands, the renderer draws them.

Because a Scene does not touch OpenGL, chapter drawing code runs in headless tests.
Coordinates are world units unless ``screen=True`` (logical pixels, origin top-left).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .palette import rgba

SPRITES = {"fish": 0, "predator": 1, "ant": 2, "car": 3, "arrowhead": 4, "cone": 5,
           "chevron": 6}
BACKGROUNDS = {"void": 0, "water": 1, "soil": 2, "grass": 3}
MESH_STYLES = {"flat": 0, "asphalt": 1, "rock": 2}


@dataclass
class Command:
    kind: str
    data: dict = field(default_factory=dict)


def _color(c, n: int) -> np.ndarray:
    if isinstance(c, str):
        c = rgba(c)
    c = np.asarray(c, np.float32)
    if c.ndim == 1:
        return np.broadcast_to(c, (n, 4))
    return c.reshape(n, 4)


def _col(v, n: int) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return np.broadcast_to(v, (n,)) if v.ndim == 0 else v.reshape(n)


def _pts(p) -> np.ndarray:
    return np.asarray(p, np.float32).reshape(-1, 2)


class Scene:
    def __init__(self, px: float = 0.1, time: float = 0.0):
        self.commands: list[Command] = []
        self.px = px            # world units per logical pixel (for pixel-sized strokes)
        self.view = (1280.0, 720.0)   # viewport size in logical pixels (for screen=True)
        self.time = time

    def _add(self, kind: str, **data) -> None:
        self.commands.append(Command(kind, data))

    # ------------------------------------------------------------- backgrounds
    def background(self, style: str = "void", tint=(1.0, 1.0, 1.0)) -> None:
        self._add("background", style=BACKGROUNDS[style], tint=tuple(tint))

    # ------------------------------------------------------------- primitives
    def lines(self, a, b, width=0.1, color="#ffffff", additive=False, screen=False) -> None:
        a, b = _pts(a), _pts(b)
        n = len(a)
        if n == 0:
            return
        inst = np.empty((n, 9), np.float32)
        inst[:, 0:2], inst[:, 2:4] = a, b
        inst[:, 4] = _col(width, n)
        inst[:, 5:9] = _color(color, n)
        self._add("lines", inst=inst, additive=additive, screen=screen)

    def polyline(self, points, width=0.1, color="#ffffff", closed=False, additive=False,
                 screen=False) -> None:
        p = _pts(points)
        if len(p) < 2:
            return
        a, b = p[:-1], p[1:]
        if closed:
            a, b = np.vstack([a, p[-1:]]), np.vstack([b, p[:1]])
        col = color
        if not isinstance(color, str) and np.asarray(color).ndim == 2:
            col = np.asarray(color, np.float32)
            col = col[: len(a)] if len(col) >= len(a) else np.vstack([col, col[-1:]])
        self.lines(a, b, width, col, additive, screen)

    def circles(self, centers, radius=0.5, color="#ffffff", ring=0.0, soft=0.0,
                additive=False, screen=False) -> None:
        c = _pts(centers)
        n = len(c)
        if n == 0:
            return
        inst = np.empty((n, 9), np.float32)
        inst[:, 0:2] = c
        inst[:, 2] = _col(radius, n)
        inst[:, 3] = _col(ring, n)
        inst[:, 4] = _col(soft, n)
        inst[:, 5:9] = _color(color, n)
        self._add("circles", inst=inst, additive=additive, screen=screen)

    def glow(self, centers, radius, color, additive=True) -> None:
        """Soft light blob (a circle with no hard edge)."""
        self.circles(centers, 0.0, color, soft=radius, additive=additive)

    def sprites(self, kind: str, pos, angle=0.0, size=(1.0, 0.5), color="#ffffff",
                phase=0.0, additive=False, screen=False) -> None:
        p = _pts(pos)
        n = len(p)
        if n == 0:
            return
        inst = np.empty((n, 10), np.float32)
        inst[:, 0:2] = p
        inst[:, 2] = _col(angle, n)
        s = np.asarray(size, np.float32)
        inst[:, 3:5] = np.broadcast_to(s, (n, 2)) if s.ndim == 1 else s.reshape(n, 2)
        inst[:, 5:9] = _color(color, n)
        inst[:, 9] = _col(phase, n)
        self._add("sprites", inst=inst, shape=SPRITES[kind], additive=additive, screen=screen)

    def arrows(self, origins, vectors, width=0.15, color="#ffffff", head=None,
               additive=False, min_len=1e-3) -> None:
        o, v = _pts(origins), _pts(vectors)
        ln = np.linalg.norm(v, axis=1)
        keep = ln > min_len
        if not keep.any():
            return
        o, v, ln = o[keep], v[keep], ln[keep]
        col = color if isinstance(color, str) or np.asarray(color).ndim == 1 else \
            np.asarray(color)[keep]
        h = np.minimum(ln * 0.45, head if head is not None else width * 4.0)
        d = v / ln[:, None]
        tip = o + v
        self.lines(o, tip - d * h[:, None] * 0.6, width, col, additive)
        self.sprites("arrowhead", tip - d * h[:, None] * 0.5, np.arctan2(d[:, 1], d[:, 0]),
                     np.stack([h, h * 0.9], 1), col, additive=additive)

    def mesh(self, vertices, color="#ffffff", style: str = "flat", additive=False,
             screen=False) -> None:
        """Triangles: vertices (3k, 2), color (4,) or per-vertex (3k, 4)."""
        v = _pts(vertices)
        n = len(v)
        if n == 0:
            return
        data = np.empty((n, 6), np.float32)
        data[:, 0:2] = v
        data[:, 2:6] = _color(color, n)
        self._add("mesh", data=data, style=MESH_STYLES[style], additive=additive, screen=screen)

    def polygon(self, points, color, style="flat", additive=False) -> None:
        """Convex (or star-shaped around its centroid) polygon as a triangle fan."""
        p = _pts(points)
        c = p.mean(0)
        a, b = p, np.roll(p, -1, 0)
        tris = np.stack([np.broadcast_to(c, a.shape), a, b], 1).reshape(-1, 2)
        self.mesh(tris, color, style, additive)

    def rect(self, x0, y0, x1, y1, color, screen=False, additive=False) -> None:
        v = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]], np.float32)
        self.mesh(v, color, additive=additive, screen=screen)

    def wedge(self, center, radius, heading, half_angle, color, segments=40,
              additive=False) -> None:
        a = heading + np.linspace(-half_angle, half_angle, segments + 1)
        rim = np.asarray(center, np.float32) + radius * np.stack([np.cos(a), np.sin(a)], 1)
        c = np.broadcast_to(np.asarray(center, np.float32), (segments, 2))
        tris = np.stack([c, rim[:-1], rim[1:]], 1).reshape(-1, 2)
        self.mesh(tris, color, additive=additive)

    def image(self, rgba_img: np.ndarray, bounds, additive=False, smooth=True,
              mode: str = "color") -> None:
        """Draw an (H, W, 4) float image stretched over world bounds (x0, y0, x1, y1).
        Row 0 is the bottom (y0) of the bounds.
        mode="mask": alpha is a 0/1 mask; edges are traced smoothly between cells and
        shaded like rock, so blocky grids look like organic shapes at any zoom."""
        self._add("image", img=np.ascontiguousarray(rgba_img, np.float32),
                  bounds=tuple(bounds), additive=additive, smooth=smooth or mode == "mask",
                  mode={"color": 0, "mask": 1}[mode])
