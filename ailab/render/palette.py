"""Colours. Values above 1.0 are HDR: they glow through the bloom pass."""

from __future__ import annotations

import numpy as np

# One consistent visual language across chapters.
INK = "#e6edf3"
MUTED = "#8b98a9"
SKY = "#38bdf8"
TEAL = "#2dd4bf"
MINT = "#34d399"
AMBER = "#fbbf24"
ORANGE = "#fb923c"
CORAL = "#f87171"
ROSE = "#fb7185"
VIOLET = "#a78bfa"
GOLD = "#facc15"

# Meaning-coded colours used by overlays in every chapter.
REPULSE = CORAL      # things pushing apart / danger
ALIGN = AMBER        # matching / orientation
ATTRACT = MINT       # pulling together / goals
SENSE = SKY          # perception
MEMORY = VIOLET      # anything remembered


def rgba(color, a: float = 1.0, glow: float = 1.0) -> np.ndarray:
    """'#rrggbb' or (r,g,b[,a]) -> float32 RGBA. `glow` > 1 makes it bloom."""
    if isinstance(color, str):
        c = color.lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        rgb = np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
        alpha = a
    else:
        c = np.asarray(color, np.float32)
        rgb, alpha = c[:3], (c[3] * a if len(c) > 3 else a)
    return np.array([*(rgb * glow), alpha], np.float32)


def lerp_colors(c0, c1, t) -> np.ndarray:
    """Per-element blend between two colours; t is an array in [0, 1]."""
    t = np.clip(np.asarray(t, np.float32), 0, 1)[..., None]
    return (rgba(c0) * (1 - t) + rgba(c1) * t).astype(np.float32)


def ramp(stops: list[tuple[float, str]], t) -> np.ndarray:
    """Piecewise-linear colour ramp; returns (..., 4)."""
    t = np.clip(np.asarray(t, np.float32), 0, 1)
    xs = np.array([s[0] for s in stops], np.float32)
    cols = np.stack([rgba(s[1]) for s in stops])
    out = np.empty(t.shape + (4,), np.float32)
    for ch in range(4):
        out[..., ch] = np.interp(t, xs, cols[:, ch])
    return out
