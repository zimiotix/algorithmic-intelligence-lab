"""Declarative simulation parameters.

A chapter lists its parameters once; the UI builds sliders/toggles from them, the
Advanced text refers to them by symbol, and tests read the same defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    help: str = ""
    symbol: str = ""          # math symbol used in the Advanced section, e.g. "r_r"
    unit: str = ""
    choices: tuple[str, ...] | None = None
    restart: bool = False     # changing it restarts the simulation
    cpu_default: Any = None   # smaller default when running on the CPU backend

    @property
    def kind(self) -> str:
        if self.choices:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, int):
            return "int"
        return "float"

    def clamp(self, value: Any) -> Any:
        if self.kind == "choice":
            return value if value in self.choices else self.default
        if self.kind == "bool":
            return bool(value)
        v = float(value)
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        return int(round(v)) if self.kind == "int" else v


@dataclass(frozen=True)
class Overlay:
    """A toggleable 'see inside the algorithm' visual layer."""

    key: str
    label: str
    default: bool = False
    help: str = ""


class Values:
    """Attribute access to current values: ``self.p.speed``."""

    def __init__(self, specs: list[Param] | list[Overlay], on_cpu: bool = False):
        object.__setattr__(self, "_specs", {s.key: s for s in specs})
        vals = {}
        for s in specs:
            d = s.default
            if on_cpu and getattr(s, "cpu_default", None) is not None:
                d = s.cpu_default
            vals[s.key] = d
        object.__setattr__(self, "_vals", vals)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._vals[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def set(self, key: str, value: Any) -> None:
        spec = self._specs[key]
        self._vals[key] = spec.clamp(value) if isinstance(spec, Param) else bool(value)

    def get(self, key: str) -> Any:
        return self._vals[key]

    def spec(self, key: str):
        return self._specs[key]

    def items(self):
        return self._vals.items()


# ----------------------------------------------------------------- the lab kit
@dataclass(frozen=True)
class Tool:
    """Something the learner holds in the Lab: shown on the hotbar, keys 1-9.

    The simulation reads the active tool from ``InputState.tool`` and decides what the
    mouse buttons do with it."""

    key: str
    label: str
    icon: str            # hotbar glyph: hunt, wall, food, inspect, cone, barrier, erase
    left: str            # what the left button does
    right: str = ""      # what the right button does ("" = nothing)
    radius: float = 0.0  # brush radius in world units, drawn as a ring at the cursor
    tip: str = ""        # a longer hint for the Guide panel


@dataclass(frozen=True)
class Experiment:
    """A guided thing to try. ``check`` names a Simulation method returning True once the
    learner has done it; empty = the learner ticks it by hand. ``learn`` is revealed when
    done: the idea the experiment demonstrates."""

    key: str
    title: str
    how: str
    learn: str = ""
    check: str = ""


@dataclass(frozen=True)
class LiveEq:
    """An equation of the algorithm, evaluated live for the focus agent.

    ``tex`` is typeset once; ``Simulation.live_math()`` supplies the numbers each frame.
    ``terms`` defines every symbol in plain words, so the equation never comes first."""

    key: str
    title: str
    tex: str
    terms: str = ""   # plain-words definitions shown before the equation: "v: speed now · ..."


@dataclass
class LiveValue:
    text: str                  # the equation with today's numbers plugged in
    holds: bool | None = None  # is the condition met right now? (None: not a condition)
    verdict: str = ""          # what that means, in plain words


def fmt(x: float | None, digits: int = 2) -> str:
    """Compact number for live maths: 12.30, 0.041, -3.20e+04, ∞."""
    if x is None:
        return "–"
    if not math.isfinite(x):
        return "∞" if x > 0 else ("−∞" if x < 0 else "–")
    a = abs(x)
    if a != 0 and (a >= 1e4 or a < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}".replace("-", "−")
