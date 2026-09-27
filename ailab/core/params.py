"""Declarative simulation parameters.

A chapter lists its parameters once; the UI builds sliders/toggles from them, the
Advanced text refers to them by symbol, and tests read the same defaults.
"""

from __future__ import annotations

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
