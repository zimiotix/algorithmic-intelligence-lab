"""The contract every chapter's simulation implements.

Determinism rules (checked by ``tests/test_chapter_contract.py`` for every chapter):
  * the simulation advances only in fixed steps of ``dt``;
  * all randomness comes from ``seed`` (numpy ``Generator`` or Warp's counter RNG);
  * GPU kernels never depend on thread execution order (double buffers, integer atomics);
  * the only outside influence is the ``InputState`` handed to ``step``.
Same seed + same inputs => same run, so every run can be replayed and tested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .params import Experiment, LiveEq, LiveValue, Overlay, Param, Preset, Tool, Values


@dataclass
class InputState:
    """What the learner is doing during one fixed step, in world coordinates."""

    mouse: tuple[float, float] | None = None       # None when the cursor is outside
    buttons: frozenset[str] = frozenset()          # held: "left" | "right" | "middle"
    keys: frozenset[str] = frozenset()             # held keys: "Left", "Up", "Shift", "C", ...
    clicks: tuple[tuple[str, tuple[float, float]], ...] = ()   # presses since last step
    key_presses: tuple[str, ...] = ()              # keys pressed since last step
    tool: str = ""                                 # key of the active Tool ("" = default)

    @property
    def mods(self) -> frozenset[str]:
        return self.keys & {"Shift", "Ctrl", "Alt"}


@dataclass
class SimContext:
    device: str = "cpu"          # "cuda:0" or "cpu"
    seed: int = 1

    @property
    def on_cpu(self) -> bool:
        return not self.device.startswith("cuda")


@dataclass
class HudItem:
    label: str
    value: str
    accent: bool = False


class Simulation:
    """Base class. Subclasses set the class attributes and override the methods."""

    world: tuple[float, float, float, float] = (0.0, 0.0, 160.0, 90.0)  # x0, y0, x1, y1
    dt: float = 1.0 / 60.0
    background: str = "void"
    playback: float = 1.0   # speed the Lab starts at (0.25, 0.5, 1, 2 or 4)
    # False when the default configuration uses no randomness at all (the seed only
    # matters once e.g. sensor noise is switched on).
    seeded: bool = True
    PARAMS: list[Param] = []
    OVERLAYS: list[Overlay] = []
    PRESETS: list[Preset] = []          # named parameter sets, one click in Controls
    TOOLS: list[Tool] = []              # hotbar; the first one is active at start
    EXPERIMENTS: list[Experiment] = []  # guided things to try, shown in the Guide
    LIVE_MATH: list[LiveEq] = []        # equations evaluated live for the focus agent

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.device = ctx.device
        self.p = Values(self.PARAMS, on_cpu=ctx.on_cpu)
        self.show = Values(self.OVERLAYS)
        self.seed = ctx.seed
        self.t = 0.0
        self.steps = 0
        self.focus: int = 0          # the agent the overlays explain
        self.reset(ctx.seed)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed: int) -> None:
        self.seed = seed
        self.t = 0.0
        self.steps = 0

    def step(self, inp: InputState) -> None:  # advance exactly self.dt
        raise NotImplementedError

    def draw(self, scene) -> None:
        raise NotImplementedError

    def advance(self, inp: InputState) -> None:
        self.step(inp)
        self.t += self.dt
        self.steps += 1

    # --------------------------------------------------------------------- info
    def hud(self) -> list[HudItem]:
        return []

    def live_math(self) -> dict[str, LiveValue]:
        """Numbers for each ``LIVE_MATH`` entry, for the focus agent, right now."""
        return {}

    def follow(self) -> tuple[float, float, float, float] | None:
        """(x, y, heading, zoom) to ride along with an agent (a chase camera that turns
        with it, so "up" on screen is its "forward"), or None for the fixed map view."""
        return None

    def tool_of(self, inp: InputState) -> str:
        """The active tool key (the first tool when the UI hasn't chosen one)."""
        return inp.tool or (self.TOOLS[0].key if self.TOOLS else "")

    def state_arrays(self) -> list[np.ndarray]:
        """Arrays that fully describe the state; used for determinism checks."""
        return []

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in self.state_arrays():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------- params
    def set_param(self, key: str, value) -> None:
        self.p.set(key, value)
        if self.p.spec(key).restart:
            self.reset(self.seed)
        else:
            self.on_param(key)

    def apply_values(self, values: dict) -> None:
        """Set every parameter at once: the given values, defaults for the rest.
        Restarts once if a restart-only parameter changed, else updates live."""
        changed = []
        for spec in self.PARAMS:
            v = spec.clamp(values.get(spec.key, self.p.default(spec.key)))
            if v != self.p.get(spec.key):
                self.p.set(spec.key, v)
                changed.append(spec)
        if any(c.restart for c in changed):
            self.reset(self.seed)
        else:
            for c in changed:
                self.on_param(c.key)

    def on_param(self, key: str) -> None:
        """Hook for live (non-restarting) parameter changes."""


@dataclass
class ScriptedInput:
    """Deterministic synthetic input for tests and headless snapshots."""

    kind: str = "none"                  # "none" | "circle"
    center: tuple[float, float] = (80.0, 45.0)
    radius: float = 25.0
    period: float = 4.0
    presses: dict[int, tuple[str, ...]] = field(default_factory=dict)
    tool: str = ""
    buttons: frozenset[str] = frozenset()     # held for the whole script (drag tests)
    click_every: int = 0                      # left-click at the cursor every N steps

    def at(self, step: int, dt: float) -> InputState:
        keys = self.presses.get(step, ())
        if self.kind == "circle":
            a = 2 * np.pi * step * dt / self.period
            m = (self.center[0] + self.radius * np.cos(a), self.center[1] + self.radius * np.sin(a))
            clicks = ((("left", m),) if self.click_every and step % self.click_every == 0
                      else ())
            return InputState(mouse=m, key_presses=keys, tool=self.tool, buttons=self.buttons,
                              clicks=clicks)
        return InputState(key_presses=keys, tool=self.tool)
