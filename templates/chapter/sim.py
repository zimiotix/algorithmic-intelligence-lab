"""{title}.

Explain the algorithm in two or three sentences here: what each agent senses, decides
and does. Keep the code close to the equations in advanced.md.
"""

from __future__ import annotations

import numpy as np

from ailab.core import (
    Experiment,
    HudItem,
    InputState,
    LiveEq,
    LiveValue,
    Overlay,
    Param,
    Preset,
    Simulation,
    Tool,
    section,
)
from ailab.core.params import fmt
from ailab.render import palette as pal


class Sim(Simulation):
    world = (0.0, 0.0, 160.0, 90.0)
    background = "void"          # "void" | "water" | "soil" | "grass"
    playback = 1.0               # speed the Lab starts at (0.25, 0.5, 1, 2, 4)
    PARAMS = [                   # every help text appears when the learner hovers
        *section(                # groups fold in the Controls panel; the first starts open
            "World",
            Param("n", "Agents", 200, 10, 2000, 10, "How many agents move around.", "N",
                  restart=True),
        ),
        *section(
            "Movement",
            Param("speed", "Speed", 10.0, 1.0, 30.0, 0.5, "How fast every agent moves.",
                  "v", "u/s"),
        ),
    ]
    PRESETS = [                  # one click; unmentioned parameters return to defaults
        Preset("default", "Default", {}, "The standard setting."),
        Preset("fast", "Fast", {"speed": 25.0}, "Everything at a run."),
    ]
    OVERLAYS = [
        Overlay("heading", "Headings", True, "Which way each agent is going."),
    ]
    TOOLS = [                    # the Lab hotbar (keys 1-9); read with self.tool_of(inp)
        Tool("inspect", "Inspect", "inspect", "Click an agent to explain it"),
    ]
    EXPERIMENTS = [              # `check` names a method returning True once achieved
        Experiment("watch", "Watch for ten seconds", "Just watch the agents move.",
                   "Replace me with the idea this experiment reveals.", check="exp_watch"),
    ]
    LIVE_MATH = [                # terms first: define every symbol in plain words
        LiveEq("move", "How the focus agent moves", r"\mathbf{x} \leftarrow \mathbf{x} + "
               r"\mathbf{v}\,\Delta t", "x: position · v: velocity · Δt: one step (1/60 s)"),
    ]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        rng = np.random.default_rng(seed)      # the ONLY source of randomness
        n = int(self.p.n)
        self.pos = rng.uniform((10, 10), (150, 80), (n, 2))
        a = rng.uniform(0, 2 * np.pi, n)
        self.vel = np.stack([np.cos(a), np.sin(a)], 1) * self.p.speed
        self.focus = 0

    def step(self, inp: InputState) -> None:
        for button, at in inp.clicks:
            if self.tool_of(inp) == "inspect" and button == "left":
                self.focus = int(np.argmin(np.sum((self.pos - np.asarray(at)) ** 2, axis=1)))
        self.pos += self.vel * self.dt
        # bounce off the walls
        for k, hi in ((0, 160.0), (1, 90.0)):
            out = (self.pos[:, k] < 0) | (self.pos[:, k] > hi)
            self.vel[out, k] *= -1
            self.pos[:, k] = np.clip(self.pos[:, k], 0, hi)

    def draw(self, s) -> None:
        s.background(self.background)
        ang = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        s.sprites("chevron", self.pos, ang, (1.6, 1.2), pal.rgba(pal.SKY))
        if self.show.heading:
            s.arrows(self.pos, self.vel * 0.2, 0.08, pal.rgba(pal.AMBER, 0.6))
        s.circles(self.pos[self.focus], 1.4, pal.rgba("#ffffff", 0.8), ring=0.1)

    def state_arrays(self):
        return [self.pos, self.vel]

    def hud(self) -> list[HudItem]:
        return [HudItem("agents", f"{len(self.pos)}")]

    def live_math(self) -> dict[str, LiveValue]:
        v = self.vel[self.focus]
        return {"move": LiveValue(f"v = ({fmt(v[0])}, {fmt(v[1])})", None,
                                  "moves in a straight line until it hits a wall")}

    def exp_watch(self) -> bool:
        return self.t > 10.0
