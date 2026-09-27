"""{title}.

Explain the algorithm in two or three sentences here: what each agent senses, decides
and does. Keep the code close to the equations in advanced.md.
"""

from __future__ import annotations

import numpy as np

from ailab.core import HudItem, InputState, Overlay, Param, Simulation
from ailab.render import palette as pal


class Sim(Simulation):
    world = (0.0, 0.0, 160.0, 90.0)
    background = "void"          # "void" | "water" | "soil" | "grass"
    PARAMS = [
        Param("n", "Agents", 200, 10, 2000, 10, "How many agents.", "N", restart=True),
        Param("speed", "Speed", 10.0, 1.0, 30.0, 0.5, "", "v", "u/s"),
    ]
    OVERLAYS = [
        Overlay("heading", "Headings", True, "Which way each agent is going."),
    ]

    def reset(self, seed: int) -> None:
        super().reset(seed)
        rng = np.random.default_rng(seed)      # the ONLY source of randomness
        n = int(self.p.n)
        self.pos = rng.uniform((10, 10), (150, 80), (n, 2))
        a = rng.uniform(0, 2 * np.pi, n)
        self.vel = np.stack([np.cos(a), np.sin(a)], 1) * self.p.speed

    def step(self, inp: InputState) -> None:
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

    def state_arrays(self):
        return [self.pos, self.vel]

    def hud(self) -> list[HudItem]:
        return [HudItem("agents", f"{len(self.pos)}")]
