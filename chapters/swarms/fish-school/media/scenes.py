"""Manim (Community Edition) explainer for the Deep dive. Render with:
    uv run tools/render_media.py swarms.fish-school
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    ORANGE,
    RED,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    GrowArrow,
    MathTex,
    Scene,
    VGroup,
    Write,
)


class ThreeZones(Scene):
    """The repulsion / orientation / attraction zones and the forces they produce."""

    def construct(self):
        fish = Dot(color=BLUE).scale(1.4)
        zones = VGroup(Circle(0.8, color=RED), Circle(1.8, color=ORANGE),
                       Circle(3.0, color=GREEN))
        labels = VGroup(MathTex("r_r", color=RED).move_to(zones[0].get_top() + 0.25 * UP),
                        MathTex("r_o", color=ORANGE).move_to(zones[1].get_top() + 0.25 * UP),
                        MathTex("r_a", color=GREEN).move_to(zones[2].get_top() + 0.25 * UP))
        self.play(FadeIn(fish))
        self.play(*(Create(z) for z in zones), *(Write(t) for t in labels))
        near = Dot([0.5, -0.3, 0], color=RED)
        far = Dot([2.4, 0.9, 0], color=GREEN)
        self.play(FadeIn(near), FadeIn(far))
        self.play(GrowArrow(Arrow([0, 0, 0], [-0.9, 0.5, 0], color=RED, buff=0)),
                  GrowArrow(Arrow([0, 0, 0], [1.1, 0.4, 0], color=GREEN, buff=0)))
        eq = MathTex(r"\mathbf{F} = \mathbf{F}^{sep} + \mathbf{F}^{ali} + \mathbf{F}^{coh}"
                     r" + \mathbf{F}^{flee}").to_edge(DOWN)
        self.play(Write(eq))
        self.wait(2)
