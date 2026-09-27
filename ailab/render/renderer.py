"""ModernGL renderer: draws a Scene into any framebuffer.

Pipeline:  procedural background -> SDF primitives into a 4x MSAA HDR (fp16) target
           -> resolve -> 5-level bloom mip chain -> composite (highlight shoulder,
           vignette, dither) -> target.
The same code draws inside the Qt window and into headless EGL snapshots.
"""

from __future__ import annotations

import moderngl
import numpy as np

from . import shaders as S
from .camera import Camera
from .scene import Scene

_ALPHA = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
_ADD = (moderngl.SRC_ALPHA, moderngl.ONE)

_INSTANCED = {
    # kind: (format, attribute names)
    "lines": ("2f 2f 1f 4f/i", ("i_a", "i_b", "i_w", "i_color")),
    "circles": ("2f 1f 1f 1f 4f/i", ("i_c", "i_r", "i_ring", "i_soft", "i_color")),
    "sprites": ("2f 1f 2f 4f 1f/i", ("i_pos", "i_angle", "i_size", "i_color", "i_phase")),
}


def _set(prog: moderngl.Program, name: str, value) -> None:
    member = prog.get(name, None)
    if member is not None:
        member.value = value


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.bloom_strength = 0.75
        self.bloom_threshold = 0.85
        self.samples = min(4, ctx.max_samples) if ctx.max_samples >= 2 else 0
        h, n, sdf = S.HEADER, S.NOISE, S.SDF
        prog = self.ctx.program
        self.progs = {
            "lines": prog(vertex_shader=h + S.LINES_VS, fragment_shader=h + S.LINES_FS),
            "circles": prog(vertex_shader=h + S.CIRCLES_VS, fragment_shader=h + S.CIRCLES_FS),
            "sprites": prog(vertex_shader=h + sdf + S.SPRITES_VS,
                            fragment_shader=h + sdf + S.SPRITES_FS),
            "mesh": prog(vertex_shader=h + S.MESH_VS, fragment_shader=h + n + S.MESH_FS),
            "image": prog(vertex_shader=h + S.IMAGE_VS, fragment_shader=h + n + S.IMAGE_FS),
            "background": prog(vertex_shader=S.FULLSCREEN_VS,
                               fragment_shader=h + n + S.BACKGROUND_FS),
            "down": prog(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.BLOOM_DOWN_FS),
            "up": prog(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.BLOOM_UP_FS),
            "composite": prog(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.COMPOSITE_FS),
        }
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        self._fs_vao = {k: ctx.vertex_array(self.progs[k], [])
                        for k in ("background", "down", "up", "composite")}
        self._image_vao = ctx.vertex_array(self.progs["image"], [(self.quad, "2f", "in_corner")])
        self._pools: dict[tuple[str, int], list] = {}
        self._textures: dict[int, moderngl.Texture] = {}
        self._size: tuple[int, int] | None = None
        self._targets: list = []

    @property
    def info(self) -> dict:
        i = self.ctx.info
        return {"renderer": i.get("GL_RENDERER", ""), "vendor": i.get("GL_VENDOR", ""),
                "version": i.get("GL_VERSION", ""), "msaa": self.samples}

    # ------------------------------------------------------------ resources
    def _ensure_targets(self, size: tuple[int, int]) -> None:
        if self._size == size:
            return
        for obj in self._targets:
            obj.release()
        ctx = self.ctx
        self._targets = []

        def tex(sz):
            t = ctx.texture(sz, 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self._targets.append(t)
            return t

        def fbo(att):
            f = ctx.framebuffer(color_attachments=[att])
            self._targets.append(f)
            return f

        self.tex_scene = tex(size)
        self.fbo_scene = fbo(self.tex_scene)
        if self.samples:
            rb = ctx.renderbuffer(size, 4, samples=self.samples, dtype="f2")
            self._targets.append(rb)
            self.fbo_draw = fbo(rb)
        else:
            self.fbo_draw = self.fbo_scene
        self.mips = []
        w, h = size[0] // 2, size[1] // 2
        for _ in range(5):
            w, h = max(1, w), max(1, h)
            t = tex((w, h))
            self.mips.append((t, fbo(t)))
            w, h = w // 2, h // 2
        self._size = size

    def _instanced_vao(self, kind: str, slot: int, data: np.ndarray) -> moderngl.VertexArray:
        key = (kind, slot)
        entry = self._pools.get(key)
        if entry is None or entry[2] < data.nbytes:
            if entry:
                entry[1].release()
                entry[0].release()
            cap = max(4096, 1 << (int(data.nbytes) - 1).bit_length())
            buf = self.ctx.buffer(reserve=cap, dynamic=True)
            if kind == "mesh":
                vao = self.ctx.vertex_array(self.progs["mesh"],
                                            [(buf, "2f 4f", "in_pos", "in_color")])
            else:
                fmt, names = _INSTANCED[kind]
                vao = self.ctx.vertex_array(
                    self.progs[kind], [(self.quad, "2f", "in_corner"), (buf, fmt, *names)])
            entry = [buf, vao, cap]
            self._pools[key] = entry
        entry[0].write(data.tobytes())
        return entry[1]

    def _texture(self, slot: int, img: np.ndarray, smooth: bool) -> moderngl.Texture:
        h, w = img.shape[:2]
        t = self._textures.get(slot)
        if t is None or t.size != (w, h):
            if t:
                t.release()
            t = self.ctx.texture((w, h), 4, dtype="f4")
            t.repeat_x = t.repeat_y = False
            self._textures[slot] = t
        f = moderngl.LINEAR if smooth else moderngl.NEAREST
        t.filter = (f, f)
        t.write(img.tobytes())
        return t

    # --------------------------------------------------------------- render
    def render(self, scene: Scene, target: moderngl.Framebuffer, width: float, height: float,
               dpr: float, camera: Camera, time: float = 0.0) -> None:
        """width/height in logical pixels; the target is width*dpr x height*dpr."""
        ctx = self.ctx
        size = (max(1, int(round(width * dpr))), max(1, int(round(height * dpr))))
        self._ensure_targets(size)
        (sx, sy), (ox, oy) = camera.uniforms(width, height)
        world_u = {"u_scale": (sx, sy), "u_offset": (ox, oy),
                   "u_px": 1.0 / (camera.ppu(width, height) * dpr)}
        screen_u = {"u_scale": (2.0 / width, -2.0 / height), "u_offset": (-1.0, 1.0),
                    "u_px": 1.0 / dpr}

        self.fbo_draw.use()
        ctx.viewport = (0, 0, *size)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.enable(moderngl.BLEND)
        slots: dict[str, int] = {}
        for cmd in scene.commands:
            d = cmd.data
            kind = cmd.kind
            slot = slots.get(kind, 0)
            slots[kind] = slot + 1
            prog = self.progs[kind]
            for k, v in (screen_u if d.get("screen") else world_u).items():
                _set(prog, k, v)
            ctx.blend_func = _ADD if d.get("additive") else _ALPHA
            if kind == "background":
                _set(prog, "u_style", d["style"])
                _set(prog, "u_time", float(time))
                _set(prog, "u_world", camera.world)
                _set(prog, "u_tint", d["tint"])
                ctx.disable(moderngl.BLEND)
                self._fs_vao["background"].render(moderngl.TRIANGLES, vertices=3)
                ctx.enable(moderngl.BLEND)
            elif kind in _INSTANCED:
                inst = d["inst"]
                if kind == "sprites":
                    _set(prog, "u_kind", d["shape"])
                vao = self._instanced_vao(kind, slot, inst)
                vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(inst))
            elif kind == "mesh":
                _set(prog, "u_style", d["style"])
                vao = self._instanced_vao("mesh", slot, d["data"])
                vao.render(moderngl.TRIANGLES, vertices=len(d["data"]))
            elif kind == "image":
                tex = self._texture(slot, d["img"], d["smooth"])
                tex.use(0)
                _set(prog, "u_tex", 0)
                _set(prog, "u_bounds", d["bounds"])
                _set(prog, "u_mode", d["mode"])
                self._image_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        ctx.disable(moderngl.BLEND)
        if self.samples:
            ctx.copy_framebuffer(self.fbo_scene, self.fbo_draw)

        # Bloom: progressive downsample (first pass keeps only bright parts) ...
        down, up = self.progs["down"], self.progs["up"]
        src = self.tex_scene
        for i, (t, f) in enumerate(self.mips):
            f.use()
            ctx.viewport = (0, 0, *t.size)
            src.use(0)
            _set(down, "u_src", 0)
            _set(down, "u_texel", (1.0 / src.size[0], 1.0 / src.size[1]))
            _set(down, "u_prefilter", 1 if i == 0 else 0)
            _set(down, "u_threshold", self.bloom_threshold)
            self._fs_vao["down"].render(moderngl.TRIANGLES, vertices=3)
            src = t
        # ... then upsample with a tent filter, accumulating into each larger level.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        for i in range(len(self.mips) - 1, 0, -1):
            small, (big, big_fbo) = self.mips[i][0], self.mips[i - 1]
            big_fbo.use()
            ctx.viewport = (0, 0, *big.size)
            small.use(0)
            _set(up, "u_src", 0)
            _set(up, "u_texel", (1.0 / small.size[0], 1.0 / small.size[1]))
            self._fs_vao["up"].render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)

        comp = self.progs["composite"]
        target.use()
        ctx.viewport = (0, 0, *size)
        self.tex_scene.use(0)
        self.mips[0][0].use(1)
        _set(comp, "u_scene", 0)
        _set(comp, "u_bloom", 1)
        _set(comp, "u_bloom_k", self.bloom_strength)
        _set(comp, "u_res", (float(size[0]), float(size[1])))
        self._fs_vao["composite"].render(moderngl.TRIANGLES, vertices=3)
