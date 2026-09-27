"""Headless rendering: run a chapter for N steps and save a PNG (no window needed).

Used by CI-style checks, for chapter thumbnails, and to eyeball visuals quickly:
    ailab --chapter swarms.fish-school --snapshot out.png --frames 300
"""

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import numpy as np

from ..core.catalog import ChapterInfo, load_sim_class
from ..core.sim import ScriptedInput, SimContext
from ..render.camera import Camera
from ..render.scene import Scene


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer (RGB8), so snapshots need no image library."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def snapshot(info: ChapterInfo, out: str, device: str, frames: int = 240,
             size: tuple[int, int] = (1600, 900), seed: int = 1, mouse: str = "auto",
             overlays: str = "default", zoom: float = 1.0, params: dict | None = None,
             center: str | tuple | None = None) -> dict:
    import moderngl

    from ..render.renderer import Renderer

    sim_cls = load_sim_class(info)
    sim = sim_cls(SimContext(device=device, seed=seed))
    for k, v in (params or {}).items():
        sim.set_param(k, v)
    if overlays == "all":
        for o in sim.OVERLAYS:
            sim.show.set(o.key, True)
    elif overlays not in ("default", ""):
        for o in sim.OVERLAYS:
            sim.show.set(o.key, o.key in overlays.split(","))
    x0, y0, x1, y1 = sim.world
    if mouse == "auto":
        mouse = "circle" if info.id.endswith("fish-school") else "none"
    script = ScriptedInput(kind=mouse, center=((x0 + x1) / 2, (y0 + y1) / 2),
                           radius=0.25 * (x1 - x0), period=5.0)
    t0 = time.perf_counter()
    for k in range(frames):
        sim.advance(script.at(k, sim.dt))
    sim_ms = (time.perf_counter() - t0) * 1000 / max(frames, 1)

    ctx = moderngl.create_standalone_context(require=330, backend="egl")
    renderer = Renderer(ctx)
    cam = Camera(sim.world)
    cam.zoom = zoom
    if center == "focus" and hasattr(sim, "_fetch"):
        cam.center = tuple(float(v) for v in sim._fetch()["pos"][sim.focus][:2])
    elif center is not None and center != "focus":
        cam.center = tuple(center)
    w, h = size
    scene = Scene(px=1.0 / cam.ppu(w, h), time=sim.t)
    scene.view = (w, h)
    sim.draw(scene)
    target = ctx.simple_framebuffer((w, h), components=4)
    t1 = time.perf_counter()
    renderer.render(scene, target, w, h, 1.0, cam, sim.t)
    img = np.frombuffer(target.read(components=3), np.uint8).reshape(h, w, 3)[::-1]
    ctx.finish()
    render_ms = (time.perf_counter() - t1) * 1000
    gl = renderer.info["renderer"]
    ctx.release()
    write_png(out, np.ascontiguousarray(img))
    return {"sim_ms_per_step": sim_ms, "render_ms": render_ms, "digest": sim.digest(),
            "gl": gl}
