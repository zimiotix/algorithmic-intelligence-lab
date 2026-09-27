"""The live simulation view: a QOpenGLWidget that runs the fixed-step loop and draws
with the ModernGL renderer. Mouse/keyboard become an InputState for the simulation."""

from __future__ import annotations

import math
import time

import moderngl
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..core import compute
from ..core.clock import Clock
from ..core.params import Tool
from ..core.sim import InputState, Simulation
from ..render import palette as pal
from ..render.camera import Camera
from ..render.renderer import Renderer
from ..render.scene import Scene

APP_KEYS = {"Space", "R", ".", "Home"}
DIGITS = {str(i) for i in range(1, 10)}
_BUTTONS = {Qt.LeftButton: "left", Qt.RightButton: "right", Qt.MiddleButton: "middle"}
_MODS = {Qt.Key_Shift: "Shift", Qt.Key_Control: "Ctrl", Qt.Key_Alt: "Alt"}


def _key_name(event) -> str:
    k = event.key()
    if k in _MODS:
        return _MODS[k]
    return QKeySequence(k).toString() or event.text()


class Viewport(QOpenGLWidget):
    glReady = Signal(dict)          # renderer info, once
    stats = Signal(dict)            # per-frame timings
    hotkey = Signal(str)            # app-level keys: pause, step, reset, "tool:N",
                                    # "overlay:N", "focus", "escape"
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.sim: Simulation | None = None
        self.clock = Clock()
        self.camera = Camera()
        self.ctx: moderngl.Context | None = None
        self.renderer: Renderer | None = None
        self._last = time.perf_counter()
        self._mouse: QPointF | None = None
        self._buttons: set[str] = set()
        self._keys: set[str] = set()
        self._clicks: list = []
        self._presses: list[str] = []
        self._pan_from: QPointF | None = None
        self._fps = 60.0
        self.tool: Tool | None = None     # what the learner holds (set by the Lab)
        self._following = False           # chase camera engaged
        self.frameSwapped.connect(self.update)

    def set_tool(self, tool: Tool | None) -> None:
        self.tool = tool
        icon = tool.icon if tool else ""
        self.setCursor({"hunt": Qt.BlankCursor, "inspect": Qt.PointingHandCursor}.get(
            icon, Qt.CrossCursor if tool and tool.radius > 0 else Qt.ArrowCursor))

    # ------------------------------------------------------------- simulation
    def set_sim(self, sim: Simulation | None) -> None:
        self.sim = sim
        self.clock = Clock(sim.dt if sim else 1 / 60)
        if sim:
            self.camera.set_world(sim.world)
        self._last = time.perf_counter()
        self.update()

    # --------------------------------------------------------------------- GL
    def initializeGL(self) -> None:
        platform = QGuiApplication.platformName()
        order = ["egl", None] if platform in ("wayland", "eglfs") else [None, "egl"]
        err = None
        for backend in order:
            try:
                kw = {"backend": backend} if backend else {}
                self.ctx = moderngl.create_context(require=330, **kw)
                break
            except Exception as e:  # try the other GL binding
                err = e
        if self.ctx is None:
            self.failed.emit(f"OpenGL 3.3 context unavailable: {err}")
            return
        try:
            self.renderer = Renderer(self.ctx)
        except Exception as e:
            self.failed.emit(f"Shader compilation failed: {e}")
            self.renderer = None
            return
        self.glReady.emit(self.renderer.info)

    def paintGL(self) -> None:
        if self.renderer is None or self.ctx is None:
            return
        now = time.perf_counter()
        real_dt, self._last = now - self._last, now
        if real_dt > 0:
            self._fps += (1.0 / real_dt - self._fps) * 0.05
        w, h, dpr = self.width(), self.height(), self.devicePixelRatioF()
        compute_ms = 0.0
        steps = 0
        scene = Scene(px=1.0 / self.camera.ppu(w, h))
        scene.view = (float(w), float(h))
        if self.sim is not None:
            steps = self.clock.tick(real_dt)
            t0 = time.perf_counter()
            try:
                for k in range(steps):
                    self.sim.advance(self._input(first=k == 0))
                compute.synchronize()
            except Exception as e:  # keep the app alive; show the error instead
                self.failed.emit(f"{type(e).__name__}: {e}")
                self.sim = None
                return
            compute_ms = (time.perf_counter() - t0) * 1000
            self._chase(real_dt, w, h)
            scene.px = 1.0 / self.camera.ppu(w, h)
            scene.time = self.sim.t
            try:
                self.sim.draw(scene)
            except Exception as e:
                self.failed.emit(f"draw(): {type(e).__name__}: {e}")
                self.sim = None
                return
            self._draw_brush(scene)
        else:
            scene.background("void")
        t1 = time.perf_counter()
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        self.renderer.render(scene, fbo, w, h, dpr, self.camera, scene.time)
        self.stats.emit({"fps": self._fps, "compute_ms": compute_ms, "steps": steps,
                         "render_ms": (time.perf_counter() - t1) * 1000})

    def _chase(self, real_dt: float, w: float, h: float) -> None:
        """Ride along with the agent the simulation asks to follow (if any)."""
        try:
            target = self.sim.follow()
        except Exception:
            target = None
        if target is None:
            if self._following:
                self._following = False
                self.camera.reset()
            return
        x, y, heading, zoom = target
        snap = not self._following
        if snap:
            self.camera.zoom = zoom
            self._following = True
        lead = 0.22 * h / self.camera.ppu(w, h)        # agent sits in the lower part
        k = 1.0 if snap else 1.0 - math.exp(-max(real_dt, 0.0) / 0.12)
        self.camera.follow(x, y, heading, lead, k)

    def _draw_brush(self, scene: Scene) -> None:
        """A ring at the cursor showing the active tool's reach."""
        if self.tool is None or self.tool.radius <= 0 or self._mouse is None:
            return
        m = self.camera.to_world(self._mouse.x(), self._mouse.y(), self.width(), self.height())
        color = pal.CORAL if "right" in self._buttons else "#e2e8f0"
        scene.circles(m, self.tool.radius, pal.rgba(color, 0.08), additive=True)
        scene.circles(m, self.tool.radius, pal.rgba(color, 0.7), ring=scene.px * 1.5)

    def _input(self, first: bool) -> InputState:
        mouse = None
        if self._mouse is not None:
            mouse = self.camera.to_world(self._mouse.x(), self._mouse.y(), self.width(),
                                         self.height())
        inp = InputState(mouse=mouse, buttons=frozenset(self._buttons),
                         keys=frozenset(self._keys),
                         clicks=tuple(self._clicks) if first else (),
                         key_presses=tuple(self._presses) if first else (),
                         tool=self.tool.key if self.tool else "")
        if first:
            self._clicks.clear()
            self._presses.clear()
        return inp

    # ------------------------------------------------------------------ input
    def _world(self, pos: QPointF):
        return self.camera.to_world(pos.x(), pos.y(), self.width(), self.height())

    def mouseMoveEvent(self, e) -> None:
        if self._pan_from is not None:
            d = e.position() - self._pan_from
            self.camera.pan(d.x(), d.y(), self.width(), self.height())
            self._pan_from = e.position()
        self._mouse = e.position()

    def mousePressEvent(self, e) -> None:
        self.setFocus()
        b = _BUTTONS.get(e.button())
        if b == "middle":
            self._pan_from = e.position()
            return
        if b:
            self._buttons.add(b)
            self._clicks.append((b, self._world(e.position())))

    def mouseReleaseEvent(self, e) -> None:
        b = _BUTTONS.get(e.button())
        if b == "middle":
            self._pan_from = None
        self._buttons.discard(b)

    def mouseDoubleClickEvent(self, e) -> None:
        if e.button() == Qt.MiddleButton:
            self.camera.reset()
        else:
            self.mousePressEvent(e)

    def wheelEvent(self, e) -> None:
        steps = e.angleDelta().y() / 120.0
        p = e.position()
        self.camera.zoom_at(1.15 ** steps, p.x(), p.y(), self.width(), self.height())

    def leaveEvent(self, e) -> None:
        self._mouse = None
        self._buttons.clear()

    def enterEvent(self, e) -> None:
        self._mouse = self.mapFromGlobal(self.cursor().pos()).toPointF()

    def event(self, e) -> bool:
        # Tab would move keyboard focus; in the Lab it toggles focus mode instead.
        if e.type() == QEvent.KeyPress and e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            if not e.isAutoRepeat():
                self.hotkey.emit("focus")
            return True
        return super().event(e)

    def keyPressEvent(self, e) -> None:
        if e.isAutoRepeat():
            return
        if e.key() == Qt.Key_Escape:
            self.hotkey.emit("escape")
            return
        name = _key_name(e)
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        if name in DIGITS and not e.modifiers() & Qt.AltModifier:
            self.hotkey.emit(f"{'overlay' if ctrl else 'tool'}:{name}")
            return
        if name in APP_KEYS and not self._keys & {"Ctrl", "Alt"}:
            if name == "Home":
                self.camera.reset()
                self._following = False          # a chase camera re-snaps
            self.hotkey.emit(name)
            return
        self._keys.add(name)
        self._presses.append(name)

    def keyReleaseEvent(self, e) -> None:
        if not e.isAutoRepeat():
            self._keys.discard(_key_name(e))

    def focusOutEvent(self, e) -> None:
        self._keys.clear()
        self._buttons.clear()
