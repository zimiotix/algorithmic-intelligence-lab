"""Start the native app: system check screen, then the main window."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from ..core import compute, settings
from ..core.catalog import discover
from ..core.system import SystemInfo
from ..text import latex
from . import theme


def _gl_format() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)          # vsync: smooth, and no wasted frames
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)


def _icon() -> QIcon:
    from .widgets import Logo

    icon = QIcon()
    for s in (32, 64, 128, 256):
        logo = Logo(s)
        pm = QPixmap(s, s)
        pm.fill(Qt.transparent)
        logo.render(pm)
        icon.addPixmap(pm)
    return icon


def run_app(info: SystemInfo, device_pref: str, chapter: str | None, boot: bool = True) -> int:
    _gl_format()
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Algorithmic Intelligence Lab")
    app.setDesktopFileName("ailab")
    theme.apply(app)
    app.setWindowIcon(_icon())

    from .boot import BootScreen, probe_gl
    from .main_window import MainWindow

    screen = BootScreen() if boot else None
    if screen:
        screen.show()
        screen.pump(0.15)
    step = screen.step if screen else (lambda *a, **k: None)

    step("Processor", f"{info.cpu_model} · {info.cpu_cores} cores / {info.cpu_threads} threads")
    step("Memory", f"{info.ram_total_mb / 1024:.1f} GB ({info.ram_available_mb / 1024:.1f} GB "
                   "free)")
    for i, g in enumerate(info.gpus):
        extra = f" · {g.vram_mb / 1024:.0f} GB · driver {g.driver}" if g.vram_mb else ""
        step(f"GPU {i}", f"{g.vendor} {g.name}{extra}", "ok" if g.cuda_capable else "info")
    if not info.gpus:
        step("GPU", "none detected", "warn")

    dev = compute.init(device_pref, info)
    if dev.startswith("cuda"):
        step("Compute", f"CUDA on {info.compute_name} ({info.cuda_driver_version}) via Warp "
                        f"{info.warp_version}")
    else:
        step("Compute", "CPU backend (Warp). Chapters use smaller defaults.", "warn")

    gl = probe_gl()
    info.gl_renderer, info.gl_vendor = gl.get("renderer", ""), gl.get("vendor", "")
    info.gl_version = gl.get("version", "")
    if gl:
        soft = "llvmpipe" in info.gl_renderer.lower()
        step("Rendering", f"{info.gl_renderer.split('/')[0]} · OpenGL "
                          f"{info.gl_version.split()[0]}", "warn" if soft else "ok")
    else:
        step("Rendering", "OpenGL 3.3 not available", "bad")

    catalog = discover()
    n = len(catalog.chapters)
    step("Curriculum", f"{n} chapter{'s' * (n != 1)} in {len(catalog.tracks)} tracks · "
                       f"{len(catalog.planned)} more tracks planned", "info")
    step("Equations", "TeX found: equations typeset by LaTeX" if latex.available()
         else "No TeX install: using the built-in math renderer", "ok")
    if screen:
        screen.note.setText("  ".join(info.advice()))
        screen.pump(1.1)
        if os.environ.get("AILAB_CAPTURE_BOOT"):
            screen.grab().save(os.environ["AILAB_CAPTURE_BOOT"])

    win = MainWindow(info, catalog)
    win.show_fitted()
    if screen:
        screen.close()
    if chapter and chapter in catalog.chapters:
        win.open_chapter(chapter)
    settings.save({"device": device_pref})
    _dev_capture(app, win)
    return app.exec()


def _dev_capture(app, win) -> None:
    """AILAB_CAPTURE=path.png[,seconds[,tab[,section|focus]]] grabs the window and quits
    (docs, CI, debugging)."""
    spec = os.environ.get("AILAB_CAPTURE")
    if not spec:
        return
    from PySide6.QtCore import QTimer

    parts = spec.split(",")
    path, secs = parts[0], float(parts[1]) if len(parts) > 1 else 6.0
    if len(parts) > 2:
        QTimer.singleShot(500, lambda: win.chapter.tabs.setCurrentIndex(int(parts[2])))
    if len(parts) > 3 and parts[3] == "focus":
        QTimer.singleShot(1500, lambda: win.chapter.set_focus_mode(True))
    elif len(parts) > 3:
        QTimer.singleShot(1500, lambda: win.chapter._show_section(int(parts[3])))

    def grab():
        win.grab().save(path)
        app.quit()

    QTimer.singleShot(int(secs * 1000), grab)
