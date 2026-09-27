"""Launch-time system check: probes the machine and shows the learner what it found."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import Logo


def probe_gl() -> dict:
    """Ask Qt's own OpenGL stack which GPU/driver will draw the UI."""
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

    surf = QOffscreenSurface()
    surf.setFormat(QSurfaceFormat.defaultFormat())
    surf.create()
    ctx = QOpenGLContext()
    ctx.setFormat(QSurfaceFormat.defaultFormat())
    if not ctx.create() or not ctx.makeCurrent(surf):
        return {}
    f = ctx.functions()

    def get(enum: int) -> str:
        v = f.glGetString(enum)
        return v.decode() if isinstance(v, bytes) else str(v or "")

    out = {"vendor": get(0x1F00), "renderer": get(0x1F01), "version": get(0x1F02)}
    ctx.doneCurrent()
    return out


class BootScreen(QWidget):
    def __init__(self):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(640, 430)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setStyleSheet(f"QFrame#boot {{ background:{theme.BG1}; border:1px solid "
                           f"{theme.LINE}; border-radius:18px; }}")
        card.setObjectName("boot")
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(30, 26, 30, 22)
        lay.setSpacing(14)
        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(Logo(46))
        tl = QVBoxLayout()
        tl.setSpacing(0)
        t = QLabel("Algorithmic Intelligence Lab")
        t.setStyleSheet("font-size:21px; font-weight:700;")
        sub = QLabel("System check: finding out what this computer can do")
        sub.setProperty("role", "muted")
        tl.addWidget(t)
        tl.addWidget(sub)
        head.addLayout(tl, 1)
        lay.addLayout(head)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(7)
        self.grid.setColumnStretch(2, 1)
        lay.addLayout(self.grid)
        lay.addStretch(1)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("role", "muted")
        lay.addWidget(self.note)
        self._row = 0

    def step(self, label: str, value: str, state: str = "ok") -> None:
        color = {"ok": theme.GOOD, "warn": theme.WARN, "bad": theme.BAD,
                 "info": theme.ACCENT}[state]
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{color}; font-size:11px;")
        k = QLabel(label)
        k.setProperty("role", "muted")
        k.setMinimumWidth(96)
        v = QLabel(value)
        v.setWordWrap(True)
        self.grid.addWidget(dot, self._row, 0, Qt.AlignTop)
        self.grid.addWidget(k, self._row, 1, Qt.AlignTop)
        self.grid.addWidget(v, self._row, 2)
        self._row += 1
        self.pump(0.07)

    def pump(self, seconds: float = 0.0) -> None:
        end = time.perf_counter() + seconds
        while True:
            QApplication.processEvents()
            if time.perf_counter() >= end:
                break
            time.sleep(0.01)
