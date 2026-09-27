"""Small reusable widgets: logo, chips, stat cards, chapter cards, parameter panel."""

from __future__ import annotations

import html
import math
import re

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.params import Param
from . import theme

GREEK = {"tau": "τ", "phi": "φ", "Phi": "Φ", "theta": "θ", "omega": "ω", "sigma": "σ",
         "delta": "δ", "alpha": "α", "dot": ""}


def symbol_html(tex: str) -> str:
    """Tiny TeX-ish -> HTML for parameter symbols: '\\tau_f' -> 'τ<sub>f</sub>'."""
    if not tex:
        return ""
    s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), m.group(1)), tex)
    s = re.sub(r"_\{([^}]*)\}|_(\w)", lambda m: f"<sub>{m.group(1) or m.group(2)}</sub>", s)
    return s.replace("{", "").replace("}", "")


def dot_pixmap(color: str, d: int = 10) -> QPixmap:
    pm = QPixmap(d, d)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, d, d)
    p.end()
    return pm


class Logo(QWidget):
    """A tiny school of chevrons turning together: the app's mark."""

    def __init__(self, size: int = 40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.width()
        g = QLinearGradient(0, 0, s, s)
        g.setColorAt(0, QColor("#0e7490"))
        g.setColorAt(1, QColor("#1e1b4b"))
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, s, s), s * 0.26, s * 0.26)
        p.fillPath(path, QBrush(g))
        pen = QPen(QColor("#e0f2fe"), max(1.5, s * 0.055))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for i in range(5):
            a = -0.5 + i * 0.25
            cx = s * (0.28 + 0.11 * i)
            cy = s * (0.70 - 0.10 * i - 0.02 * i * i)
            r = s * 0.10
            d = QPointF(math.cos(a), math.sin(a) * -1)
            n = QPointF(-d.y(), d.x())
            tip = QPointF(cx, cy) + d * r
            p.drawLine(tip, tip - d * r * 1.6 + n * r * 0.9)
            p.drawLine(tip, tip - d * r * 1.6 - n * r * 0.9)
        p.end()


def chip(text: str, color: str = theme.MUTED, filled: bool = False) -> QLabel:
    lbl = QLabel(text)
    bg = f"background:{color}22;" if filled else "background: transparent;"
    lbl.setStyleSheet(f"QLabel {{ {bg} color:{color}; border:1px solid {color}55;"
                      "border-radius:9px; padding:2px 9px; font-size:11px; font-weight:600; }")
    lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    return lbl


class StatCard(QFrame):
    def __init__(self, title: str, value: str, detail: str = "", color: str = theme.ACCENT):
        super().__init__()
        self.setObjectName("stat")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        t = QLabel(title.upper())
        t.setProperty("role", "section")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"font-size:15px; font-weight:650; color:{color};")
        self.value.setWordWrap(True)
        self.detail = QLabel(detail)
        self.detail.setProperty("role", "faint")
        self.detail.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(self.value)
        lay.addWidget(self.detail)

    def set(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class ChapterCard(QFrame):
    clicked = Signal(str)

    def __init__(self, info, track_color: str, runs_on: str):
        super().__init__()
        self.info = info
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(280)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 14)
        lay.setSpacing(8)
        self.thumb = QLabel()
        self.thumb.setFixedHeight(168)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {track_color}33,"
            f" stop:1 #0b1220); border-top-left-radius:12px; border-top-right-radius:12px;"
            f" color:{track_color}; font-size:11px;")
        self.thumb.setText("rendering preview…")
        lay.addWidget(self.thumb)
        body = QVBoxLayout()
        body.setContentsMargins(16, 4, 16, 0)
        body.setSpacing(6)
        title = QLabel(info.title)
        title.setStyleSheet("font-size:16px; font-weight:650;")
        title.setWordWrap(True)
        summary = QLabel(info.summary)
        summary.setWordWrap(True)
        summary.setProperty("role", "muted")
        foot = QHBoxLayout()
        foot.setSpacing(6)
        diff = QLabel(theme.dots(info.difficulty, color=track_color))
        diff.setToolTip(f"Difficulty {info.difficulty}/5")
        foot.addWidget(diff)
        foot.addStretch(1)
        gpu = info.compute == "gpu"
        c = chip(runs_on.split(" · ")[0] if gpu else "CPU", theme.GOOD if gpu else theme.MUTED)
        c.setToolTip(f"Runs on {runs_on}" if gpu else "Light chapter: runs on the CPU")
        foot.addWidget(c)
        if info.memory:
            m = chip("memory", "#a78bfa")
            m.setToolTip("Uses memory systems: " + ", ".join(info.memory))
            foot.addWidget(m)
        body.addWidget(title)
        body.addWidget(summary)
        body.addLayout(foot)
        lay.addLayout(body)

    def set_thumbnail(self, path: str) -> None:
        pm = QPixmap(path)
        if pm.isNull():
            return
        w = max(self.thumb.width(), 280)
        pm = pm.scaled(w, self.thumb.height(), Qt.KeepAspectRatioByExpanding,
                       Qt.SmoothTransformation)
        self.thumb.setPixmap(pm)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.info.id)


def param_tooltip(p: Param) -> str:
    """Hover card for a parameter: what it does, its symbol, range and default."""
    sym = symbol_html(p.symbol)
    head = f"<b>{html.escape(p.label)}</b>" + (f" &nbsp;<i>{sym}</i>" if sym else "")
    lines = [head, f"<span style='color:{theme.TEXT}'>{html.escape(p.help)}</span>"
             if p.help else ""]
    unit = f" {p.unit}" if p.unit and p.unit != "deg" else ("°" if p.unit == "deg" else "")
    if p.choices:
        lines.append(f"<span style='color:{theme.MUTED}'>Options: "
                     f"{html.escape(', '.join(p.choices))}</span>")
    elif p.kind != "bool" and p.lo is not None:
        lines.append(f"<span style='color:{theme.MUTED}'>Range {p.lo:g} – {p.hi:g}{unit}"
                     f" · default {p.default:g}{unit}</span>")
    if p.symbol:
        lines.append(f"<span style='color:{theme.FAINT}'>Written <i>{sym}</i> in the Deep dive"
                     " and Live Math.</span>")
    if p.restart:
        lines.append("<span style='color:#fbbf24'>⟲ Changing it restarts the run.</span>")
    return "<div style='max-width:320px'>" + "<br>".join(x for x in lines if x) + "</div>"


class ParamPanel(QWidget):
    """Sliders / toggles / dropdowns generated from a simulation's PARAMS."""

    changed = Signal(str, object)

    def __init__(self, params: list[Param], values, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self._widgets = {}
        for p in params:
            lay.addWidget(self._row(p, values.get(p.key)))
        lay.addStretch(1)

    def _row(self, p: Param, value) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(4)
        sym = symbol_html(p.symbol)
        name = QLabel(p.label + (f"  <span style='color:{theme.FAINT}'><i>{sym}</i></span>"
                                 if sym else "") + (" <span style='color:#fbbf24'>⟲</span>"
                                                    if p.restart else "")
                      + f" <span style='color:{theme.FAINT}'>ⓘ</span>")
        tip = param_tooltip(p)
        w.setToolTip(tip)            # hover anywhere on the row
        name.setToolTip(tip)
        g.addWidget(name, 0, 0)
        if p.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda v, k=p.key: self.changed.emit(k, v))
            cb.setToolTip(tip)
            g.addWidget(cb, 0, 1, Qt.AlignRight)
            self._widgets[p.key] = cb
        elif p.kind == "choice":
            combo = QComboBox()
            combo.addItems(list(p.choices))
            combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(lambda v, k=p.key: self.changed.emit(k, v))
            combo.setToolTip(tip)
            g.addWidget(combo, 1, 0, 1, 2)
            self._widgets[p.key] = combo
        else:
            unit = f" {p.unit}" if p.unit else ""
            val = QLabel()
            val.setStyleSheet(f"color:{theme.TEXT}; font-weight:600;")
            g.addWidget(val, 0, 1, Qt.AlignRight)
            s = QSlider(Qt.Horizontal)
            step = p.step or ((p.hi - p.lo) / 200.0)
            n = max(1, int(round((p.hi - p.lo) / step)))
            s.setRange(0, n)
            def to_val(i, p=p, step=step):
                return p.clamp(p.lo + i * step)

            def fmt(v, p=p, unit=unit):
                return (f"{int(v):,}" if p.kind == "int" else f"{v:.3g}") + unit

            s.setValue(int(round((float(value) - p.lo) / step)))
            val.setText(fmt(value))
            s.valueChanged.connect(lambda i, val=val, f=fmt, tv=to_val: val.setText(f(tv(i))))
            if p.restart:
                s.sliderReleased.connect(lambda s=s, k=p.key, tv=to_val:
                                         self.changed.emit(k, tv(s.value())))
                s.valueChanged.connect(lambda i, s=s, k=p.key, tv=to_val:
                                       None if s.isSliderDown() else self.changed.emit(k, tv(i)))
            else:
                s.valueChanged.connect(lambda i, k=p.key, tv=to_val: self.changed.emit(k, tv(i)))
            s.setToolTip(tip)
            g.addWidget(s, 1, 0, 1, 2)
            self._widgets[p.key] = s
        return w
