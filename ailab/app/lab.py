"""The Lab's side furniture: tool hotbar, Guide panel (tool, live math, goals, keys, model
vs. nature, each folding away), live-math cards and toasts. Everything is generic: chapters only declare
TOOLS, EXPERIMENTS and LIVE_MATH, and implement ``live_math()``."""

from __future__ import annotations

import html

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.params import Experiment, LiveEq, LiveValue, Swatch, Tool
from ..text import latex
from . import theme
from .markdown_view import rasterize_svg
from .responsive import FitWidthScroll, Fold


# ------------------------------------------------------------------- icons
def tool_icon(kind: str, color: str = "#e6edf3", size: int = 26, dpr: float = 2.0) -> QIcon:
    """Small vector icons painted in code: crisp at any DPI, identical on every OS."""
    pm = QPixmap(round(size * dpr), round(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    pen = QPen(c, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    s = size
    if kind == "hunt":              # a predator's fin + crosshair
        p.drawEllipse(QPointF(s / 2, s / 2), s * 0.34, s * 0.34)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            p.drawLine(QPointF(s / 2 + dx * s * 0.22, s / 2 + dy * s * 0.22),
                       QPointF(s / 2 + dx * s * 0.46, s / 2 + dy * s * 0.46))
        p.setBrush(c)
        p.drawEllipse(QPointF(s / 2, s / 2), s * 0.07, s * 0.07)
    elif kind == "wall":            # stacked stones
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 60))
        for x, y, w in ((0.14, 0.58, 0.34), (0.52, 0.58, 0.34), (0.30, 0.30, 0.40)):
            p.drawRoundedRect(QRectF(s * x, s * y, s * w, s * 0.24), 3, 3)
    elif kind == "food":            # a cluster of pellets
        p.setBrush(c)
        for x, y, r in ((0.35, 0.40, 0.11), (0.62, 0.36, 0.09), (0.50, 0.64, 0.12),
                        (0.28, 0.66, 0.07), (0.72, 0.62, 0.07)):
            p.drawEllipse(QPointF(s * x, s * y), s * r, s * r)
    elif kind == "inspect":         # magnifier
        p.drawEllipse(QPointF(s * 0.42, s * 0.42), s * 0.22, s * 0.22)
        p.setPen(QPen(c, 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.58, s * 0.58), QPointF(s * 0.82, s * 0.82))
    elif kind == "cone":            # traffic cone
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.14)
        path.lineTo(s * 0.74, s * 0.78)
        path.lineTo(s * 0.26, s * 0.78)
        path.closeSubpath()
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 70))
        p.drawPath(path)
        p.drawLine(QPointF(s * 0.18, s * 0.84), QPointF(s * 0.82, s * 0.84))
        p.drawLine(QPointF(s * 0.37, s * 0.5), QPointF(s * 0.63, s * 0.5))
    elif kind == "barrier":         # a row of cones
        for x in (0.22, 0.5, 0.78):
            path = QPainterPath()
            path.moveTo(s * x, s * 0.3)
            path.lineTo(s * (x + 0.12), s * 0.72)
            path.lineTo(s * (x - 0.12), s * 0.72)
            path.closeSubpath()
            p.drawPath(path)
        p.drawLine(QPointF(s * 0.08, s * 0.8), QPointF(s * 0.92, s * 0.8))
    elif kind == "erase":
        p.save()
        p.translate(s / 2, s / 2)
        p.rotate(-35)
        p.drawRoundedRect(QRectF(-s * 0.3, -s * 0.15, s * 0.6, s * 0.3), 3, 3)
        p.drawLine(QPointF(-s * 0.05, -s * 0.15), QPointF(-s * 0.05, s * 0.15))
        p.restore()
    else:                           # a dot: unknown tool kinds still get something
        p.setBrush(c)
        p.drawEllipse(QPointF(s / 2, s / 2), s * 0.18, s * 0.18)
    p.end()
    return QIcon(pm)


# ------------------------------------------------------------------ hotbar
class Hotbar(QFrame):
    """The floating tool bar at the bottom of the view (keys 1-9)."""

    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hotbar")
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(8, 6, 8, 6)
        self._lay.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QToolButton] = {}

    def set_tools(self, tools: list[Tool]) -> None:
        for b in self.buttons.values():
            self.group.removeButton(b)
            b.deleteLater()
        self.buttons.clear()
        for i, t in enumerate(tools):
            b = QToolButton()
            b.setProperty("role", "tool")
            b.setCheckable(True)
            b.setIcon(tool_icon(t.icon, "#e6edf3", dpr=self.devicePixelRatioF() or 1.0))
            b.setIconSize(QSize(24, 24))
            b.setText(f"{t.label}  {i + 1}")
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(f"<b>{html.escape(t.label)}</b> (key {i + 1})<br>Left: "
                         f"{html.escape(t.left)}" + (f"<br>Right: {html.escape(t.right)}"
                                                     if t.right else ""))
            b.clicked.connect(lambda _=False, k=t.key: self.chosen.emit(k))
            self.group.addButton(b)
            self._lay.addWidget(b)
            self.buttons[t.key] = b
        self.setVisible(len(tools) > 1)
        self.adjustSize()

    def set_compact(self, compact: bool) -> None:
        """Icons only when the view is narrow (the labels live in the tooltips)."""
        style = Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextUnderIcon
        for b in self.buttons.values():
            b.setToolButtonStyle(style)

    def set_active(self, key: str) -> None:
        b = self.buttons.get(key)
        if b:
            b.setChecked(True)


# --------------------------------------------------------------- live math
def terms_html(terms: str) -> str:
    """ "v: speed now · a_b: braking" -> symbols in bold, meanings muted."""
    parts = []
    for part in terms.split("·"):
        sym, sep, meaning = part.partition(":")
        if sep:
            parts.append(f"<b style='color:#e2e8f0'>{html.escape(sym.strip())}</b> "
                         f"{html.escape(meaning.strip())}")
        elif part.strip():
            parts.append(html.escape(part.strip()))
    return " &nbsp;·&nbsp; ".join(parts)


class _EqImage(QWidget):
    """A typeset equation that shrinks to fit its width (like max-width: 100%)."""

    def __init__(self, img, size):
        super().__init__()
        self.img, self.natural = img, size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(40)

    def _scale(self, width: int) -> float:
        return min(1.0, width / max(self.natural.width(), 1))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(1, round(self.natural.height() * self._scale(width)))

    def sizeHint(self) -> QSize:
        return self.natural

    def paintEvent(self, e) -> None:
        k = self._scale(self.width())
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, self.natural.width() * k, self.natural.height() * k), self.img)
        p.end()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        h = self.heightForWidth(self.width())
        if self.height() != h:
            self.setFixedHeight(h)


class _MathCard(QFrame):
    def __init__(self, spec: LiveEq, img, size):
        super().__init__()
        self.setObjectName("mathcard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        t = QLabel(spec.title)
        t.setWordWrap(True)
        t.setStyleSheet("font-weight:650; font-size:12px; color:#cbd5e1;")
        self.badge = QLabel()
        self.badge.hide()
        top.addWidget(t, 1)
        top.addWidget(self.badge)
        lay.addLayout(top)
        if spec.terms:
            terms = QLabel(terms_html(spec.terms))
            terms.setWordWrap(True)
            terms.setStyleSheet(f"color:{theme.MUTED}; font-size:11px;")
            lay.addWidget(terms)
        if img is not None:
            eq = _EqImage(img, size)
        else:
            eq = QLabel()
            eq.setText(f"<code>{html.escape(spec.tex)}</code>")
            eq.setWordWrap(True)
        lay.addWidget(eq)
        self.values = QLabel()
        self.values.setWordWrap(True)
        self.values.setStyleSheet(f"font-family:'{theme.mono_family()}'; font-size:12px;"
                                  "color:#e2e8f0;")
        self.values.setTextFormat(Qt.PlainText)
        self.values.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.values)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setProperty("role", "muted")
        self.verdict.setStyleSheet("font-size:12px;")
        self.verdict.setTextFormat(Qt.PlainText)
        lay.addWidget(self.verdict)
        self._last = None

    def show_value(self, v: LiveValue | None) -> None:
        key = None if v is None else (v.text, v.holds, v.verdict)
        if key == self._last:
            return
        self._last = key
        if v is None:
            self.values.setText("–")
            self.verdict.hide()
            self.badge.hide()
            return
        self.values.setText(v.text)
        self.verdict.setText(v.verdict)
        self.verdict.setVisible(bool(v.verdict))
        if v.holds is None:
            self.badge.hide()
        else:
            ok = bool(v.holds)
            fg, bg, word = (("#34d399", "#0b2a22", "✓ holds") if ok
                            else ("#fbbf24", "#2a2008", "✗ not met"))
            self.badge.setText(word)
            self.badge.setStyleSheet(f"color:{fg}; background:{bg}; border-radius:8px;"
                                     "padding:2px 8px; font-size:11px; font-weight:700;")
            self.badge.show()


class LiveMathPanel(QWidget):
    """The algorithm's equations, with the focus agent's numbers plugged in, live."""

    MAX_W = 290    # widest an equation is drawn; narrower panels scale it down

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        self.cards: dict[str, _MathCard] = {}

    def set_specs(self, specs: list[LiveEq]) -> None:
        while self._lay.count():
            w = self._lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.cards.clear()
        if not specs:
            return
        eqs = [latex.Equation(s.tex, True) for s in specs]
        paths = latex.render(eqs)
        dpr = self.devicePixelRatioF() or 1.0
        for spec, e in zip(specs, eqs, strict=True):
            img = size = None
            p = paths.get(e.key)
            if p:
                img, size = rasterize_svg(str(p), 1.5, dpr)
                if img is not None and size.width() > self.MAX_W:
                    img, size = rasterize_svg(str(p), 1.5 * self.MAX_W / size.width(), dpr)
            card = _MathCard(spec, img, size)
            self._lay.addWidget(card)
            self.cards[spec.key] = card

    def update_values(self, values: dict[str, LiveValue]) -> None:
        for k, card in self.cards.items():
            card.show_value(values.get(k))


# -------------------------------------------------------------------- guide
def _section(text: str) -> QLabel:
    h = QLabel(text)
    h.setProperty("role", "section")
    return h


class _ExperimentRow(QFrame):
    toggled = Signal(str, bool)

    def __init__(self, exp: Experiment, done: bool):
        super().__init__()
        self.exp = exp
        self.setObjectName("exp")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        self.mark = QLabel()
        self.mark.setFixedWidth(20)
        self.mark.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        lay.addWidget(self.mark)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.title = QLabel(exp.title)
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight:650;")
        self.how = QLabel(exp.how)
        self.how.setWordWrap(True)
        self.how.setProperty("role", "muted")
        self.how.setStyleSheet("font-size:12px;")
        self.learn = QLabel(f"<span style='color:#5eead4'>Discovery:</span> {html.escape(exp.learn)}")
        self.learn.setWordWrap(True)
        self.learn.setStyleSheet("font-size:12px; color:#ccfbf1;")
        col.addWidget(self.title)
        col.addWidget(self.how)
        col.addWidget(self.learn)
        lay.addLayout(col, 1)
        self.set_done(done)

    def set_done(self, done: bool) -> None:
        self.done = done
        self.setProperty("done", "true" if done else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.mark.setText("<span style='color:#34d399; font-size:15px'>✓</span>" if done else
                          "<span style='color:#475569; font-size:15px'>○</span>")
        self.learn.setVisible(done and bool(self.exp.learn))
        auto = " It ticks itself when you manage it." if self.exp.check else \
            " Click to tick it off."
        self.setToolTip(self.exp.how + ("" if done else auto))

    def mousePressEvent(self, e) -> None:
        if not self.exp.check or self.done:      # auto ones can still be un-ticked
            self.toggled.emit(self.exp.key, not self.done)


class GuidePanel(QFrame):
    """Left side of the Lab: what you hold, the maths live, goals to try, the keys and
    what the model simplifies. Everything but the tool folds away; goals and notes start
    closed so the panel stays calm."""

    experimentToggled = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("guide")
        self.setMinimumWidth(210)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FitWidthScroll()
        inner = QWidget()
        inner.setStyleSheet(f"background:{theme.BG1};")
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(14, 12, 14, 18)
        self.lay.setSpacing(6)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # tool card
        self.tool_fold = Fold("IN YOUR HAND", "guide:tool", True)
        self.tool_card = QFrame()
        self.tool_card.setObjectName("toolcard")
        tl = QVBoxLayout(self.tool_card)
        tl.setContentsMargins(12, 10, 12, 12)
        tl.setSpacing(6)
        head = QHBoxLayout()
        self.tool_icon = QLabel()
        self.tool_icon.setFixedSize(28, 28)
        self.tool_name = QLabel()
        self.tool_name.setStyleSheet("font-size:15px; font-weight:700;")
        head.addWidget(self.tool_icon)
        head.addWidget(self.tool_name, 1)
        tl.addLayout(head)
        self.tool_text = QLabel()
        self.tool_text.setWordWrap(True)
        self.tool_text.setStyleSheet("font-size:12px;")
        tl.addWidget(self.tool_text)
        self.tool_fold.body.addWidget(self.tool_card)
        self.lay.addWidget(self.tool_fold)

        # colour key
        self.legend = Fold("COLOURS", "guide:legend", True,
                           tip="What each colour on screen means")
        self.legend_box = QVBoxLayout()
        self.legend_box.setSpacing(3)
        self.legend.body.addLayout(self.legend_box)
        self.lay.addWidget(self.legend)

        # live maths
        self.live_fold = Fold("LIVE MATH", "guide:live", True,
                              tip="The rules, with the focus agent's numbers plugged in")
        note = QLabel("The rules, with the focus agent's numbers plugged in right now.")
        note.setProperty("role", "faint")
        note.setWordWrap(True)
        self.live_fold.body.addWidget(note)
        self.live = LiveMathPanel()
        self.live_fold.body.addWidget(self.live)
        self.lay.addWidget(self.live_fold)

        # goals (optional: closed by default; while closed they tick silently)
        self.goals = Fold("GOALS", "guide:goals", False,
                          tip="Optional things to try. They tick themselves off; while this "
                              "is closed they do it quietly.")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.goals.body.addWidget(self.progress)
        self.exp_box = QVBoxLayout()
        self.exp_box.setSpacing(2)
        self.goals.body.addLayout(self.exp_box)
        self.rows: dict[str, _ExperimentRow] = {}
        self.lay.addWidget(self.goals)

        # keys
        self.keys = Fold("KEYS", "guide:keys", False)
        self.controls = QLabel()
        self.controls.setWordWrap(True)
        self.controls.setTextFormat(Qt.RichText)
        self.keys.body.addWidget(self.controls)
        self.lay.addWidget(self.keys)

        # model vs nature
        self.nature_fold = Fold("MODEL VS. NATURE", "guide:nature", False,
                                tip="What this model simplifies about the real thing")
        self.nature = QFrame()
        self.nature.setObjectName("nature")
        nl = QVBoxLayout(self.nature)
        nl.setContentsMargins(12, 10, 12, 12)
        self.nature_text = QLabel()
        self.nature_text.setWordWrap(True)
        self.nature_text.setStyleSheet("color:#c5cfb5; font-size:12px;")
        nl.addWidget(self.nature_text)
        self.nature_fold.body.addWidget(self.nature)
        self.lay.addWidget(self.nature_fold)
        self.lay.addStretch(1)

    @property
    def live_visible(self) -> bool:
        return self.isVisible() and self.live_fold.opened and bool(self.live.cards)

    def set_legend(self, items: list[Swatch]) -> None:
        while self.legend_box.count():
            w = self.legend_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for it in items:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 1, 0, 1)
            h.setSpacing(8)
            chip = QLabel()
            chip.setFixedSize(14, 14)
            c = it.color
            chip.setStyleSheet({
                "glow": f"background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, "
                        f"fy:0.5, stop:0 {c}, stop:0.55 {c}, stop:1 transparent); "
                        f"border-radius:7px;",
                "ring": f"background: transparent; border: 2px solid {c}; border-radius:7px;",
            }.get(it.shape, f"background:{c}; border-radius:7px;"))
            text = QLabel(it.label)
            text.setWordWrap(True)
            text.setStyleSheet("font-size:12px;")
            h.addWidget(chip, 0, Qt.AlignTop)
            h.addWidget(text, 1)
            self.legend_box.addWidget(row)
        self.legend.setVisible(bool(items))

    def set_live(self, specs: list[LiveEq]) -> None:
        self.live.set_specs(specs)
        self.live_fold.setVisible(bool(specs))

    # -------------------------------------------------------------- content
    def set_chapter(self, controls: list, reality: str, experiments: list[Experiment],
                    done: set[str]) -> None:
        while self.exp_box.count():
            w = self.exp_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.rows.clear()
        for e in experiments:
            r = _ExperimentRow(e, e.key in done)
            r.toggled.connect(self.experimentToggled)
            self.exp_box.addWidget(r)
            self.rows[e.key] = r
        self._update_progress()
        rows = "".join(
            f"<tr><td style='padding:3px 10px 3px 0; color:#e2e8f0; white-space:nowrap'>"
            f"<span style='background:{theme.BG3}; border-radius:4px'>&nbsp;{html.escape(k)}"
            f"&nbsp;</span></td><td style='padding:3px 0; color:{theme.MUTED}'>"
            f"{html.escape(a)}</td></tr>" for k, a in controls)
        self.controls.setText(f"<table style='font-size:12px'>{rows}</table>")
        self.nature_text.setText(reality or "This chapter is a simplified model of the real "
                                 "thing: it shows one idea clearly, not every detail.")

    def set_tool(self, tool: Tool | None) -> None:
        self.tool_card.setVisible(tool is not None)
        if tool is None:
            return
        self.tool_icon.setPixmap(tool_icon(tool.icon, theme.ACCENT, 28,
                                           self.devicePixelRatioF() or 1.0).pixmap(28, 28))
        self.tool_name.setText(tool.label)
        parts = [f"<span style='color:{theme.MUTED}'>Left</span>&nbsp; {html.escape(tool.left)}"]
        if tool.right:
            parts.append(f"<span style='color:{theme.MUTED}'>Right</span>&nbsp; "
                         f"{html.escape(tool.right)}")
        if tool.tip:
            parts.append(f"<span style='color:#94a3b8'><i>{html.escape(tool.tip)}</i></span>")
        self.tool_text.setText("<br>".join(parts))

    def set_done(self, key: str, done: bool) -> None:
        r = self.rows.get(key)
        if r:
            r.set_done(done)
        self._update_progress()

    def _update_progress(self) -> None:
        n = len(self.rows)
        k = sum(r.done for r in self.rows.values())
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(k)
        self.goals.set_badge(f"{k} / {n}" if n else "")
        self.goals.setVisible(n > 0)


# -------------------------------------------------------------------- toast
class Toast(QLabel):
    """A short message over the view that fades away by itself."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setWordWrap(True)
        self.setMaximumWidth(520)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, ms: int = 4200) -> None:
        self.setText(text)
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        self._timer.start(ms)

    def reposition(self) -> None:
        p = self.parentWidget()
        if p:
            self.move(max(8, (p.width() - self.width()) // 2), 18)

