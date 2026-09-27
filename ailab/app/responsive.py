"""Browser-style responsiveness for native Qt widgets.

* ``breakpoint(width)``: "compact" | "medium" | "wide", like CSS media queries. The main
  window recomputes it on every resize and each page adapts its own layout.
* ``FlowGrid``: cards in as many equal columns as fit, like CSS
  ``grid-template-columns: repeat(auto-fill, minmax(min_col, 1fr))``.
* ``Drawer``: a panel that slides over its host from the left or right edge, with a
  dimmed scrim behind it (click the scrim or press Esc to close), like a phone nav drawer.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QScrollArea, QVBoxLayout, QWidget

COMPACT, MEDIUM, WIDE = "compact", "medium", "wide"


def breakpoint(width: int, compact_below: int = 1100, wide_from: int = 1500) -> str:
    if width < compact_below:
        return COMPACT
    return MEDIUM if width < wide_from else WIDE


class FlowGrid(QWidget):
    """Equal-width columns; as many as fit at ``min_col`` pixels each."""

    def __init__(self, min_col: int = 300, spacing: int = 16, max_cols: int = 4, parent=None):
        super().__init__(parent)
        self.min_col, self.max_cols = min_col, max_cols
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(spacing)
        self._items: list[QWidget] = []
        self._cols = 0

    def add(self, w: QWidget) -> None:
        self._items.append(w)
        self._cols = 0
        self._relayout()

    def columns_for(self, width: int) -> int:
        sp = self._grid.spacing()
        return max(1, min(self.max_cols, (width + sp) // (self.min_col + sp)))

    def _relayout(self) -> None:
        cols = self.columns_for(max(self.width(), 1))
        if cols == self._cols:
            return
        self._cols = cols
        for w in self._items:
            self._grid.removeWidget(w)
        for i, w in enumerate(self._items):
            self._grid.addWidget(w, i // cols, i % cols)
        for c in range(self.max_cols):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._relayout()


class _Scrim(QWidget):
    clicked = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(2, 6, 12, 150);")
        self.hide()

    def mousePressEvent(self, e) -> None:
        self.clicked.emit()


class Drawer(QFrame):
    """Hosts ``content`` as an overlay sliding in from ``side`` ("left" | "right") of
    ``host``. While docked elsewhere the drawer is simply unused (see ``take``/``give``)."""

    toggled = Signal(bool)

    def __init__(self, host: QWidget, side: str = "left", width: int = 320):
        super().__init__(host)
        self.host, self.side, self.drawer_width = host, side, width
        self.setObjectName("drawer")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self.content: QWidget | None = None
        self.scrim = _Scrim(host)
        self.scrim.clicked.connect(self.close_drawer)
        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.is_open = False
        host.installEventFilter(self)
        self.hide()

    # ---------------------------------------------------------------- content
    def take(self, content: QWidget) -> None:
        """Move ``content`` into the drawer (it leaves wherever it was docked)."""
        self.content = content
        self._lay.addWidget(content)
        content.show()

    def give(self) -> QWidget | None:
        """Release the content so it can be docked again; closes the drawer."""
        self.close_drawer(animate=False)
        c, self.content = self.content, None
        if c is not None:
            self._lay.removeWidget(c)
        return c

    # ------------------------------------------------------------ open/close
    def _geometry(self, opened: bool) -> QPoint:
        w = min(self.drawer_width, int(self.host.width() * 0.88))
        self.resize(w, self.host.height())
        if self.side == "left":
            return QPoint(0 if opened else -w, 0)
        return QPoint(self.host.width() - w if opened else self.host.width(), 0)

    def open_drawer(self) -> None:
        if self.content is None or self.is_open:
            return
        self.is_open = True
        self.scrim.setGeometry(self.host.rect())
        self.scrim.show()
        self.scrim.raise_()
        self.move(self._geometry(False))
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(self._geometry(True))
        self._anim.start()
        self.toggled.emit(True)

    def close_drawer(self, animate: bool = True) -> None:
        if not self.is_open:
            return
        self.is_open = False
        self.scrim.hide()
        self._anim.stop()
        if animate:
            self._anim.setStartValue(self.pos())
            self._anim.setEndValue(self._geometry(False))
            self._anim.finished.connect(self._hide_after, Qt.SingleShotConnection)
            self._anim.start()
        else:
            self.hide()
        self.toggled.emit(False)

    def _hide_after(self) -> None:
        if not self.is_open:
            self.hide()

    def toggle(self) -> None:
        self.close_drawer() if self.is_open else self.open_drawer()

    def eventFilter(self, obj, e) -> bool:
        if obj is self.host and e.type() == QEvent.Resize and self.is_open:
            self.scrim.setGeometry(self.host.rect())
            self.move(self._geometry(True))
        return super().eventFilter(obj, e)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close_drawer()
        else:
            super().keyPressEvent(e)


class FitWidthScroll(QScrollArea):
    """A vertical scroller whose content always takes exactly the visible width, so
    wrapped text wraps instead of pushing the panel wider (like overflow-x: hidden)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())
