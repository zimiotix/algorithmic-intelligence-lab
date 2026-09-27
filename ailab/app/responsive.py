"""Browser-style responsiveness for native Qt widgets.

* ``breakpoint(width)``: "compact" | "medium" | "wide", like CSS media queries. The main
  window recomputes it on every resize and each page adapts its own layout.
* ``FlowGrid``: cards in as many equal columns as fit, like CSS
  ``grid-template-columns: repeat(auto-fill, minmax(min_col, 1fr))``.
* ``Drawer``: a panel that slides over its host from the left or right edge, with a
  dimmed scrim behind it (click the scrim or press Esc to close), like a phone nav drawer.
* ``FlowLayout``: items side by side at their natural width, wrapping onto new lines
  (CSS ``flex-wrap: wrap``), for tags and chips.
* ``Fold``: a section with a clickable header that opens and closes, like ``<details>``;
  it remembers its state between runs.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ailab.core import settings

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


class FlowLayout(QLayout):
    """Children at their preferred width, left to right, wrapping when the row is full."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        return size

    def _arrange(self, rect, move: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            if x > rect.x() and x + hint.width() > rect.right() + 1:
                x, y, line = rect.x(), y + line + self._gap, 0
            if move:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._gap
            line = max(line, hint.height())
        return y + line - rect.y()


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


class Fold(QWidget):
    """A section that opens and closes from its header (like HTML ``<details>``).

    ``key`` remembers the state in settings (``folds``); ``style`` is "section" for a
    panel's top-level headings or "group" for smaller headings inside one. Put content
    in ``body`` (a QVBoxLayout); ``set_badge`` shows a short note at the right end of the
    header, visible even when closed."""

    toggled = Signal(bool)

    def __init__(self, title: str, key: str = "", opened: bool = True, style: str = "section",
                 tip: str = "", parent=None):
        super().__init__(parent)
        self.key, self.title = key, title
        if key:
            opened = bool(settings.load().get("folds", {}).get(key, opened))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6 if style == "section" else 8)
        self.head = QPushButton()
        self.head.setObjectName("fold")
        self.head.setProperty("fold", style)
        self.head.setCursor(Qt.PointingHandCursor)
        self.head.setFocusPolicy(Qt.NoFocus)    # keep Space/keys for the simulation
        self.head.setCheckable(True)
        self.head.setToolTip(tip or "Click to open or close")
        hl = QHBoxLayout(self.head)
        hl.setContentsMargins(0, 0, 6, 0)
        hl.setAlignment(Qt.AlignVCenter)
        hl.addStretch(1)
        self.badge = QLabel()
        self.badge.setProperty("role", "faint")
        self.badge.setAttribute(Qt.WA_TransparentForMouseEvents)
        hl.addWidget(self.badge)
        lay.addWidget(self.head)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 2, 0, 4)
        self.body.setSpacing(10)
        lay.addWidget(self.content)
        self.head.toggled.connect(self._toggled)
        self.head.setChecked(opened)
        self._toggled(opened, save=False)

    @property
    def opened(self) -> bool:
        return self.head.isChecked()

    def set_open(self, on: bool) -> None:
        self.head.setChecked(on)

    def set_badge(self, text: str) -> None:
        self.badge.setText(text)

    def _toggled(self, on: bool, save: bool = True) -> None:
        self.head.setText(("▾  " if on else "▸  ") + self.title)
        self.content.setVisible(on)
        if save and self.key:
            folds = dict(settings.load().get("folds", {}))
            folds[self.key] = on
            settings.save({"folds": folds})
        self.toggled.emit(on)
