"""The main window: sidebar catalogue, home page, chapter page (Overview / Lab / Deep dive)."""

from __future__ import annotations

import html
import sys
import traceback

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import compute, paths, settings
from ..core.catalog import Catalog, ChapterInfo, load_sim_class
from ..core.rng import SEED_MAX, fresh_seed
from ..core.sim import SimContext
from ..core.system import SystemInfo
from ..text.markdown import split_sections, symbol_table
from . import theme
from .lab import GuidePanel, Hotbar, Toast
from .markdown_view import MarkdownView
from .responsive import (
    COMPACT,
    Drawer,
    FitWidthScroll,
    FlowGrid,
    FlowLayout,
    Fold,
    breakpoint,
)
from .viewport import Viewport
from .widgets import ChapterCard, Logo, ParamPanel, StatCard, chip, dot_pixmap

THUMBS = paths.cache_dir() / "thumbs"


def runs_on_label(info: SystemInfo) -> str:
    if info.on_gpu:
        return "CUDA · " + (info.compute_name or "GPU").replace("GeForce ", "")
    return "CPU"


def device_for(chapter: ChapterInfo) -> str:
    """Heavy chapters go to the GPU when there is one; light ones stay on the CPU."""
    return compute.device() if chapter.compute == "gpu" else "cpu"


def menu_button() -> QToolButton:
    """☰: opens the chapter list when it is tucked away on narrow windows."""
    b = QToolButton()
    b.setText("☰")
    b.setToolTip("Chapters")
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(f"QToolButton {{ background: transparent; border: 1px solid {theme.LINE};"
                    f" border-radius: 8px; padding: 2px 9px; font-size: 17px; color: {theme.TEXT}; }}"
                    f"QToolButton:hover {{ background: {theme.BG3}; }}")
    b.hide()
    return b


# ======================================================================= sidebar
class Sidebar(QFrame):
    chapterChosen = Signal(str)
    homeRequested = Signal()
    systemRequested = Signal()

    def __init__(self, catalog: Catalog, info: SystemInfo):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(300)
        self.catalog = catalog
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 18, 16, 14)
        lay.setSpacing(12)

        brand = QPushButton()
        brand.setProperty("role", "ghost")
        brand.setCursor(Qt.PointingHandCursor)
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.addWidget(Logo(34))
        name = QLabel("<b>Algorithmic</b><br><span style='color:#8b98a9'>Intelligence Lab</span>")
        name.setStyleSheet("font-size:14px;")
        bl.addWidget(name, 1)
        brand.setMinimumHeight(52)
        brand.clicked.connect(self.homeRequested)
        lay.addWidget(brand)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search chapters, ideas, tags…")
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setRootIsDecorated(False)
        self.tree.itemClicked.connect(self._clicked)
        self._items: dict[str, QTreeWidgetItem] = {}
        for t in catalog.tracks:
            top = QTreeWidgetItem([t.title])
            top.setIcon(0, QIcon(dot_pixmap(t.color)))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            top.setFlags(Qt.ItemIsEnabled)
            self.tree.addTopLevelItem(top)
            for c in t.chapters:
                it = QTreeWidgetItem([c.title])
                it.setData(0, Qt.UserRole, c.id)
                it.setToolTip(0, c.summary)
                top.addChild(it)
                self._items[c.id] = it
            top.setExpanded(True)
        for t in catalog.planned:
            it = QTreeWidgetItem([t.title + "  · soon"])
            it.setIcon(0, QIcon(dot_pixmap("#334155")))
            it.setForeground(0, Qt.gray)
            it.setFlags(Qt.NoItemFlags)
            it.setToolTip(0, t.description)
            self.tree.addTopLevelItem(it)
        lay.addWidget(self.tree, 1)

        self.sys_card = QFrame()
        self.sys_card.setObjectName("stat")
        sl = QVBoxLayout(self.sys_card)
        sl.setContentsMargins(12, 10, 12, 10)
        sl.setSpacing(3)
        head = QLabel("THIS MACHINE")
        head.setProperty("role", "section")
        self.sys_compute = QLabel()
        self.sys_render = QLabel()
        self.sys_host = QLabel()
        for w in (self.sys_compute, self.sys_render, self.sys_host):
            w.setWordWrap(True)
            w.setStyleSheet("font-size:12px;")
        more = QPushButton("System details")
        more.setProperty("role", "ghost")
        more.clicked.connect(self.systemRequested)
        sl.addWidget(head)
        sl.addWidget(self.sys_compute)
        sl.addWidget(self.sys_render)
        sl.addWidget(self.sys_host)
        sl.addWidget(more, 0, Qt.AlignLeft)
        lay.addWidget(self.sys_card)
        self.refresh_system(info)

    def refresh_system(self, info: SystemInfo) -> None:
        c = theme.GOOD if info.on_gpu else theme.WARN
        self.sys_compute.setText(f"<span style='color:{c}'>●</span> Compute: <b>"
                                 f"{info.compute_label}</b>")
        r = info.gl_renderer.split("/")[0] if info.gl_renderer else "starting…"
        self.sys_render.setText(f"<span style='color:{theme.ACCENT}'>●</span> Draws on: {r}")
        self.sys_host.setText(f"<span style='color:{theme.MUTED}'>●</span> "
                              f"{info.cpu_threads} threads · {info.ram_total_mb / 1024:.0f} GB RAM")

    def select(self, cid: str) -> None:
        it = self._items.get(cid)
        if it:
            self.tree.setCurrentItem(it)

    def _clicked(self, item: QTreeWidgetItem) -> None:
        cid = item.data(0, Qt.UserRole)
        if cid:
            self.chapterChosen.emit(cid)

    def _filter(self, text: str) -> None:
        q = text.lower().strip()
        for cid, it in self._items.items():
            c = self.catalog.chapters[cid]
            hay = " ".join([c.title, c.summary, *c.tags, c.track]).lower()
            it.setHidden(bool(q) and q not in hay)


# ========================================================================= home
class HomePage(QScrollArea):
    chapterChosen = Signal(str)

    def __init__(self, catalog: Catalog, info: SystemInfo):
        super().__init__()
        self.setWidgetResizable(True)
        self.catalog, self.info = catalog, info
        root = QWidget()
        self.setWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(44, 36, 44, 40)
        lay.setSpacing(18)
        self._lay = lay
        self.menu = menu_button()
        lay.addWidget(self.menu, 0, Qt.AlignLeft)

        h1 = QLabel("Watch intelligence emerge from simple, exact rules.")
        h1.setProperty("role", "h1")
        h1.setWordWrap(True)
        sub = QLabel("Every chapter is a living simulation driven by a deterministic algorithm, "
                     "no neural networks. Play with it, switch on the overlays to see what it "
                     "senses and decides, then open the Deep dive for the maths.")
        sub.setWordWrap(True)
        sub.setProperty("role", "muted")
        sub.setStyleSheet("font-size:15px;")
        lay.addWidget(h1)
        lay.addWidget(sub)

        stats = FlowGrid(min_col=190, spacing=12, max_cols=5)
        gpu = info.cuda_gpu
        self.stat_compute = StatCard("Compute", info.compute_label.split(" · ")[0],
                                     info.compute_name or info.cpu_model,
                                     theme.GOOD if info.on_gpu else theme.WARN)
        self.stat_render = StatCard("Rendering", "OpenGL", "detecting…")
        self.stat_gpu = StatCard("GPU memory", f"{gpu.vram_mb / 1024:.0f} GB" if gpu else "—",
                                 f"driver {gpu.driver}" if gpu else "no CUDA device")
        self.stat_cpu = StatCard("CPU", f"{info.cpu_cores} cores / {info.cpu_threads} threads",
                                 info.cpu_model, theme.TEXT)
        self.stat_ram = StatCard("Memory", f"{info.ram_total_mb / 1024:.1f} GB",
                                 f"{info.ram_available_mb / 1024:.1f} GB free", theme.TEXT)
        for s in (self.stat_compute, self.stat_render, self.stat_gpu, self.stat_cpu,
                  self.stat_ram):
            stats.add(s)
        lay.addWidget(stats)
        notes = info.advice()
        if notes:
            n = QLabel("  ·  ".join(notes))
            n.setWordWrap(True)
            n.setProperty("role", "faint")
            lay.addWidget(n)

        self.cards: dict[str, ChapterCard] = {}
        label = runs_on_label(info)
        for t in catalog.tracks:
            lay.addSpacing(10)
            head = QHBoxLayout()
            dot = QLabel()
            dot.setPixmap(dot_pixmap(t.color, 12))
            title = QLabel(t.title)
            title.setProperty("role", "h2")
            head.addWidget(dot)
            head.addWidget(title)
            head.addStretch(1)
            lay.addLayout(head)
            if t.description:
                d = QLabel(t.description)
                d.setProperty("role", "muted")
                lay.addWidget(d)
            grid = FlowGrid(min_col=300, spacing=16, max_cols=4)
            for c in t.chapters:
                card = ChapterCard(c, t.color, label)
                card.clicked.connect(self.chapterChosen)
                grid.add(card)
                self.cards[c.id] = card
            lay.addWidget(grid)
        if catalog.planned:
            lay.addSpacing(10)
            soon = QLabel("COMING NEXT")
            soon.setProperty("role", "section")
            lay.addWidget(soon)
            row = QHBoxLayout()
            row.setSpacing(8)
            for t in catalog.planned:
                c = chip(t.title, t.color)
                c.setToolTip(t.description)
                row.addWidget(c)
            row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(1)
        self._queue: list[str] = []

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        m = 20 if breakpoint(self.width()) == COMPACT else 44
        self._lay.setContentsMargins(m, 24 if m == 20 else 36, m, 40)

    def set_gl(self, info: SystemInfo) -> None:
        self.stat_render.set(info.render_gpu_label, info.gl_renderer.split("/")[0])

    # thumbnails are rendered headlessly once, then cached outside the repo
    def start_thumbnails(self) -> None:
        THUMBS.mkdir(parents=True, exist_ok=True)
        self._queue = []
        for cid, card in self.cards.items():
            p = THUMBS / f"{cid}.png"
            if p.exists():
                card.set_thumbnail(str(p))
            else:
                self._queue.append(cid)
        QTimer.singleShot(300, self._next_thumb)

    def _next_thumb(self) -> None:
        if not self._queue:
            return
        cid = self._queue.pop(0)
        card = self.cards[cid]
        info = self.catalog.chapters[cid]
        try:
            from .snapshot import snapshot

            snapshot(info, str(THUMBS / f"{cid}.png"), device_for(info), frames=420,
                     size=(640, 360))
            card.set_thumbnail(str(THUMBS / f"{cid}.png"))
        except Exception as e:  # a preview is optional; never block the app
            card.thumb.setText(f"preview unavailable ({type(e).__name__})")
        QTimer.singleShot(50, self._next_thumb)


# ====================================================================== chapter
class ChapterPage(QWidget):
    statusText = Signal(str)
    focusMode = Signal(bool)

    def __init__(self, info: SystemInfo, seed: int | None = None):
        super().__init__()
        self.sysinfo = info
        self._start_seed = fresh_seed() if seed is None else seed   # a new run each launch
        self.chapter: ChapterInfo | None = None
        self.sim = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # --- header
        header = QFrame()
        header.setObjectName("toolbar")
        self.header = header
        hl = QVBoxLayout(header)
        hl.setContentsMargins(28, 18, 28, 0)
        hl.setSpacing(6)
        top = QHBoxLayout()
        self.track_chip = QLabel()
        self.badges = QHBoxLayout()
        self.badges.setSpacing(6)
        top.addWidget(self.track_chip)
        top.addLayout(self.badges)
        top.addStretch(1)
        self.title = QLabel()
        self.title.setStyleSheet("font-size:24px; font-weight:700;")
        self.summary = QLabel()
        self.summary.setProperty("role", "muted")
        self.summary.setWordWrap(True)
        self.top_row = QWidget()
        self.top_row.setLayout(top)
        top.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.top_row)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.menu = menu_button()
        title_row.addWidget(self.menu)
        title_row.addWidget(self.title, 1)
        hl.addLayout(title_row)
        hl.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        hl.addSpacing(4)
        lay.addWidget(header)
        lay.addWidget(self.tabs, 1)

        # --- overview
        ov = QWidget()
        ol = QHBoxLayout(ov)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        self.intro = MarkdownView()
        ol.addWidget(self.intro, 3)
        side = QFrame()
        self.overview_side = side
        side.setObjectName("panel")
        side.setMinimumWidth(260)
        side.setMaximumWidth(340)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(22, 24, 22, 22)
        sl.setSpacing(12)
        enter = QPushButton("Enter the lab  ▶")
        enter.setProperty("role", "primary")
        enter.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        sl.addWidget(enter)
        h = QLabel("HOW TO PLAY")
        h.setProperty("role", "section")
        sl.addWidget(h)
        self.controls_box = QVBoxLayout()
        self.controls_box.setSpacing(8)
        sl.addLayout(self.controls_box)
        h2 = QLabel("RUNS ON")
        h2.setProperty("role", "section")
        sl.addSpacing(6)
        sl.addWidget(h2)
        self.runs_on = QLabel()
        self.runs_on.setWordWrap(True)
        sl.addWidget(self.runs_on)
        sl.addStretch(1)
        ol.addWidget(side)
        self.tabs.addTab(ov, "Overview")

        # --- lab: Guide | view | Inspector, all resizable; Tab = focus mode
        lab = QWidget()
        ll = QVBoxLayout(lab)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        self.lab_toolbar = self._toolbar()
        ll.addWidget(self.lab_toolbar)
        self.view_host = QWidget()
        vh = QGridLayout(self.view_host)
        vh.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport()
        vh.addWidget(self.viewport, 0, 0)
        self.hud = QLabel(self.view_host)      # floats over the viewport, not in the layout
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud.setStyleSheet("QLabel { background: rgba(8,12,20,190); border:1px solid "
                               "#1e2a3b; border-radius:10px; padding:8px 12px; font-size:12px;}")
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("QLabel { background: rgba(8,12,20,225); border:1px solid "
                                  "#2f4460; border-radius:12px; padding:18px 24px; "
                                  "font-size:14px; }")
        self.banner.setMaximumWidth(560)
        self.banner.hide()
        vh.addWidget(self.banner, 0, 0, Qt.AlignCenter)
        self.hotbar = Hotbar(self.view_host)
        self.hotbar.chosen.connect(self.set_tool)
        self.hotbar.hide()
        self.toast = Toast(self.view_host)
        self.view_host.installEventFilter(self)

        self.guide = GuidePanel()
        self.guide.experimentToggled.connect(self._experiment_toggled)
        self.inspector = self._side_panel()
        self.split = QSplitter(Qt.Horizontal)
        self.split.setHandleWidth(1)
        self.split.addWidget(self.guide)
        self.split.addWidget(self.view_host)
        self.split.addWidget(self.inspector)
        for i in range(3):              # side panels shrink to their minimum, never to 0
            self.split.setCollapsible(i, False)
        self.split.setStretchFactor(1, 1)
        self._restore_split()
        self._split_save = QTimer(self)
        self._split_save.setSingleShot(True)
        self._split_save.timeout.connect(self._save_split)
        self.split.splitterMoved.connect(lambda *_: self._split_save.start(600))
        ll.addWidget(self.split, 1)
        self.tabs.addTab(lab, "Lab")
        # narrow windows: Guide and Inspector slide over the view instead of squeezing it
        self.guide_drawer = Drawer(self.view_host, "left", 320)
        self.insp_drawer = Drawer(self.view_host, "right", 340)
        self.guide_drawer.toggled.connect(lambda on: self._sync_btn(self.guide_btn, on))
        self.insp_drawer.toggled.connect(lambda on: self._sync_btn(self.insp_btn, on))
        self.compact = False
        self.focus_mode = False
        self.tool_key = ""
        self.done: set[str] = set()
        self._live_ok = True
        self.params_panel = None
        self._preset_btns: list = []

        # --- deep dive
        dd = QWidget()
        dl = QVBoxLayout(dd)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(0)
        segbar = QFrame()
        segbar.setObjectName("toolbar")
        self.seg_layout = QHBoxLayout(segbar)
        self.seg_layout.setContentsMargins(24, 10, 24, 10)
        self.seg_layout.setSpacing(0)
        self.seg_group = QButtonGroup(self)
        self.seg_group.setExclusive(True)
        dl.addWidget(segbar)
        self.deep = MarkdownView()
        dl.addWidget(self.deep, 1)
        self.tabs.addTab(dd, "Deep dive")
        self.tabs.currentChanged.connect(self._tab_changed)

        # wiring
        self.viewport.hotkey.connect(self._hotkey)
        self.viewport.stats.connect(self._stats)
        self.viewport.failed.connect(self._failed)
        self._last_stats: dict = {}
        self.hud_timer = QTimer(self)
        self.hud_timer.timeout.connect(self._update_hud)
        self.hud_timer.start(160)

    # ------------------------------------------------------------- lab chrome
    def _toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("toolbar")
        tl = QHBoxLayout(bar)
        tl.setContentsMargins(16, 8, 16, 8)
        tl.setSpacing(8)
        self.play = QPushButton("Pause")
        self.play.setToolTip("Space")
        self.play.clicked.connect(self.toggle_pause)
        step = QPushButton("Step")
        step.setToolTip("Advance one fixed step (.)")
        step.clicked.connect(lambda: self._step())
        restart = QPushButton("Restart")
        restart.setToolTip("Same seed, same run: everything here is deterministic (R)")
        restart.clicked.connect(self.restart)
        tl.addWidget(self.play)
        tl.addWidget(step)
        tl.addWidget(restart)
        tl.addSpacing(10)
        sl = QLabel("seed")
        sl.setProperty("role", "muted")
        self.seed = QSpinBox()
        self.seed.setRange(0, SEED_MAX - 1)
        self.seed.setValue(self._start_seed)
        self.seed.setToolTip("The only source of randomness. Every launch picks a new one; "
                             "type a seed back in to replay that exact run.")
        self.seed.editingFinished.connect(self.restart)
        dice = QPushButton("New seed")
        dice.clicked.connect(lambda: (self.seed.setValue(fresh_seed()), self.restart()))
        tl.addWidget(sl)
        tl.addWidget(self.seed)
        tl.addWidget(dice)
        tl.addSpacing(10)
        self.speed_group = QButtonGroup(self)
        self._speeds = (0.25, 0.5, 1.0, 2.0, 4.0)
        for i, s in enumerate(self._speeds):
            b = QPushButton(f"{s:g}×")
            b.setCheckable(True)
            b.setProperty("role", "seg")
            b.setChecked(s == 1.0)
            b.clicked.connect(lambda _=False, s=s: setattr(self.viewport.clock, "speed", s))
            self.speed_group.addButton(b, i)
            tl.addWidget(b)
        tl.addSpacing(10)
        view = QPushButton("Reset view")
        view.setProperty("role", "ghost")
        view.setToolTip("Wheel to zoom, middle-drag to pan (Home)")
        view.clicked.connect(self.viewport_reset)
        tl.addWidget(view)
        tl.addStretch(1)
        self._tb_speed = list(self.speed_group.buttons())
        self._tb_view, self._tb_seed = view, [sl, self.seed, dice]
        self.guide_btn = QPushButton("◧ Guide")
        self.insp_btn = QPushButton("Controls ◨")
        focus = QPushButton("⛶ Focus")
        self._focus_btn = focus
        focus.setToolTip("Only the simulation (Tab). Esc or Tab to come back.")
        for b in (self.guide_btn, self.insp_btn):
            b.setCheckable(True)
            b.setChecked(True)
            b.setProperty("role", "ghost")
        focus.setProperty("role", "ghost")
        self.guide_btn.toggled.connect(lambda on: self._show_pane("guide", on))
        self.insp_btn.toggled.connect(lambda on: self._show_pane("inspector", on))
        focus.clicked.connect(lambda: self.set_focus_mode(True))
        tl.addWidget(self.guide_btn)
        tl.addWidget(self.insp_btn)
        tl.addWidget(focus)
        tl.addSpacing(8)
        self.perf = QLabel()
        self.perf.setProperty("role", "muted")
        self.perf.setStyleSheet(f"font-family:'{theme.mono_family()}'; font-size:12px;")
        tl.addWidget(self.perf)
        return bar

    # -------------------------------------------------------------- responsive
    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._responsive()

    def _responsive(self) -> None:
        if not hasattr(self, "insp_drawer"):      # still being built
            return
        w = self.width()
        self.set_compact(w < 1150)
        self._fit_toolbar()
        narrow = w < 980
        self.title.setStyleSheet(f"font-size:{20 if narrow else 24}px; font-weight:700;")
        self.overview_side.setVisible(w >= 900)

    def _fit_toolbar(self) -> None:
        """Like a responsive web toolbar: show everything, then hide the least important
        items one at a time until the rest fits without squashing."""
        bar = self.lab_toolbar
        steps = ([lambda: self.perf.hide()]
                 + [lambda b=b: b.hide() for b in self._tb_speed if not b.isChecked()]
                 + [lambda: (self.guide_btn.setText("◧"), self.insp_btn.setText("◨"),
                             self._focus_btn.setText("⛶"))]
                 + [lambda: self._tb_view.hide()]
                 + [lambda: [x.hide() for x in self._tb_seed]]
                 + [lambda b=b: b.hide() for b in self._tb_speed if b.isChecked()])
        self.perf.show()
        for x in self._tb_speed + [self._tb_view] + self._tb_seed:
            x.show()
        self.guide_btn.setText("◧ Guide")
        self.insp_btn.setText("Controls ◨")
        self._focus_btn.setText("⛶ Focus")
        lay = bar.layout()
        for step in steps:
            lay.invalidate()
            if lay.sizeHint().width() <= bar.width():
                break
            step()

    def set_compact(self, compact: bool) -> None:
        """Dock the Guide and Inspector beside the view, or tuck them into drawers."""
        if compact == self.compact:
            return
        self.compact = compact
        if compact:
            self.guide_drawer.take(self.guide)
            self.insp_drawer.take(self.inspector)
            for b in (self.guide_btn, self.insp_btn):
                self._sync_btn(b, False)
        else:
            g, i = self.guide_drawer.give(), self.insp_drawer.give()
            self.split.insertWidget(0, g)
            self.split.insertWidget(2, i)
            g.show()
            i.show()
            self._restore_split()
            for b in (self.guide_btn, self.insp_btn):
                self._sync_btn(b, True)
        self._place_overlays()

    SPLIT_DEFAULT = (290, 1000, 340)

    def _saved_split(self) -> list[int]:
        """Saved panel widths, or the defaults if they are missing or unusable (a panel
        saved at 0 px would otherwise stay invisible forever)."""
        sizes = settings.load().get("lab_split")
        ok = (isinstance(sizes, list) and len(sizes) == 3
              and all(isinstance(x, int) for x in sizes)
              and sizes[0] >= self.guide.minimumWidth()
              and sizes[2] >= self.inspector.minimumWidth() and sizes[1] >= 200)
        return list(sizes) if ok else list(self.SPLIT_DEFAULT)

    def _restore_split(self) -> None:
        self.split.setSizes(self._saved_split())

    def _save_split(self) -> None:
        """Remember the widths of the panels that are showing; a hidden panel reports 0,
        so it keeps its previously saved width."""
        sizes, old = self.split.sizes(), self._saved_split()
        for i, w in enumerate((self.guide, self.view_host, self.inspector)):
            if not w.isVisible() or sizes[i] <= 0:
                sizes[i] = old[i]
        settings.save({"lab_split": sizes})

    def _show_pane(self, which: str, on: bool) -> None:
        pane = self.guide if which == "guide" else self.inspector
        drawer = self.guide_drawer if which == "guide" else self.insp_drawer
        if self.compact:
            if on:
                (self.insp_drawer if which == "guide" else self.guide_drawer).close_drawer()
                drawer.open_drawer()
            else:
                drawer.close_drawer()
        else:
            pane.setVisible(on)
            i = 0 if which == "guide" else 2
            sizes = self.split.sizes()
            if on and sizes[i] < pane.minimumWidth():
                want = self._saved_split()[i]
                sizes[1] = max(200, sizes[1] - want)
                sizes[i] = want
                self.split.setSizes(sizes)

    @staticmethod
    def _sync_btn(button, on: bool) -> None:
        button.blockSignals(True)
        button.setChecked(on)
        button.blockSignals(False)

    def viewport_reset(self) -> None:
        self.viewport.camera.reset()

    def _side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(250)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FitWidthScroll()
        inner = QWidget()
        inner.setStyleSheet(f"background:{theme.BG1};")
        self.panel_layout = QVBoxLayout(inner)
        self.panel_layout.setContentsMargins(18, 16, 18, 18)
        self.panel_layout.setSpacing(12)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return panel

    def _rebuild_panel(self) -> None:
        """The Controls panel: presets, then every parameter in folding groups, then the
        overlays. Live maths lives in the Guide."""
        lay = self.panel_layout
        while lay.count():
            it = lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        sim = self.sim
        self.params_panel = None
        self._preset_btns = []
        if sim is None:
            return
        self.guide.set_live(sim.LIVE_MATH)
        cid = self.chapter.id if self.chapter else ""
        if sim.PRESETS:
            fold = Fold("PRESETS", "controls:presets", True,
                        tip="Named starting points. Anything a preset doesn't mention goes "
                            "back to its default.")
            chips = QWidget()
            flow = FlowLayout(chips, spacing=6)
            group = QButtonGroup(fold)
            for pr in sim.PRESETS:
                b = QPushButton(pr.title)
                b.setProperty("role", "preset")
                b.setCheckable(True)
                b.setToolTip(pr.tip)
                b.clicked.connect(lambda _=False, pr=pr: self._apply_preset(pr))
                group.addButton(b)
                flow.addWidget(b)
                self._preset_btns.append((pr, b))
            self._preset_group = group
            fold.body.addWidget(chips)
            lay.addWidget(fold)
        head = QHBoxLayout()
        h = QLabel("PARAMETERS")
        h.setProperty("role", "section")
        head.addWidget(h)
        head.addStretch(1)
        reset = QPushButton("Reset all")
        reset.setProperty("role", "ghost")
        reset.setToolTip("Every parameter back to its default")
        reset.clicked.connect(lambda: self._apply_values({}))
        head.addWidget(reset)
        lay.addSpacing(4)
        lay.addLayout(head)
        note = QLabel("Hover a name for what it does. ⟲ restarts the run.")
        note.setProperty("role", "faint")
        note.setWordWrap(True)
        lay.addWidget(note)
        panel = ParamPanel(sim.PARAMS, sim.p, fold_key=f"params:{cid}")
        panel.changed.connect(self._param)
        self.params_panel = panel
        lay.addWidget(panel)
        self._ov_boxes: list = []
        if sim.OVERLAYS:
            fold = Fold("SEE INSIDE", "controls:overlays", False,
                        tip="Visual layers that show what the algorithm is thinking")
            fold.set_badge(f"{sum(bool(sim.show.get(o.key)) for o in sim.OVERLAYS)} on")
            self._ov_fold = fold
            for i, o in enumerate(sim.OVERLAYS):
                key = f"Ctrl+{i + 1}" if len(sim.TOOLS) > 1 else f"{i + 1}"
                cb = QCheckBox(o.label.replace("&", "&&"))
                cb.setChecked(sim.show.get(o.key))
                cb.setToolTip((o.help + "  " if o.help else "") + (f"[{key}]" if i < 9 else ""))
                cb.toggled.connect(lambda v, k=o.key: self._overlay(k, v))
                cb.setObjectName(f"ov_{i + 1}")
                self._ov_boxes.append((o.key, cb))
                fold.body.addWidget(cb)
            lay.addSpacing(6)
            lay.addWidget(fold)
        lay.addStretch(1)
        self._sync_presets()

    def _overlay(self, key: str, on: bool) -> None:
        if self.sim:
            self.sim.show.set(key, on)
            n = sum(bool(self.sim.show.get(o.key)) for o in self.sim.OVERLAYS)
            self._ov_fold.set_badge(f"{n} on")

    def _apply_preset(self, preset) -> None:
        self._apply_values(preset.values)
        self.toast.show_message(f"<b>{html.escape(preset.title)}</b>"
                                + (f"<br>{html.escape(preset.tip)}" if preset.tip else ""),
                                3500)

    def _apply_values(self, values: dict) -> None:
        if not self.sim:
            return
        try:
            self.sim.apply_values(values)
        except Exception as e:
            self._failed(f"preset: {type(e).__name__}: {e}")
            return
        self.viewport.clock.reset()
        if self.params_panel is not None:
            self.params_panel.refresh(self.sim.p)
        self._sync_presets()

    def _sync_presets(self) -> None:
        """Light up the preset the current values match exactly, if any."""
        if not self.sim or not self._preset_btns:
            return
        p = self.sim.p
        match = None
        for pr, _ in self._preset_btns:
            want = {s.key: s.clamp(pr.values.get(s.key, p.default(s.key)))
                    for s in self.sim.PARAMS}
            if all(want[k] == p.get(k) for k in want):
                match = pr
                break
        self._preset_group.setExclusive(False)
        for pr, b in self._preset_btns:
            b.setChecked(pr is match)
        self._preset_group.setExclusive(True)

    # ---------------------------------------------------------------- loading
    def load(self, chapter: ChapterInfo, keep_tab: bool = False) -> None:
        self.chapter = chapter
        self.viewport.set_sim(None)
        self.sim = None
        track_color = getattr(chapter, "_color", theme.ACCENT)
        self.track_chip.setText(chapter.track.upper())
        self.track_chip.setStyleSheet(f"color:{track_color}; font-size:11px; font-weight:700;"
                                      "letter-spacing:1.5px;")
        while self.badges.count():
            w = self.badges.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.badges.addWidget(QLabel(theme.dots(chapter.difficulty, color=track_color)))
        for tag in chapter.tags[:4]:
            self.badges.addWidget(chip(tag))
        self.title.setText(chapter.title)
        self.summary.setText(chapter.summary)

        # overview
        self.intro.set_markdown(chapter.text("intro.md") or f"# {chapter.title}\n\n"
                                + chapter.summary)
        while self.controls_box.count():
            it = self.controls_box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
            elif it.widget():
                it.widget().deleteLater()
        for key, action in chapter.controls + [("1 – 9", "Pick a tool (Lab hotbar)"),
                                               ("Ctrl + 1 – 9", "Toggle overlays"),
                                               ("Space · . · R", "Pause · step · restart"),
                                               ("Tab", "Focus mode: only the simulation")]:
            row = QHBoxLayout()
            k = QLabel(key)
            k.setStyleSheet(f"background:{theme.BG3}; border:1px solid {theme.LINE};"
                            "border-radius:6px; padding:3px 8px; font-size:12px;")
            k.setFixedWidth(130)
            a = QLabel(action)
            a.setWordWrap(True)
            a.setProperty("role", "muted")
            row.addWidget(k, 0, Qt.AlignTop)
            row.addWidget(a, 1)
            self.controls_box.addLayout(row)
        dev = device_for(chapter)
        if dev.startswith("cuda"):
            self.runs_on.setText(f"<span style='color:{theme.GOOD}'>●</span> NVIDIA GPU via "
                                 f"CUDA ({self.sysinfo.compute_name}). Heavy chapter: thousands "
                                 "of agents update in parallel.")
        elif chapter.compute == "gpu":
            self.runs_on.setText(f"<span style='color:{theme.WARN}'>●</span> CPU fallback "
                                 "(no CUDA). Same algorithm, smaller default agent counts.")
        else:
            self.runs_on.setText(f"<span style='color:{theme.MUTED}'>●</span> CPU. Light "
                                 "chapter: the GPU isn't needed here.")

        # deep dive
        for b in self.seg_group.buttons():
            self.seg_group.removeButton(b)
            b.deleteLater()
        while self.seg_layout.count():
            it = self.seg_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.sections = split_sections(chapter.text("advanced.md"))
        levels = QLabel("LEVEL  ")
        levels.setProperty("role", "section")
        self.seg_layout.addWidget(levels)
        for i, (title, _) in enumerate(self.sections):
            b = QPushButton(title)
            b.setCheckable(True)
            b.setProperty("role", "seg")
            b.clicked.connect(lambda _=False, i=i: self._show_section(i))
            self.seg_group.addButton(b, i)
            self.seg_layout.addWidget(b)
        self.seg_layout.addStretch(1)
        media = sorted(chapter.media_dir.glob("*.mp4")) if chapter.media_dir.exists() else []
        for m in media:
            mb = QPushButton(f"▶ {m.stem.replace('_', ' ')}")
            mb.setProperty("role", "ghost")
            mb.clicked.connect(lambda _=False, m=m: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(m))))
            self.seg_layout.addWidget(mb)
        self._deep_loaded = False
        if not keep_tab:
            self.tabs.setCurrentIndex(0)
        self._start_sim()

    def _show_section(self, i: int) -> None:
        if 0 <= i < len(self.sections):
            b = self.seg_group.button(i)
            if b:
                b.setChecked(True)
            title, body = self.sections[i]
            if title.strip().lower() == "the math" and self.chapter:
                # terms before equations: the notation primer and every symbol, up front
                try:
                    cls = type(self.sim) if self.sim else load_sim_class(self.chapter)
                    table = symbol_table(cls.PARAMS, self.sim.p if self.sim else None,
                                         self.chapter.glossary)
                    body = f"## Symbols used here\n\n{table}\n{body}"
                except Exception:
                    traceback.print_exc()
            self.deep.set_markdown(f"# {title}\n\n{body}")

    def _tab_changed(self, idx: int) -> None:
        # The Lab gets the room: the header shrinks to the title.
        self.top_row.setVisible(idx != 1)
        self.summary.setVisible(idx != 1)
        if idx != 1 and self.focus_mode:
            self.set_focus_mode(False)
        if idx == 2 and not self._deep_loaded:
            self._deep_loaded = True
            self._show_section(0)
        if idx == 1:
            self.viewport.setFocus()

    def _start_sim(self) -> None:
        chapter = self.chapter
        dev = device_for(chapter)
        where = f"the GPU ({self.sysinfo.compute_name})" if dev.startswith("cuda") else "the CPU"
        self.banner.setText(f"<b>Preparing {chapter.title}</b><br><span style='color:#8b98a9'>"
                            f"Compiling kernels for {where}. The first run takes a few "
                            "seconds; after that they are cached.</span>")
        self.banner.show()
        QTimer.singleShot(30, lambda: self._create_sim(chapter, dev))

    def _create_sim(self, chapter: ChapterInfo, dev: str) -> None:
        if chapter is not self.chapter:
            return
        try:
            cls = load_sim_class(chapter)
            _precompile(cls, dev)
            self.sim = cls(SimContext(device=dev, seed=self.seed.value()))
        except Exception as e:
            traceback.print_exc()
            self.banner.setText(f"<b>Could not start this chapter</b><br>"
                                f"<span style='color:#f87171'>{type(e).__name__}: {e}</span>")
            return
        self.banner.hide()
        self.viewport.set_sim(self.sim)
        self.viewport.clock.paused = False
        self.play.setText("Pause")
        self._set_speed(self.sim.playback)
        self._rebuild_panel()
        self._setup_lab(chapter)
        self.statusText.emit(f"{chapter.title} · running on {dev}")

    # ---------------------------------------------------------------- actions
    def _set_speed(self, speed: float) -> None:
        """Start at the chapter's playback speed and light up the matching button."""
        speed = min(self._speeds, key=lambda s: abs(s - speed))
        self.viewport.clock.speed = speed
        self.speed_group.button(self._speeds.index(speed)).setChecked(True)
        self._fit_toolbar()

    def toggle_pause(self) -> None:
        c = self.viewport.clock
        c.paused = not c.paused
        self.play.setText("Play" if c.paused else "Pause")

    def _step(self) -> None:
        c = self.viewport.clock
        if not c.paused:
            self.toggle_pause()
        c.request_step()

    def restart(self) -> None:
        if self.sim:
            self.sim.reset(self.seed.value())
            self.viewport.clock.reset()

    def _param(self, key: str, value) -> None:
        if self.sim:
            try:
                self.sim.set_param(key, value)
            except Exception as e:
                self._failed(f"{key}: {type(e).__name__}: {e}")
            self._sync_presets()

    def _hotkey(self, name: str) -> None:
        if name == "Space":
            self.toggle_pause()
        elif name == ".":
            self._step()
        elif name == "R":
            self.restart()
        elif name.startswith("tool:"):
            k = int(name[5:]) - 1
            tools = self.sim.TOOLS if self.sim else []
            if len(tools) > 1:
                if k < len(tools):
                    self.set_tool(tools[k].key)
            else:                                   # no hotbar: digits toggle overlays
                self._toggle_overlay(k + 1)
        elif name.startswith("overlay:"):
            self._toggle_overlay(int(name[8:]))
        elif name == "focus":
            self.set_focus_mode(not self.focus_mode)
        elif name == "escape" and self.focus_mode:
            self.set_focus_mode(False)

    def _toggle_overlay(self, n: int) -> None:
        cb = self.findChild(QCheckBox, f"ov_{n}")
        if cb:
            cb.toggle()

    def _stats(self, s: dict) -> None:
        self._last_stats = s

    def _update_hud(self) -> None:
        if not self.sim or not self.isVisible():
            self.hud.hide()
            return
        s = self._last_stats
        dev = self.sim.device.replace("cuda:0", "CUDA")
        self.perf.setText(f"{s.get('fps', 0):3.0f} fps · {s.get('compute_ms', 0):4.1f} ms {dev}")
        self.perf.setToolTip(f"Frames per second, and time per frame spent on the simulation "
                             f"({dev}). Drawing takes {s.get('render_ms', 0):.1f} ms.")
        try:
            items = self.sim.hud()
        except Exception:
            items = []
        rows = "".join(
            f"<tr><td style='color:#8b98a9; padding-right:14px'>{i.label}</td>"
            f"<td style='color:{'#fbbf24' if i.accent else '#e6edf3'}; font-weight:600'>"
            f"{i.value}</td></tr>" for i in items)
        paused = ("<div style='color:#fbbf24; font-weight:700'>PAUSED</div>"
                  if self.viewport.clock.paused else "")
        self.hud.setText(f"{paused}<table>{rows}</table>")
        self.hud.adjustSize()
        self.hud.move(16, 16)
        self.hud.show()
        self.hud.raise_()
        if self._live_ok and self.guide.live_visible:
            try:
                self.guide.live.update_values(self.sim.live_math())
            except Exception:
                traceback.print_exc()
                self._live_ok = False
        self._check_experiments()
        for key, cb in getattr(self, "_ov_boxes", []):   # keys can flip overlays (e.g. V)
            on = bool(self.sim.show.get(key))
            if cb.isChecked() != on:
                cb.blockSignals(True)
                cb.setChecked(on)
                cb.blockSignals(False)
                self._overlay(key, on)

    # ------------------------------------------------------------ lab furniture
    def _setup_lab(self, chapter: ChapterInfo) -> None:
        sim = self.sim
        self.hotbar.set_tools(sim.TOOLS)
        keys = [t.key for t in sim.TOOLS]
        self.set_tool(self.tool_key if self.tool_key in keys else (keys[0] if keys else ""))
        prog = settings.load().get("progress") or {}
        self.done = set(prog.get(chapter.id, [])) & {e.key for e in sim.EXPERIMENTS}
        app_keys = [("1 – 9", "Pick a tool") if len(sim.TOOLS) > 1 else None,
                    ("Ctrl+1 – 9" if len(sim.TOOLS) > 1 else "1 – 9", "Toggle overlays"),
                    ("Space  .  R", "Pause, step, restart"),
                    ("Wheel, middle-drag", "Zoom, pan"),
                    ("Tab", "Focus mode"), ("F11", "Full screen")]
        self.guide.set_chapter(list(chapter.controls) + [k for k in app_keys if k],
                               chapter.reality, sim.EXPERIMENTS, self.done)
        self._place_overlays()

    def set_tool(self, key: str) -> None:
        tools = {t.key: t for t in (self.sim.TOOLS if self.sim else [])}
        tool = tools.get(key)
        self.tool_key = key if tool else ""
        self.viewport.set_tool(tool)
        self.hotbar.set_active(key)
        self.guide.set_tool(tool)
        if self.viewport.isVisible():      # focusing a hidden tab's child would switch tabs
            self.viewport.setFocus()

    def _check_experiments(self) -> None:
        if not self.sim or not self.chapter:
            return
        for e in self.sim.EXPERIMENTS:
            if e.check and e.key not in self.done:
                try:
                    ok = bool(getattr(self.sim, e.check)())
                except Exception:
                    ok = False
                if ok:
                    self._experiment_toggled(e.key, True, announce=True)

    def _experiment_toggled(self, key: str, done: bool, announce: bool = False) -> None:
        if not self.chapter:
            return
        (self.done.add if done else self.done.discard)(key)
        prog = dict(settings.load().get("progress") or {})
        prog[self.chapter.id] = sorted(self.done)
        settings.save({"progress": prog})
        self.guide.set_done(key, done)
        exp = next((e for e in self.sim.EXPERIMENTS if e.key == key), None) if self.sim else None
        if announce and exp and self.guide.goals.opened:   # closed goals tick quietly
            learn = (f"<br><span style='color:#99f6e4'>{exp.learn}</span>" if exp.learn
                     else "")
            self.toast.show_message(f"<b>✓ Goal complete: {exp.title}</b>{learn}", 7000)

    def set_focus_mode(self, on: bool) -> None:
        if on == self.focus_mode:
            return
        self.focus_mode = on
        if self.compact:
            self.guide_drawer.close_drawer(animate=False)
            self.insp_drawer.close_drawer(animate=False)
        elif on:
            self.guide.hide()
            self.inspector.hide()
        else:
            self.guide.setVisible(self.guide_btn.isChecked())
            self.inspector.setVisible(self.insp_btn.isChecked())
        self.header.setVisible(not on)
        self.tabs.tabBar().setVisible(not on)
        self.lab_toolbar.setVisible(not on)
        self.focusMode.emit(on)
        if on:
            self.toast.show_message("Focus mode · <b>Tab</b> or <b>Esc</b> to come back", 2200)
        self.viewport.setFocus()

    def _place_overlays(self) -> None:
        w, h = self.view_host.width(), self.view_host.height()
        self.hotbar.set_compact(w < 720)
        self.hotbar.adjustSize()
        self.hotbar.move(max(8, (w - self.hotbar.width()) // 2), h - self.hotbar.height() - 16)
        self.hotbar.raise_()
        self.toast.reposition()

    def eventFilter(self, obj, e) -> bool:
        if obj is self.view_host and e.type() in (QEvent.Resize, QEvent.Show):
            self._place_overlays()
        return super().eventFilter(obj, e)

    def _failed(self, msg: str) -> None:
        self.banner.setText(f"<b>Something went wrong</b><br><span style='color:#f87171'>"
                            f"{msg}</span><br><span style='color:#8b98a9'>Press R or Restart "
                            "to try again.</span>")
        self.banner.show()
        if self.sim and self.viewport.sim is None:
            self.viewport.set_sim(self.sim)


def _precompile(cls, device: str) -> None:
    """Compile a chapter's Warp kernels (and the shared memory kernels) up front."""
    try:
        import warp as wp

        from ..core import memory

        wp.load_module(sys.modules[cls.__module__], device=device)
        wp.load_module(memory, device=device)
    except Exception:
        pass  # kernels will compile lazily on first launch instead


# ================================================================= system info
class SystemDialog(QDialog):
    deviceChanged = Signal(str)

    def __init__(self, info: SystemInfo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Your machine")
        self.setMinimumWidth(640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(12)
        t = QLabel("What this computer brings to the lab")
        t.setProperty("role", "h2")
        lay.addWidget(t)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        for r, (k, v) in enumerate(info.summary_lines()):
            kl = QLabel(k)
            kl.setProperty("role", "muted")
            vl = QLabel(v)
            vl.setWordWrap(True)
            vl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(kl, r, 0, Qt.AlignTop)
            grid.addWidget(vl, r, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        for n in info.advice():
            nl = QLabel("• " + n)
            nl.setWordWrap(True)
            nl.setProperty("role", "muted")
            lay.addWidget(nl)

        lay.addSpacing(6)
        h = QLabel("COMPUTE BACKEND")
        h.setProperty("role", "section")
        lay.addWidget(h)
        row = QHBoxLayout()
        self.group = QButtonGroup(self)
        current = settings.load().get("device", "auto")
        for key, label in (("auto", "Automatic"), ("cuda", "CUDA GPU"), ("cpu", "CPU only")):
            rb = QRadioButton(label)
            rb.setEnabled(key != "cuda" or info.cuda_gpu is not None)
            rb.setChecked(key == current)
            rb.toggled.connect(lambda on, k=key: on and self._set_device(k))
            self.group.addButton(rb)
            row.addWidget(rb)
        row.addStretch(1)
        lay.addLayout(row)
        tip = QLabel("Switching reloads the open chapter. Try CPU only to feel why GPUs matter.")
        tip.setProperty("role", "faint")
        lay.addWidget(tip)
        if info.is_hybrid and sys.platform.startswith("linux"):
            nv = QCheckBox("Draw on the NVIDIA GPU too (PRIME offload, applies after restart)")
            nv.setChecked(settings.load().get("render_gpu") == "nvidia")
            nv.toggled.connect(lambda on: settings.save(
                {"render_gpu": "nvidia" if on else "system"}))
            lay.addWidget(nv)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close, 0, Qt.AlignRight)

    def _set_device(self, key: str) -> None:
        settings.save({"device": key})
        import warp as wp

        dev = "cuda:0" if key in ("auto", "cuda") and wp.is_cuda_available() else "cpu"
        compute.set_device(dev)
        self.deviceChanged.emit(dev)


# ================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self, info: SystemInfo, catalog: Catalog, seed: int | None = None):
        super().__init__()
        self.info, self.catalog = info, catalog
        self.setWindowTitle("Algorithmic Intelligence Lab")
        for t in catalog.tracks:
            for c in t.chapters:
                c._color = t.color
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.sidebar = Sidebar(catalog, info)
        self.pages = QStackedWidget()
        self.home = HomePage(catalog, info)
        self.chapter = ChapterPage(info, seed)
        self.pages.addWidget(self.home)
        self.pages.addWidget(self.chapter)
        h.addWidget(self.sidebar)
        h.addWidget(self.pages, 1)
        self.sidebar.chapterChosen.connect(self.open_chapter)
        self.sidebar.homeRequested.connect(self.go_home)
        self.sidebar.systemRequested.connect(self.show_system)
        self.home.chapterChosen.connect(self.open_chapter)
        self.chapter.viewport.glReady.connect(self._gl_ready)
        self.chapter.statusText.connect(lambda s: self.status_left.setText(s))
        self.chapter.focusMode.connect(self._focus_mode)
        self.chapter.tabs.currentChanged.connect(lambda _: self._fit_panels())
        self._focus = False
        self.sidebar_drawer = Drawer(central, "left", 300)
        self._docked = True
        for b in (self.home.menu, self.chapter.menu):
            b.clicked.connect(self.sidebar_drawer.toggle)
        self.sidebar.chapterChosen.connect(lambda _: self.sidebar_drawer.close_drawer())
        self.sidebar.homeRequested.connect(lambda: self.sidebar_drawer.close_drawer())

        sb = self.statusBar()
        self.status_left = QLabel("Ready")
        self.status_right = QLabel()
        sb.addWidget(self.status_left, 1)
        sb.addPermanentWidget(self.status_right)
        self._refresh_status()
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.sidebar.search.setFocus)
        self.home.set_gl(info)
        QTimer.singleShot(400, self.home.start_thumbnails)

    def _refresh_status(self) -> None:
        i = self.info
        render = i.gl_renderer.split("/")[0] if i.gl_renderer else "…"
        self.status_right.setText(f"compute: {i.compute_label}   ·   draw: {render}   ·   "
                                  f"{i.cpu_threads} CPU threads")

    def _gl_ready(self, gl: dict) -> None:
        self.info.gl_renderer = gl.get("renderer", self.info.gl_renderer)
        self.info.gl_vendor = gl.get("vendor", self.info.gl_vendor)
        self.info.gl_version = gl.get("version", self.info.gl_version)
        self.sidebar.refresh_system(self.info)
        self.home.set_gl(self.info)
        self._refresh_status()

    def open_chapter(self, cid: str) -> None:
        c = self.catalog.chapters.get(cid)
        if not c:
            return
        self.sidebar.select(cid)
        self.pages.setCurrentWidget(self.chapter)
        self.chapter.load(c)
        self._fit_panels()
        settings.save({"last_chapter": cid})

    def go_home(self) -> None:
        self.chapter.set_focus_mode(False)
        self.chapter.viewport.set_sim(None)
        self.pages.setCurrentWidget(self.home)
        self.status_left.setText("Home")
        self._fit_panels()

    def show_system(self) -> None:
        dlg = SystemDialog(self.info, self)
        dlg.deviceChanged.connect(self._device_changed)
        dlg.exec()

    def _device_changed(self, dev: str) -> None:
        self.info.compute_device = dev
        self.sidebar.refresh_system(self.info)
        self._refresh_status()
        if self.chapter.chapter and self.pages.currentWidget() is self.chapter:
            self.chapter.load(self.chapter.chapter, keep_tab=True)

    # ------------------------------------------------------------ screen fitting
    def show_fitted(self) -> None:
        """Size the window to the screen it opens on: never bigger than the screen,
        maximised on laptops, centred otherwise."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.setMinimumSize(min(960, avail.width()), min(600, avail.height()))
        if avail.width() < 1500 or avail.height() < 880:
            self.showMaximized()
        else:
            w = min(1560, int(avail.width() * 0.9))
            h = min(940, int(avail.height() * 0.9))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2)
            self.show()
        self._fit_panels()

    def _fit_panels(self) -> None:
        """Like a responsive web page: on wide windows the chapter list sits beside the
        content; on narrow ones (or in the Lab below ~1650 px) it becomes a ☰ drawer."""
        if not hasattr(self, "sidebar_drawer"):
            return
        w = self.width()
        in_lab = self.pages.currentWidget() is self.chapter and self.chapter.tabs.currentIndex() == 1
        docked = not self._focus and (w >= 1650 or (w >= 1100 and not in_lab))
        self.sidebar.setFixedWidth(300 if w >= 1700 else 260)
        if docked and not self._docked:
            self.sidebar_drawer.give()
            self.centralWidget().layout().insertWidget(0, self.sidebar)
            self.sidebar.show()
        elif not docked and self._docked:
            self.centralWidget().layout().removeWidget(self.sidebar)
            self.sidebar_drawer.take(self.sidebar)
        self._docked = docked
        for b in (self.home.menu, self.chapter.menu):
            b.setVisible(not docked and not self._focus)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_panels()

    def _focus_mode(self, on: bool) -> None:
        self._focus = on
        self.statusBar().setVisible(not on)
        self._fit_panels()

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, e) -> None:
        self.chapter.viewport.set_sim(None)
        super().closeEvent(e)

