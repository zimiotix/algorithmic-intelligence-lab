"""The main window: sidebar catalogue, home page, chapter page (Overview / Lab / Deep dive)."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QShortcut
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
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import compute, settings
from ..core.catalog import Catalog, ChapterInfo, load_sim_class
from ..core.sim import SimContext
from ..core.system import SystemInfo
from ..text.markdown import split_sections
from . import theme
from .markdown_view import MarkdownView
from .viewport import Viewport
from .widgets import ChapterCard, Logo, ParamPanel, StatCard, chip, dot_pixmap

THUMBS = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ailab" / "thumbs"


def runs_on_label(info: SystemInfo) -> str:
    if info.on_gpu:
        return "CUDA · " + (info.compute_name or "GPU").replace("GeForce ", "")
    return "CPU"


def device_for(chapter: ChapterInfo) -> str:
    """Heavy chapters go to the GPU when there is one; light ones stay on the CPU."""
    return compute.device() if chapter.compute == "gpu" else "cpu"


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

        stats = QHBoxLayout()
        stats.setSpacing(12)
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
            stats.addWidget(s, 1)
        lay.addLayout(stats)
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
            grid = QGridLayout()
            grid.setSpacing(16)
            for i, c in enumerate(t.chapters):
                card = ChapterCard(c, t.color, label)
                card.clicked.connect(self.chapterChosen)
                grid.addWidget(card, i // 3, i % 3)
                self.cards[c.id] = card
            for col in range(3):
                grid.setColumnStretch(col, 1)
            lay.addLayout(grid)
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

    def __init__(self, info: SystemInfo):
        super().__init__()
        self.sysinfo = info
        self.chapter: ChapterInfo | None = None
        self.sim = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # --- header
        header = QFrame()
        header.setObjectName("toolbar")
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
        hl.addLayout(top)
        hl.addWidget(self.title)
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
        side.setObjectName("panel")
        side.setFixedWidth(340)
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

        # --- lab
        lab = QWidget()
        ll = QVBoxLayout(lab)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        ll.addWidget(self._toolbar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
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
        body.addWidget(self.view_host, 1)
        body.addWidget(self._side_panel())
        ll.addLayout(body, 1)
        self.tabs.addTab(lab, "Lab")

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
        self.seed.setRange(0, 999_999)
        self.seed.setValue(1)
        self.seed.setToolTip("The only source of randomness. Same seed = same run.")
        self.seed.editingFinished.connect(self.restart)
        dice = QPushButton("New seed")
        dice.clicked.connect(lambda: (self.seed.setValue((self.seed.value() * 7919 + 17)
                                                         % 1_000_000), self.restart()))
        tl.addWidget(sl)
        tl.addWidget(self.seed)
        tl.addWidget(dice)
        tl.addSpacing(10)
        self.speed_group = QButtonGroup(self)
        for i, s in enumerate((0.25, 0.5, 1.0, 2.0, 4.0)):
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
        self.perf = QLabel()
        self.perf.setProperty("role", "muted")
        self.perf.setStyleSheet(f"font-family:'{theme.mono_family()}'; font-size:12px;")
        tl.addWidget(self.perf)
        return bar

    def viewport_reset(self) -> None:
        self.viewport.camera.reset()

    def _side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(320)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setStyleSheet(f"background:{theme.BG1};")
        self.panel_layout = QVBoxLayout(inner)
        self.panel_layout.setContentsMargins(18, 16, 18, 18)
        self.panel_layout.setSpacing(12)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return panel

    def _rebuild_panel(self) -> None:
        lay = self.panel_layout
        while lay.count():
            it = lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        sim = self.sim
        if sim is None:
            return
        h = QLabel("SEE INSIDE")
        h.setProperty("role", "section")
        lay.addWidget(h)
        for i, o in enumerate(sim.OVERLAYS):
            cb = QCheckBox(o.label.replace("&", "&&") + (f"   [{i + 1}]" if i < 9 else ""))
            cb.setChecked(sim.show.get(o.key))
            cb.setToolTip(o.help)
            cb.toggled.connect(lambda v, k=o.key: self.sim and self.sim.show.set(k, v))
            cb.setObjectName(f"ov_{i + 1}")
            lay.addWidget(cb)
        lay.addSpacing(8)
        h = QLabel("PARAMETERS")
        h.setProperty("role", "section")
        lay.addWidget(h)
        note = QLabel("Symbols match the Deep dive. ⟲ restarts the run.")
        note.setProperty("role", "faint")
        lay.addWidget(note)
        panel = ParamPanel(sim.PARAMS, sim.p)
        panel.changed.connect(self._param)
        lay.addWidget(panel)
        reset = QPushButton("Default parameters")
        reset.setProperty("role", "ghost")
        reset.clicked.connect(lambda: self.load(self.chapter, keep_tab=True))
        lay.addWidget(reset)
        lay.addStretch(1)

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
        for key, action in chapter.controls + [("Space · . · R", "Pause · step · restart"),
                                               ("1 – 9", "Toggle overlays")]:
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
            self.deep.set_markdown(f"# {title}\n\n{body}")

    def _tab_changed(self, idx: int) -> None:
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
        self._rebuild_panel()
        self.statusText.emit(f"{chapter.title} · running on {dev}")

    # ---------------------------------------------------------------- actions
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

    def _hotkey(self, name: str) -> None:
        if name == "Space":
            self.toggle_pause()
        elif name == ".":
            self._step()
        elif name == "R":
            self.restart()
        elif name.isdigit():
            cb = self.findChild(QCheckBox, f"ov_{name}")
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
        self.perf.setText(f"{s.get('fps', 0):5.0f} fps · sim {s.get('compute_ms', 0):5.2f} ms "
                          f"[{dev}] · draw {s.get('render_ms', 0):4.1f} ms")
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
        if info.is_hybrid:
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
    def __init__(self, info: SystemInfo, catalog: Catalog):
        super().__init__()
        self.info, self.catalog = info, catalog
        self.setWindowTitle("Algorithmic Intelligence Lab")
        self.resize(1560, 940)
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
        self.chapter = ChapterPage(info)
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
        settings.save({"last_chapter": cid})

    def go_home(self) -> None:
        self.chapter.viewport.set_sim(None)
        self.pages.setCurrentWidget(self.home)
        self.status_left.setText("Home")

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

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, e) -> None:
        self.chapter.viewport.set_sim(None)
        super().closeEvent(e)

