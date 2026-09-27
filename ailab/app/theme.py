"""Visual identity for the native UI: one dark theme, track accent colours, Inter font."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

BG0 = "#070b12"      # window
BG1 = "#0c121c"      # sidebar / panels
BG2 = "#111a27"      # cards
BG3 = "#172233"      # hover / inputs
LINE = "#1e2a3b"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
FAINT = "#5b6778"
ACCENT = "#38bdf8"
GOOD = "#34d399"
WARN = "#fbbf24"
BAD = "#f87171"

QSS = f"""
* {{ outline: none; }}
QWidget {{ background: {BG0}; color: {TEXT}; font-size: 13px; }}
QToolTip {{ background: {BG3}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px 8px; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="faint"] {{ color: {FAINT}; font-size: 11px; }}
QLabel[role="h1"] {{ font-size: 30px; font-weight: 700; }}
QLabel[role="h2"] {{ font-size: 18px; font-weight: 600; }}
QLabel[role="section"] {{ color: {MUTED}; font-size: 11px; font-weight: 600;
                          letter-spacing: 1.5px; }}
QFrame#sidebar {{ background: {BG1}; border-right: 1px solid {LINE}; }}
QFrame#panel {{ background: {BG1}; border-left: 1px solid {LINE}; }}
QFrame#card {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#card:hover {{ border: 1px solid #2f4460; background: #131e2d; }}
QFrame#stat {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#toolbar {{ background: {BG1}; border-bottom: 1px solid {LINE}; }}
QLineEdit, QSpinBox, QComboBox {{
    background: {BG3}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px 10px;
    selection-background-color: #1d4ed8; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid #2f5f8a; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {BG2}; border: 1px solid {LINE};
                               selection-background-color: {BG3}; }}
QPushButton {{ background: {BG3}; border: 1px solid {LINE}; border-radius: 8px;
               padding: 7px 14px; }}
QPushButton:hover {{ background: #1c2a3e; border-color: #2f4460; }}
QPushButton:pressed {{ background: #0f1826; }}
QPushButton:checked {{ background: #12324a; border-color: {ACCENT}; color: #e0f2fe; }}
QPushButton[role="primary"] {{ background: #0e7490; border-color: #22d3ee; color: white;
                               font-weight: 600; padding: 10px 22px; }}
QPushButton[role="primary"]:hover {{ background: #0891b2; }}
QPushButton[role="seg"] {{ border-radius: 0; padding: 7px 16px; }}
QPushButton[role="ghost"] {{ background: transparent; border: none; color: {MUTED};
                             padding: 6px 10px; }}
QPushButton[role="ghost"]:hover {{ color: {TEXT}; background: {BG3}; }}
QTreeWidget {{ background: transparent; border: none; }}
QTreeWidget::item {{ padding: 6px 4px; border-radius: 6px; }}
QTreeWidget::item:hover {{ background: {BG3}; }}
QTreeWidget::item:selected {{ background: #12324a; color: white; }}
QTreeWidget::branch {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #243246; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #0e7490; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e0f2fe; width: 14px; height: 14px;
                              margin: -5px 0; border-radius: 7px; }}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
                        border: 1px solid #2f4460; background: {BG3}; }}
QCheckBox::indicator:checked {{ background: #0e7490; border-color: #22d3ee; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 10px 18px;
                border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTextBrowser {{ background: {BG0}; border: none; }}
QStatusBar {{ background: {BG1}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QDialog {{ background: {BG1}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG0), (QPalette.Base, BG3), (QPalette.Text, TEXT),
                        (QPalette.WindowText, TEXT), (QPalette.Button, BG3),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, "#1d4ed8"),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.Link, ACCENT),
                        (QPalette.ToolTipBase, BG3), (QPalette.ToolTipText, TEXT)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    families = set(QFontDatabase.families())
    for fam in ("Inter", "Inter Variable", "Ubuntu Sans", "Cantarell", "Noto Sans",
                "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue"):
        if fam in families:
            f = QFont(fam, 10)
            f.setHintingPreference(QFont.PreferNoHinting)
            app.setFont(f)
            break
    app.setStyleSheet(QSS + QSS_LAB)


def mono_family() -> str:
    families = set(QFontDatabase.families())
    for fam in ("JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Noto Sans Mono",
                "Cascadia Mono", "Consolas", "Menlo"):
        if fam in families:
            return fam
    return "monospace"


def dots(n: int, total: int = 5, color: str = ACCENT) -> str:
    """Difficulty as rich text dots."""
    return (f"<span style='color:{color}'>{'●' * n}</span>"
            f"<span style='color:{LINE}'>{'●' * (total - n)}</span>")


QSS_LAB = f"""
QFrame#hotbar {{ background: rgba(8,12,20,215); border: 1px solid #243246; border-radius: 14px; }}
QToolButton[role="tool"] {{ background: transparent; border: 1px solid transparent;
    border-radius: 10px; padding: 4px 6px; color: {MUTED}; font-size: 11px; font-weight: 600; }}
QToolButton[role="tool"]:hover {{ background: #16233a; color: {TEXT}; }}
QToolButton[role="tool"]:checked {{ background: #12324a; border-color: {ACCENT}; color: #e0f2fe; }}
QFrame#drawer {{ background: {BG1}; border: 1px solid #243246; }}
QFrame#guide {{ background: {BG1}; border-right: 1px solid {LINE}; }}
QFrame#mathcard {{ background: #0a1320; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#toolcard {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#exp {{ background: transparent; border: 1px solid transparent; border-radius: 10px; }}
QFrame#exp:hover {{ background: {BG2}; border-color: {LINE}; }}
QFrame#exp[done="true"] {{ background: #0c2320; border-color: #134e4a; }}
QFrame#nature {{ background: #151a12; border: 1px solid #2a3320; border-radius: 10px; }}
QLabel#toast {{ background: rgba(6,40,36,235); border: 1px solid #14b8a6; border-radius: 12px;
    padding: 12px 18px; color: #ccfbf1; font-size: 13px; }}
QPushButton#fold {{ background: transparent; border: none; border-radius: 6px; text-align: left;
    padding: 5px 2px; color: {MUTED}; }}
QPushButton#fold:hover {{ color: {TEXT}; background: {BG2}; }}
QPushButton#fold[fold="section"] {{ font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
QPushButton#fold[fold="group"] {{ font-size: 13px; font-weight: 600; color: #cbd5e1;
    border-top: 1px solid {LINE}; border-radius: 0; padding: 8px 2px 6px 2px; }}
QPushButton#fold[fold="group"]:hover {{ color: #ffffff; }}
QPushButton[role="preset"] {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 5px 10px; font-size: 12px; color: #cbd5e1; }}
QPushButton[role="preset"]:hover {{ border-color: #2f4460; color: #ffffff; }}
QPushButton[role="preset"]:checked {{ background: #12324a; border-color: {ACCENT}; color: #e0f2fe; }}
QSplitter::handle {{ background: {LINE}; }}
QSplitter::handle:hover {{ background: #2f4460; }}
QProgressBar {{ background: {BG3}; border: none; border-radius: 3px; height: 6px; }}
QProgressBar::chunk {{ background: {GOOD}; border-radius: 3px; }}
"""
