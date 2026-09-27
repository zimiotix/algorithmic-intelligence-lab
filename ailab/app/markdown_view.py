"""Native rich-text view for chapter markdown, with crisp LaTeX equations."""

from __future__ import annotations

import html as _html
import re

from PySide6.QtCore import QRectF, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFontInfo, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QTextBrowser

from ..text import latex
from ..text.markdown import to_html
from . import theme

PT_TO_PX = {True: 1.8, False: 1.7}   # equation size relative to TeX points
EQ_COLOR = "#e6edf3"

CSS_TEMPLATE = """
body {{ color: #cfd8e3; font-size: 15px; font-family: '{body}'; font-weight: 400; }}
h1 {{ color: #f1f5f9; font-size: 26px; font-weight: 700; margin-top: 4px; margin-bottom: 10px; }}
h2 {{ color: #f1f5f9; font-size: 20px; font-weight: 650; margin-top: 26px; margin-bottom: 6px; }}
h3 {{ color: #e2e8f0; font-size: 16px; font-weight: 650; margin-top: 18px; }}
p, li {{ line-height: 155%; }}
p {{ margin-top: 6px; margin-bottom: 10px; }}
a {{ color: #38bdf8; text-decoration: none; }}
strong {{ color: #f8fafc; }}
em {{ color: #e2e8f0; }}
code {{ font-family: '{mono}'; color: #fcd34d; background-color: #161f2d; }}
pre {{ font-family: '{mono}'; background-color: #0e1622; color: #d6e2f0;
       padding: 12px; margin: 8px 0px; }}
blockquote {{ background-color: #0f2236; color: #d4e6f7; margin: 12px 0px; padding: 6px 14px; }}
table {{ margin: 8px 0px; }}
th {{ color: #f1f5f9; background-color: #131d2b; padding: 6px 12px; text-align: left; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #1e2a3b; }}
hr {{ color: #1e2a3b; }}
"""


def _css(widget) -> str:
    body = QFontInfo(widget.font()).family()
    return CSS_TEMPLATE.format(body=body, mono=theme.mono_family())


class MarkdownView(QTextBrowser):
    """Shows chapter markdown. Equations are rendered by TeX and cached as SVG."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open)
        self.document().setDefaultStyleSheet(_css(self))
        self.document().setDocumentMargin(28)
        self._images: dict[str, QImage] = {}

    def _open(self, url: QUrl) -> None:
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def loadResource(self, rtype: int, url: QUrl):
        if url.scheme() == "eq":
            return self._images.get(url.path() or url.toString()[3:])
        return super().loadResource(rtype, url)

    def set_markdown(self, text: str) -> None:
        dpr = self.devicePixelRatioF()
        # first pass: which equations are there?
        _, eqs = to_html(text, lambda e, i: "")
        paths = latex.render(eqs) if eqs else {}
        sizes: dict[str, QSize] = {}
        for e in eqs:
            p = paths.get(e.key)
            if p and e.key not in self._images:
                img, size = _rasterize(str(p), PT_TO_PX[e.display], dpr)
                if img is not None:
                    self._images[e.key] = img
                    sizes[e.key] = size
            elif e.key in self._images:
                img = self._images[e.key]
                sizes[e.key] = QSize(round(img.width() / dpr), round(img.height() / dpr))

        def eq_html(e, i):
            if e.key in sizes:
                s = sizes[e.key]
                img = (f'<img src="eq:{e.key}" width="{s.width()}" height="{s.height()}" '
                       f'style="vertical-align: middle">')
                return f'<p align="center">{img}</p>' if e.display else img
            src = _html.escape(e.tex)
            return (f'<p align="center"><code>{src}</code></p>' if e.display
                    else f"<code>{src}</code>")

        html, _ = to_html(text, eq_html)
        html = re.sub(r"<table>", '<table cellspacing="0" width="100%">', html)
        self.document().setDefaultStyleSheet(_css(self))
        self.setHtml(html)


def _rasterize(svg_path: str, scale: float, dpr: float):
    r = QSvgRenderer(svg_path)
    if not r.isValid():
        return None, None
    size = r.defaultSize()
    w, h = max(1, round(size.width() * scale)), max(1, round(size.height() * scale))
    img = QImage(round(w * dpr), round(h * dpr), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    img.setDevicePixelRatio(dpr)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, w, h))
    # Tint to the text colour: keeps the glyph shapes' alpha, replaces their colour.
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.fillRect(QRectF(0, 0, w, h), QColor(EQ_COLOR))
    p.end()
    return img, QSize(w, h)

