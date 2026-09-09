"""Vector PDF writer: to-scale, tiled across pages when the sheet outgrows the paper.

``QPdfWriter`` plus the same :func:`~studio.ui.render.paint_scene` the preview
uses, so the print is crisp at any size with no intermediate raster. Each tile
from :func:`studio.pdf_export.plan_tiles` becomes one page painted at exact
physical scale, with corner crop marks and a "row r / col c" label in the page
margin so the sheets assemble unambiguously. A sheet that fits the paper gets a
single clean page with no marks.

This is what makes a mosaic grid printable at true tessera size: a 32-tile grid
at 12 mm is 38 cm across, so it lands on four A4 sheets you tape together and
paint from.
"""
from __future__ import annotations

from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen

from ..pdf_export import MM_PER_INCH, PAPER_SIZES_IN, plan_tiles
from ..scene import Scene
from .render import paint_scene

RES_DPI = 600  # QPdfWriter device resolution; every device measure derives from this


def write_tiled_pdf(
    scene: Scene,
    path: str,
    paper: str = "A4",
    landscape: bool = False,
    page_margin_mm: float = 10.0,
    overlap_mm: float = 10.0,
) -> tuple:
    """Write ``scene`` to ``path`` at its physical scale. Returns (pages, rows, cols)."""
    ipp = float(scene.inches_per_px)
    if ipp <= 0:
        raise ValueError("A to-scale PDF needs a physical scale (set a tessera size).")
    pw_in, ph_in = PAPER_SIZES_IN.get(paper, PAPER_SIZES_IN["A4"])
    if landscape:
        pw_in, ph_in = ph_in, pw_in
    margin_in = max(0.2, float(page_margin_mm) / MM_PER_INCH)
    overlap_in = max(0.0, float(overlap_mm) / MM_PER_INCH)
    printable_w = pw_in - 2 * margin_in
    printable_h = ph_in - 2 * margin_in

    m = max(0, int(scene.margin))
    tiles, rows, cols = plan_tiles(
        scene.width, scene.height, m, ipp, printable_w, printable_h, overlap_in
    )

    pdf = QPdfWriter(path)
    pdf.setResolution(RES_DPI)  # default is 1200 — set explicitly, derive everything
    size = QPageSize(QPageSize.PageSizeId.A4)
    if paper == "Letter":
        size = QPageSize(QPageSize.PageSizeId.Letter)
    elif paper == "Legal":
        size = QPageSize(QPageSize.PageSizeId.Legal)
    orient = (QPageLayout.Orientation.Landscape if landscape
              else QPageLayout.Orientation.Portrait)
    pdf.setPageLayout(QPageLayout(size, orient, QMarginsF(0, 0, 0, 0)))

    # Device-pixel geometry (RES_DPI dots per inch).
    ox, oy = margin_in * RES_DPI, margin_in * RES_DPI       # printable origin
    tw, th = printable_w * RES_DPI, printable_h * RES_DPI   # printable size
    scale = ipp * RES_DPI                                   # device px per scene px
    single = len(tiles) == 1

    painter = QPainter(pdf)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        for i, (r, c, sx, sy) in enumerate(tiles):
            if i:
                pdf.newPage()
            painter.save()
            if not single:
                _page_furniture(painter, ox, oy, tw, th, r, c, rows, cols, i, len(tiles))
            painter.setClipRect(QRectF(ox, oy, tw, th))
            painter.translate(ox, oy)
            painter.scale(scale, scale)
            painter.translate(-sx, -sy)
            paint_scene(painter, scene)
            painter.restore()
    finally:
        painter.end()
    return len(tiles), rows, cols


def write_scale_pdf(scene: Scene, path: str) -> tuple:
    """One page, exactly the scene's physical size, painted as vectors.

    The single-page counterpart of :func:`write_tiled_pdf`, for documents whose
    size the artist set directly (a board or a plan at its card width). Same
    page geometry as the old raster path it replaces — the difference is that
    text stays text and swatch edges stay crisp at any zoom. Returns the page
    size in inches.
    """
    ipp = float(scene.inches_per_px)
    if ipp <= 0:
        raise ValueError("A to-scale PDF needs a physical scale (set a card width).")
    m = max(0, int(scene.margin))
    w_in = (scene.width + 2 * m) * ipp
    h_in = (scene.height + 2 * m) * ipp

    pdf = QPdfWriter(path)
    pdf.setResolution(RES_DPI)
    pdf.setPageLayout(QPageLayout(
        QPageSize(QSizeF(w_in, h_in), QPageSize.Unit.Inch),
        QPageLayout.Orientation.Portrait,
        QMarginsF(0, 0, 0, 0),
    ))
    painter = QPainter(pdf)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.scale(ipp * RES_DPI, ipp * RES_DPI)
        painter.translate(m, m)
        paint_scene(painter, scene)
    finally:
        painter.end()
    return w_in, h_in


def _page_furniture(painter, ox, oy, tw, th, r, c, rows, cols, i, n):
    """Corner crop marks at the printable rect + the assembly label in the margin."""
    pen = QPen(Qt.GlobalColor.black, RES_DPI / 300.0)
    painter.setPen(pen)
    L = RES_DPI * 0.14  # mark length ~3.5 mm
    for cx, cy, dx, dy in (
        (ox, oy, 1, 1), (ox + tw, oy, -1, 1),
        (ox, oy + th, 1, -1), (ox + tw, oy + th, -1, -1),
    ):
        painter.drawLine(int(cx), int(cy), int(cx + dx * L), int(cy))
        painter.drawLine(int(cx), int(cy), int(cx), int(cy + dy * L))
    font = QFont()
    font.setPixelSize(int(RES_DPI * 0.11))  # ~8 pt
    painter.setFont(font)
    painter.drawText(
        QRectF(ox, oy + th + RES_DPI * 0.02, tw, RES_DPI * 0.2),
        Qt.AlignmentFlag.AlignHCenter,
        f"row {r + 1} / col {c + 1}   —   page {i + 1} of {n} ({rows}×{cols})",
    )
