"""To-scale PDF export: paper sizes and the multi-page tile plan.

``plan_tiles`` is the arithmetic: a mosaic grid printed at true tessera size is
routinely wider than any paper, so it has to span sheets that assemble 1:1.
Kept Qt-free so the smoke suite can verify the tile plan without a display; the
painting itself is done by ``studio.ui.pdf_print``, which writes vectors
straight onto a ``QPdfWriter`` — no intermediate raster anywhere.
"""
from __future__ import annotations

from math import ceil
from typing import List, Tuple

MM_PER_INCH = 25.4

# Printable paper sizes (full sheet, portrait, inches).
PAPER_SIZES_IN = {
    "A4": (8.27, 11.69),
    "Letter": (8.5, 11.0),
    "Legal": (8.5, 14.0),
}

Tile = Tuple[int, int, float, float]  # (row, col, scene_x, scene_y) of the tile origin


def plan_tiles(
    scene_w: int,
    scene_h: int,
    margin: int,
    ipp: float,
    printable_w_in: float,
    printable_h_in: float,
    overlap_in: float,
) -> Tuple[List[Tile], int, int]:
    """Tile the padded scene across pages of the given printable area.

    The padded page spans scene coords ``[-margin, dim+margin]`` (the same rect
    the raster/SVG exports cover), so the first tile starts at
    ``(-margin, -margin)``. Adjacent tiles share ``overlap_in`` of physical
    overlap — the tape band — so the stride is ``printable − overlap``. Returns
    ``(tiles, rows, cols)`` in row-major order; a sheet that fits yields the
    single ``(0, 0)`` tile.
    """
    ipp = float(ipp)
    if ipp <= 0:
        raise ValueError("plan_tiles needs a physical scale (inches per px > 0).")
    overlap_in = max(0.0, float(overlap_in))
    if printable_w_in <= overlap_in or printable_h_in <= overlap_in:
        raise ValueError("Paper printable area must exceed the tile overlap.")

    m = max(0, int(margin))
    total_w_in = (scene_w + 2 * m) * ipp
    total_h_in = (scene_h + 2 * m) * ipp
    stride_w = printable_w_in - overlap_in
    stride_h = printable_h_in - overlap_in
    cols = max(1, ceil((total_w_in - overlap_in) / stride_w))
    rows = max(1, ceil((total_h_in - overlap_in) / stride_h))

    tiles: List[Tile] = []
    for r in range(rows):
        for c in range(cols):
            sx = -m + (c * stride_w) / ipp
            sy = -m + (r * stride_h) / ipp
            tiles.append((r, c, sx, sy))
    return tiles, rows, cols
