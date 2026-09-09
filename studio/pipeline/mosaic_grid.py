"""Mosaic stage A — cut the photo into tesserae and reduce them to a few colours.

Two layouts, both producing the same thing: a flat list of :class:`Tessera`,
each a four-cornered footprint in normalized image coordinates carrying a
colour index. Everything downstream — the preview, the cartoon, the to-scale
sheet — works off that list and never needs to know which layout made it.

* ``"flow"`` (the default) lays the tesserae along the figure's contours, the
  way a mosaicist does. See :mod:`studio.pipeline.mosaic_flow`. This is what
  makes the mosaic read as a picture with the colour taken away.
* ``"grid"`` keeps them on a plain lattice — easier to rule onto paper and
  address by row and column, but it is a pixel grid, not a mosaic.

Colour reduction is shared. Pixels are clustered in LAB and the clusters come
out ordered **lightest first**, so colour 1 is the palest and painting in
number order is also painting light to dark.

Deterministic throughout: seeded k-means, fixed iteration counts, a stable
sort, explicit tie-breaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..color import lab_to_rgb, rgb_to_hex, rgb_to_lab
from .extract import kmeans
from .mosaic_flow import (
    ANGLE_SMOOTH_FRAC,
    FLOW_EDGE,
    SMOOTH_FRAC,
    contour_tiers,
    course_angles,
    distance_field,
    place_tesserae,
)

MIN_TILES = 8
MAX_TILES = 60
MIN_COLORS = 2
MAX_COLORS = 20
LAYOUT_FLOW = "flow"
LAYOUT_GRID = "grid"
LAYOUTS = (LAYOUT_FLOW, LAYOUT_GRID)
_FIT_SAMPLES = 20000


@dataclass(frozen=True)
class Tessera:
    """One stone. ``points`` is its *footprint*, in normalized image coords.

    The footprint is the full space the tessera reserves; the grout is taken
    out of it at compose time, which is why grout and irregularity stay
    cosmetic and recompute nothing.
    """

    points: Tuple[Tuple[float, float], ...]
    color: int
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class MosaicGrid:
    """Stage output: the stones, and the colour stock they are cut from."""

    cols: int
    rows: int
    layout: str
    colors: Tuple[str, ...]           # target hex per colour index, lightest first
    counts: Tuple[int, ...]           # tesserae per colour index
    tesserae: Tuple[Tessera, ...]
    cells: Optional[np.ndarray] = None  # grid layout only: (rows, cols) colour indices

    @property
    def total(self) -> int:
        return len(self.tesserae)

    def fraction(self, index: int) -> float:
        return self.counts[index] / float(max(1, self.total))


def grid_shape(img_w: int, img_h: int, tiles_x: int) -> Tuple[int, int]:
    """(cols, rows) for a photo, keeping its aspect ratio."""
    cols = max(MIN_TILES, min(MAX_TILES, int(tiles_x)))
    rows = max(1, int(round(cols * img_h / float(max(1, img_w)))))
    return cols, rows


# --- colour reduction -----------------------------------------------------------
def _quantize(lab: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster LAB pixels into ``k`` colours, lightest first.

    Fitted on a strided subsample and then applied to every pixel — the full
    distance matrix over a whole image would be hundreds of megabytes of
    temporaries for no extra accuracy.
    """
    flat = lab.reshape(-1, 3)
    stride = max(1, len(flat) // _FIT_SAMPLES)
    centers, _labels = kmeans(flat[::stride], k)
    order = sorted(range(len(centers)), key=lambda i: (-centers[i][0], i))
    centers = centers[order]
    # Assign every pixel to its nearest centre, one centre at a time.
    best = np.full(len(flat), np.inf)
    labels = np.zeros(len(flat), dtype=np.int32)
    for i, c in enumerate(centers):
        d = np.sum((flat - c) ** 2, axis=1)
        closer = d < best
        best = np.where(closer, d, best)
        labels = np.where(closer, i, labels)
    return centers, labels.reshape(lab.shape[:2])


def _hexes(centers: np.ndarray) -> Tuple[str, ...]:
    return tuple(
        rgb_to_hex(tuple(int(c) for c in lab_to_rgb(centers[i]))) for i in range(len(centers))
    )


def _box_average(rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean colour of the source pixels inside each of ``rows × cols`` cells."""
    h, w = rgb.shape[:2]
    if rows >= h or cols >= w:
        ys = np.clip((np.arange(rows) + 0.5) * h / rows, 0, h - 1).astype(int)
        xs = np.clip((np.arange(cols) + 0.5) * w / cols, 0, w - 1).astype(int)
        return rgb[np.ix_(ys, xs)].astype(np.float64)
    ys = np.unique(np.linspace(0, h, rows + 1).astype(int))
    xs = np.unique(np.linspace(0, w, cols + 1).astype(int))
    out = np.add.reduceat(rgb.astype(np.float64), ys[:-1], axis=0)
    out /= np.diff(ys).reshape(-1, 1, 1)
    out = np.add.reduceat(out, xs[:-1], axis=1)
    out /= np.diff(xs).reshape(1, -1, 1)
    return out


def _resize(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return rgb.astype(np.float64)
    scale = max_edge / float(longest)
    return _box_average(rgb, max(1, int(round(h * scale))), max(1, int(round(w * scale))))


# --- the two layouts --------------------------------------------------------------
def _lattice_tesserae(cells: np.ndarray) -> Tuple[Tessera, ...]:
    """Plain grid: every footprint is one lattice cell."""
    rows, cols = cells.shape
    out: List[Tessera] = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c / cols, (c + 1) / cols
            y0, y1 = r / rows, (r + 1) / rows
            out.append(Tessera(
                points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                color=int(cells[r, c]),
                cx=(x0 + x1) / 2.0,
                cy=(y0 + y1) / 2.0,
            ))
    return tuple(out)


def build_mosaic_grid(
    rgb: np.ndarray, tiles_x: int, mosaic_colors: int, layout: str = LAYOUT_FLOW
) -> MosaicGrid:
    """Reduce ``rgb`` to tesserae drawn from a small colour stock."""
    ih, iw = rgb.shape[:2]
    cols, rows = grid_shape(iw, ih, tiles_x)
    k = max(MIN_COLORS, min(int(mosaic_colors), MAX_COLORS, rows * cols))

    if layout == LAYOUT_GRID:
        cell_rgb = _box_average(rgb, rows, cols)
        centers, cells = _quantize(rgb_to_lab(cell_rgb), k)
        tesserae = _lattice_tesserae(cells)
    else:
        work = _resize(rgb, FLOW_EDGE)
        centers, labels = _quantize(rgb_to_lab(work), k)
        size_px = work.shape[1] / float(cols)
        smooth = max(1, int(round(size_px * SMOOTH_FRAC)))
        tier = contour_tiers(work, smooth)
        dist = distance_field(tier)
        angles = course_angles(dist, max(1, int(round(size_px * ANGLE_SMOOTH_FRAC))))
        placed = place_tesserae(dist, angles, labels, k, size_px)
        tesserae = tuple(
            Tessera(points=pts, color=col, cx=cx, cy=cy) for pts, col, cx, cy in placed
        )
        cells = None

    counts = np.bincount([t.color for t in tesserae], minlength=k) if tesserae else np.zeros(k, int)
    return MosaicGrid(
        cols=cols,
        rows=rows,
        layout=LAYOUT_GRID if layout == LAYOUT_GRID else LAYOUT_FLOW,
        colors=_hexes(centers),
        counts=tuple(int(c) for c in counts),
        tesserae=tesserae,
        cells=cells,
    )
