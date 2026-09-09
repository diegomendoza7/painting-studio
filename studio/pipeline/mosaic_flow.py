"""Mosaic stage A2 — *andamento*: lay the tesserae in courses that follow the drawing.

A square lattice of tiles is a pixel grid, not a mosaic. What makes a Roman or
Byzantine floor read as a figure — even with every stone the same colour — is
that the courses **follow the drawing**: a line of tesserae hugs each contour
(*opus vermiculatum*), the next course stacks behind it, and each area fills
inward in bands that echo its own outline. The outlines alone carry the image.

Four steps:

1. **Find the drawing.** Not every colour change — a mosaicist follows the main
   contours. Luminance is blurred and cut into a few broad value bands (the
   same :func:`~studio.pipeline.plan_regions.value_tiers` the painting plan
   uses); the boundaries between those bands are the drawing.
2. **Grow a distance field inside each area**, seeded on its own outline and
   never propagating across it. Courses therefore stop at a contour instead of
   running through it, which is what a real floor does.
3. **Take the tangent** to that field. Its level sets are the courses, so a
   stone sits square-on to the contour when it aligns with the tangent.
4. **March along each course**, stepping one stone at a time, rather than
   filling by nearest-contour-first. This is the part that matters: ordering
   globally by distance hops around the picture and scatters the stones, while
   marching lays a continuous row the way a hand does.

Everything is deterministic: fixed iteration caps, raster-order seeding, and
explicit tie-breaks, so a photo always yields the same mosaic.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..color import rgb_to_lab
from .plan_regions import value_tiers

# Working resolution for the flow field. Contours, not detail.
FLOW_EDGE = 360
# Value bands whose boundaries count as "the drawing".
FLOW_BANDS = 4
# Blur radius (as a fraction of a stone) applied before banding, so photo noise
# cannot shatter a contour into speckle.
SMOOTH_FRAC = 0.6
# The distance field is blurred much harder than that before its gradient is
# taken. The angle field has to vary slowly or the courses come out short and
# swirly — long smooth arcs are what carry the drawing.
ANGLE_SMOOTH_FRAC = 2.0
_MAX_PROPAGATE = 512
_MAX_WALK = 4000            # guard: a course can never outrun the picture
# A stone may claim this fraction of a neighbour's footprint. Without a
# tolerance, two stones whose angles differ by a degree reject each other on a
# corner and the course opens a hole — packing drops by a third. The drawn stone
# is inset by the grout, so as long as the composer keeps the grout at or above
# this fraction (``MIN_GROUT``), overlapping footprints still leave clear grout
# between the stones actually painted.
OVERLAP_TOL = 0.10
MIN_GROUT = OVERLAP_TOL
# The first course is held back from its contour by this fraction of a stone, so
# an unbroken interstice runs along every edge in the drawing. This is the whole
# reason a mosaic reads with the colour taken away: without it the courses either
# side of an edge are both parallel to it and the fabric looks continuous, and
# the silhouette disappears. With it, the grout itself draws the figure.
CONTOUR_GAP = 0.10
# Stone sizes the filling passes try, as fractions of a full stone. A mosaicist
# cuts a smaller tessera for an awkward gap rather than leaving a hole, and
# without the small sizes any gap narrower than one stone stays bare forever —
# which reads as deliberate white shapes, not as grout.
FILL_SCALES = (1.0, 0.74, 0.55, 0.42)
Placed = Tuple[Tuple[Tuple[float, float], ...], int, float, float]


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums. No scipy, exact, O(n)."""
    r = max(0, int(radius))
    if r == 0:
        return a.astype(np.float64)
    out = a.astype(np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
        out = (hi - lo) / float(2 * r + 1)
    return out


def contour_tiers(rgb: np.ndarray, smooth_px: int, bands: int = FLOW_BANDS) -> np.ndarray:
    """The picture's broad value masses — "the drawing" the courses will follow."""
    lab_l = rgb_to_lab(rgb)[..., 0]
    smooth = _box_blur(lab_l, smooth_px)
    centers = value_tiers(smooth, bands)
    return np.argmin(np.abs(smooth[..., None] - centers[None, None, :]), axis=2).astype(np.int32)


def distance_field(tier: np.ndarray) -> np.ndarray:
    """Distance to the nearest contour, measured *within* each area.

    Seeded on every pixel that touches a different tier (or the picture's edge,
    which frames the outermost course the way a real panel does), and
    propagated only between pixels of the same tier. Confining it this way is
    what stops a band from straddling a contour and laying one row of stones
    down the middle of an edge.
    """
    h, w = tier.shape
    big = float(h + w)
    seed = np.zeros(tier.shape, dtype=bool)
    seed[:-1, :] |= tier[:-1, :] != tier[1:, :]
    seed[1:, :] |= tier[1:, :] != tier[:-1, :]
    seed[:, :-1] |= tier[:, :-1] != tier[:, 1:]
    seed[:, 1:] |= tier[:, 1:] != tier[:, :-1]
    seed[0, :] = seed[-1, :] = True
    seed[:, 0] = seed[:, -1] = True

    d = np.where(seed, 0.0, big)
    diag = math.sqrt(2.0)
    for _ in range(min(_MAX_PROPAGATE, int(big))):
        prev = d
        cand = [d]
        for dy, dx, wgt in (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
        ):
            shifted = np.roll(np.roll(d, dy, axis=0), dx, axis=1)
            same = np.roll(np.roll(tier, dy, axis=0), dx, axis=1) == tier
            if dy == -1:
                shifted[-1, :] = big
                same[-1, :] = False
            elif dy == 1:
                shifted[0, :] = big
                same[0, :] = False
            if dx == -1:
                shifted[:, -1] = big
                same[:, -1] = False
            elif dx == 1:
                shifted[:, 0] = big
                same[:, 0] = False
            cand.append(np.where(same, shifted + wgt, big))
        d = np.minimum.reduce(cand)
        if np.array_equal(d, prev):
            break
    return d


def course_angles(dist: np.ndarray, smooth_px: int) -> np.ndarray:
    """The direction a stone should lie in: tangent to the distance field.

    The field is blurred first — the gradient of a raw distance transform is
    noisy along its ridges, and a noisy tangent shows up as stones twitching
    out of line halfway down a course.
    """
    smooth = _box_blur(dist, max(1, smooth_px))
    gy, gx = np.gradient(smooth)
    ang = np.arctan2(gx, -gy)          # gradient rotated a quarter turn
    ang[(np.abs(gx) + np.abs(gy)) < 1e-9] = 0.0
    return ang


# --- marching placement -----------------------------------------------------------
class _Layer:
    """Occupancy plus the geometry helpers for one placement run."""

    def __init__(self, dist: np.ndarray, angles: np.ndarray, size_px: float):
        self.h, self.w = dist.shape
        self.dist = dist
        self.angles = angles
        self.half = max(1.0, float(size_px)) / 2.0
        self.occ = np.zeros(dist.shape, dtype=bool)
        self.ys, self.xs = np.mgrid[0:self.h, 0:self.w]

    def footprint(self, y: int, x: int, ang: float, half: Optional[float] = None):
        """(slice, mask) of the square this stone would occupy, or None if off-image."""
        hf = self.half if half is None else half
        reach = int(math.ceil(hf * math.sqrt(2.0))) + 1
        y0, y1 = max(0, y - reach), min(self.h, y + reach + 1)
        x0, x1 = max(0, x - reach), min(self.w, x + reach + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        ca, sa = math.cos(ang), math.sin(ang)
        dy = self.ys[y0:y1, x0:x1] - y
        dx = self.xs[y0:y1, x0:x1] - x
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        mask = (np.abs(u) <= hf) & (np.abs(v) <= hf)
        return (y0, y1, x0, x1), mask

    def free(self, y: int, x: int, ang: float, half: Optional[float] = None) -> bool:
        fp = self.footprint(y, x, ang, half)
        if fp is None:
            return False
        (y0, y1, x0, x1), mask = fp
        area = int(mask.sum())
        if area == 0:
            return False
        taken = int(self.occ[y0:y1, x0:x1][mask].sum())
        return taken <= OVERLAP_TOL * area

    def claim(self, y: int, x: int, ang: float, half: Optional[float] = None) -> None:
        (y0, y1, x0, x1), mask = self.footprint(y, x, ang, half)
        self.occ[y0:y1, x0:x1] |= mask

    @property
    def coverage(self) -> float:
        """Share of the picture the stones actually cover."""
        return float(self.occ.mean())


def _corners(x: int, y: int, ang: float, half: float, w: int, h: int):
    """The stone's footprint, clamped to the picture.

    A stone laid against the edge of the panel is cut flush with it, so a corner
    that would fall outside is pulled back to the border rather than the stone
    being refused — that is what keeps the outermost course tight to the frame.
    """
    ca, sa = math.cos(ang), math.sin(ang)
    return tuple(
        (min(1.0, max(0.0, (x + cu * ca - cv * sa) / w)),
         min(1.0, max(0.0, (y + cu * sa + cv * ca) / h)))
        for cu, cv in ((-half, -half), (half, -half), (half, half), (-half, half))
    )


def _snap(layer: _Layer, band: np.ndarray, b: int, iy: float, ix: float, radius: int):
    """The free pixel near ``(iy, ix)`` in band ``b`` whose stone would fit.

    Falls back to the neighbouring bands: a curving course drifts a band as the
    distance field changes under it, and insisting on an exact band match ends
    the walk right where the picture is most interesting.
    """
    y0, y1 = max(0, int(iy) - radius), min(layer.h, int(iy) + radius + 1)
    x0, x1 = max(0, int(ix) - radius), min(layer.w, int(ix) + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    sub_band = band[y0:y1, x0:x1]
    sub_occ = layer.occ[y0:y1, x0:x1]
    for tol in (0, 1):
        ok = (np.abs(sub_band - b) <= tol) & ~sub_occ
        if not ok.any():
            continue
        yy, xx = np.nonzero(ok)
        d2 = (yy + y0 - iy) ** 2 + (xx + x0 - ix) ** 2
        # Nearest first; ties by raster order so the walk is reproducible.
        for j in np.lexsort((xx, yy, d2))[:24]:
            cy, cx = int(yy[j]) + y0, int(xx[j]) + x0
            if layer.free(cy, cx, float(layer.angles[cy, cx])):
                return cy, cx
    return None


def _march(layer: _Layer, band: np.ndarray, b: int, start, step: float,
           out: List[Tuple[int, int, float, float]], sign: float) -> None:
    """Walk one course from ``start``, one stone at a time, until it runs out."""
    cy, cx = start
    prev: Optional[Tuple[float, float]] = None
    snap_r = max(2, int(round(step * 0.75)))
    for _ in range(_MAX_WALK):
        ang = float(layer.angles[cy, cx])
        tx, ty = math.cos(ang), math.sin(ang)
        if prev is None:
            tx, ty = tx * sign, ty * sign
        elif tx * prev[0] + ty * prev[1] < 0:
            tx, ty = -tx, -ty          # keep going the same way round the curve
        nxt = _snap(layer, band, b, cy + step * ty, cx + step * tx, snap_r)
        if nxt is None:
            return
        cy, cx = nxt
        ang = float(layer.angles[cy, cx])
        layer.claim(cy, cx, ang)
        out.append((cy, cx, ang, layer.half))
        prev = (tx, ty)


def place_tesserae(
    dist: np.ndarray,
    angles: np.ndarray,
    labels: np.ndarray,
    n_colors: int,
    size_px: float,
) -> List[Placed]:
    """Lay the tesserae course by course.

    Returns ``(corners, colour index, cx, cy)`` per stone, in normalized 0..1
    image coordinates. ``corners`` is the stone's full *footprint*; the grout is
    taken out of it at compose time, which keeps grout and irregularity
    cosmetic.
    """
    layer = _Layer(dist, angles, size_px)
    step = max(1.0, float(size_px))
    # Reserve the interstice along every contour before anything is laid, so no
    # stone can straddle an edge of the drawing.
    gap = CONTOUR_GAP * step
    layer.occ |= dist < gap
    band = np.maximum(0, np.floor((dist - gap) / step)).astype(np.int32)
    placed: List[Tuple[int, int, float, float]] = []

    # Band 0 hugs the contours; each next band stacks behind it.
    for b in range(int(band.max()) + 1):
        ys, xs = np.nonzero(band == b)
        if len(ys) == 0:
            continue
        for j in range(len(ys)):
            sy, sx = int(ys[j]), int(xs[j])
            if layer.occ[sy, sx]:
                continue
            ang = float(layer.angles[sy, sx])
            if not layer.free(sy, sx, ang):
                continue
            layer.claim(sy, sx, ang)
            placed.append((sy, sx, ang, layer.half))
            # Run the course out in both directions from the seed.
            _march(layer, band, b, (sy, sx), step, placed, 1.0)
            _march(layer, band, b, (sy, sx), step, placed, -1.0)

    # Fillers, cut smaller and smaller. A real floor has odd stones shaped to
    # whatever gap a course leaves; skipping this the mosaic ends up with bare
    # patches that read as deliberate white shapes rather than as grout.
    order = np.argsort(dist.ravel(), kind="stable").tolist()
    w_px = dist.shape[1]
    quarter = math.pi / 4.0
    for scale in FILL_SCALES:
        hf = layer.half * scale
        if hf < 1.0:
            continue
        for idx in order:
            fy, fx = divmod(idx, w_px)
            if layer.occ[fy, fx]:
                continue
            base = float(layer.angles[fy, fx])
            # A filler is cut to fit, so try it turned as well as square-on.
            for fang in (base, base + quarter, base - quarter, base + quarter / 2.0):
                if layer.free(fy, fx, fang, hf):
                    layer.claim(fy, fx, fang, hf)
                    placed.append((fy, fx, fang, hf))
                    break

    h, w = dist.shape
    out: List[Placed] = []
    for cy, cx, ang, hf in placed:
        fp = layer.footprint(cy, cx, ang, hf)
        if fp is None:
            continue
        (y0, y1, x0, x1), mask = fp
        under = labels[y0:y1, x0:x1][mask]
        colour = (int(np.bincount(under, minlength=n_colors).argmax())
                  if under.size else int(labels[cy, cx]))
        out.append((_corners(cx, cy, ang, hf, w, h),
                    colour, (cx + 0.5) / w, (cy + 0.5) / h))
    return out
