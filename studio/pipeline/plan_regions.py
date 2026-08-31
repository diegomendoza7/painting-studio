"""Plan stage A — value tiers and paintable passages.

A watercolor is planned as *value* first and *shape* second, so this stage
produces both:

* **Value tiers** — the photo's L* quantized into ``wash_tiers`` bands by a
  seeded 1-D k-means. Tier 0 is the lightest. The tier count is literally how
  many layers of wash the painting will be built from.
* **Passages** — the shapes a brush actually covers. Pixels are clustered in
  LAB (the same k-means the board's palette extraction uses), the clusters are
  cut into connected components, and anything too small to paint is merged
  into the neighbour it shares the most border with. What comes out is a
  handful of nameable shapes rather than a pixel map.

Everything runs on a downsampled copy (``PLAN_EDGE``) — a painting plan needs
shapes, not detail, and the whole stage stays comfortably interactive.
Deterministic throughout: seeded init, fixed iteration counts, explicit
tie-breaks, so the same photo and settings always give the same numbered plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..color import lab_to_rgb, rgb_to_hex, rgb_to_lab
from ..models import Passage
from .extract import kmeans

# Segmentation resolution (longest edge). Shapes, not detail.
PLAN_EDGE = 200
# A passage is left as bare paper when it is both the lightest tier in the
# picture *and* genuinely light in absolute terms. The absolute floor is what
# stops a dark photo from having its least-dark shape declared "white"; the
# tier test is what stops a bright photo from reserving half the paper.
RESERVE_L = 82.0
# Never print more passages than a person will actually number and paint.
MAX_PASSAGES = 22
# Touching regions closer than this in LAB are one wash, not two. Set high on
# purpose: for a beginner, too few big shapes beats too many fiddly ones.
MERGE_DE = 7.0
_TIER_ITER = 30


@dataclass
class PassageMap:
    """Stage output: the shapes, plus the label image the plan map draws from."""

    passages: Tuple[Passage, ...]
    labels: np.ndarray            # (h,w) int, values index into ``passages``
    tier_centers: Tuple[float, ...]      # L* per tier, lightest first
    tier_fracs: Tuple[float, ...]        # share of the image in each tier
    adjacency: Dict[int, Dict[int, int]]  # passage id -> {neighbour id: shared border}

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.labels.shape[:2]
        return (w, h)


# --- helpers ------------------------------------------------------------------
def downsample(rgb: np.ndarray, max_edge: int = PLAN_EDGE) -> np.ndarray:
    """Area-average ``rgb`` down to ``max_edge`` on its longest side.

    Box-averaged rather than sampled: a sampled photo keeps its noise, and
    noise turns into spurious little passages further down the stage.
    """
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return rgb.astype(np.float64)
    scale = max_edge / float(longest)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    ys = np.unique(np.linspace(0, h, nh + 1).astype(int))
    xs = np.unique(np.linspace(0, w, nw + 1).astype(int))
    out = np.add.reduceat(rgb.astype(np.float64), ys[:-1], axis=0)
    out /= (np.diff(ys)).reshape(-1, 1, 1)
    out = np.add.reduceat(out, xs[:-1], axis=1)
    out /= (np.diff(xs)).reshape(1, -1, 1)
    return out


def value_tiers(lstars: np.ndarray, k: int) -> np.ndarray:
    """1-D k-means over L*, returning centers sorted lightest -> darkest.

    Quantile-seeded so the bands land where the picture's values actually sit
    (a high-key photo gets high-key tiers) instead of on an even ramp.
    """
    k = max(2, int(k))
    flat = lstars.reshape(-1)
    centers = np.quantile(flat, np.linspace(0.1, 0.9, k))
    centers = np.sort(np.unique(centers))
    if len(centers) < k:  # a flat image: spread duplicates apart
        centers = np.linspace(max(0.0, flat.min()), min(100.0, flat.max() + 1e-6), k)
    for _ in range(_TIER_ITER):
        idx = np.argmin(np.abs(flat[:, None] - centers[None, :]), axis=1)
        moved = False
        for i in range(len(centers)):
            pts = flat[idx == i]
            if len(pts):
                new = float(pts.mean())
                if abs(new - centers[i]) > 1e-6:
                    moved = True
                centers[i] = new
        if not moved:
            break
    return np.sort(centers)[::-1]  # tier 0 = lightest


def _tier_of(lstar: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - lstar)))


# --- connected components ------------------------------------------------------
def _components(labels: np.ndarray) -> np.ndarray:
    """4-connected components of an integer label image, via union-find.

    Only same-label neighbours are unioned, so each colour cluster is cut into
    its separate physical shapes.
    """
    h, w = labels.shape
    parent = list(range(h * w))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]   # path halving
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Always hang the larger index off the smaller: deterministic roots.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    flat_idx = np.arange(h * w).reshape(h, w)
    same_v = labels[:-1, :] == labels[1:, :]
    for a, b in zip(flat_idx[:-1, :][same_v], flat_idx[1:, :][same_v]):
        union(int(a), int(b))
    same_h = labels[:, :-1] == labels[:, 1:]
    for a, b in zip(flat_idx[:, :-1][same_h], flat_idx[:, 1:][same_h]):
        union(int(a), int(b))

    roots = np.array([find(i) for i in range(h * w)], dtype=np.int64)
    _uniq, comp = np.unique(roots, return_inverse=True)
    return comp.reshape(h, w).astype(np.int32)


def _adjacency(comp: np.ndarray) -> Dict[int, Dict[int, int]]:
    """Shared 4-connected border length between every pair of components."""
    adj: Dict[int, Dict[int, int]] = {}

    def add(a: np.ndarray, b: np.ndarray) -> None:
        diff = a != b
        if not diff.any():
            return
        pairs = np.stack([np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        for (u, v), c in zip(uniq, counts):
            adj.setdefault(int(u), {})[int(v)] = adj.setdefault(int(u), {}).get(int(v), 0) + int(c)
            adj.setdefault(int(v), {})[int(u)] = adj.setdefault(int(v), {}).get(int(u), 0) + int(c)

    add(comp[:-1, :], comp[1:, :])
    add(comp[:, :-1], comp[:, 1:])
    return adj


# --- merging -------------------------------------------------------------------
def _merge_regions(
    comp: np.ndarray, lab_img: np.ndarray, min_px: int, max_regions: int
) -> np.ndarray:
    """Reduce raw components to shapes a person would actually paint.

    Three rules, applied in order of urgency until none of them fires:

    1. **Too small to paint** — fold it into the neighbour it shares the most
       border with (ties on colour distance, then id).
    2. **Too similar to tell apart** — two touching regions closer than
       ``MERGE_DE`` are the same wash, whatever the clusterer thought. Photo
       noise splits a flat area into several clusters, and without this rule a
       plain wall arrives as nine numbered passages.
    3. **Too many to number** — while over ``max_regions``, keep merging the
       closest touching pair.

    Ties break deterministically throughout, so the same photo always yields
    the same numbered plan.
    """
    n = int(comp.max()) + 1
    flat = comp.reshape(-1)
    lab_flat = lab_img.reshape(-1, 3)
    area = np.bincount(flat, minlength=n).astype(np.int64)
    sums = np.zeros((n, 3), dtype=np.float64)
    for ch in range(3):
        sums[:, ch] = np.bincount(flat, weights=lab_flat[:, ch], minlength=n)
    adj = _adjacency(comp)
    alive = {i for i in range(n) if area[i] > 0}
    merged_into: Dict[int, int] = {}

    def mean_lab(i: int) -> np.ndarray:
        return sums[i] / max(1, area[i])

    def best_neighbour(i: int) -> int:
        cands = [(j, b) for j, b in adj.get(i, {}).items() if j in alive and j != i]
        if not cands:
            return -1
        mi = mean_lab(i)
        return min(
            cands,
            key=lambda jb: (-jb[1], float(np.linalg.norm(mean_lab(jb[0]) - mi)), jb[0]),
        )[0]

    def closest_pair() -> Tuple[float, int, int]:
        """The touching pair with the smallest colour distance."""
        best = (float("inf"), -1, -1)
        for i in sorted(alive):
            mi = mean_lab(i)
            for j in adj.get(i, {}):
                if j <= i or j not in alive:
                    continue
                d = float(np.linalg.norm(mean_lab(j) - mi))
                if (d, i, j) < best:
                    best = (d, i, j)
        return best

    stuck: set = set()   # regions with no neighbour left to merge into
    while True:
        pool = [i for i in alive if i not in stuck]
        small = [i for i in pool if area[i] < min_px]
        if small:
            src = min(small, key=lambda i: (area[i], i))
            dst = best_neighbour(src)
        else:
            d, a, b = closest_pair()
            if a < 0:
                break
            if d >= MERGE_DE and len(alive) <= max_regions:
                break
            # Fold the smaller of the pair into the larger.
            src, dst = (a, b) if (area[a], -a) < (area[b], -b) else (b, a)
        if dst < 0:
            # Isolated shape (or the whole image is one region): leave it be.
            stuck.add(src)
            continue
        area[dst] += area[src]
        sums[dst] += sums[src]
        for j, b in adj.get(src, {}).items():
            if j == dst or j not in alive:
                continue
            adj.setdefault(dst, {})[j] = adj.setdefault(dst, {}).get(j, 0) + b
            adj.setdefault(j, {})[dst] = adj.setdefault(j, {}).get(dst, 0) + b
            adj[j].pop(src, None)
        adj.get(dst, {}).pop(src, None)
        adj.pop(src, None)
        alive.discard(src)
        merged_into[src] = dst
        area[src] = 0

    # Resolve merge chains, then relabel to a dense 0..n-1 range.
    lut = np.arange(n, dtype=np.int32)
    for i in range(n):
        root = i
        seen = 0
        while root in merged_into and seen < n:
            root = merged_into[root]
            seen += 1
        lut[i] = root
    out = lut[comp]
    _uniq, dense = np.unique(out, return_inverse=True)
    return dense.reshape(comp.shape).astype(np.int32)


# --- public entry point ----------------------------------------------------------
def build_passages(
    rgb: np.ndarray, wash_tiers: int, plan_colors: int, min_passage_pct: float
) -> PassageMap:
    """Segment ``rgb`` into value tiers and numbered paintable passages."""
    small = downsample(rgb)
    lab_img = rgb_to_lab(small)
    h, w = lab_img.shape[:2]
    total = float(h * w)

    centers = value_tiers(lab_img[..., 0], wash_tiers)
    tier_px = np.argmin(np.abs(lab_img[..., 0][..., None] - centers[None, None, :]), axis=2)
    tier_fracs = tuple(
        float((tier_px == t).sum() / total) for t in range(len(centers))
    )

    k = max(2, min(int(plan_colors), int(total)))
    _cc, clus = kmeans(lab_img.reshape(-1, 3), k)
    comp = _components(clus.reshape(h, w))
    min_px = max(4, int(round(total * max(0.05, min_passage_pct) / 100.0)))
    comp = _merge_regions(comp, lab_img, min_px, MAX_PASSAGES)

    # Per-region statistics.
    n = int(comp.max()) + 1
    flat = comp.reshape(-1)
    area = np.bincount(flat, minlength=n).astype(np.int64)
    lab_flat = lab_img.reshape(-1, 3)
    means = np.zeros((n, 3), dtype=np.float64)
    for ch in range(3):
        means[:, ch] = np.bincount(flat, weights=lab_flat[:, ch], minlength=n) / np.maximum(area, 1)
    ys, xs = np.mgrid[0:h, 0:w]
    cy = np.bincount(flat, weights=ys.reshape(-1), minlength=n) / np.maximum(area, 1)
    cx = np.bincount(flat, weights=xs.reshape(-1), minlength=n) / np.maximum(area, 1)

    order = sorted(
        range(n),
        key=lambda i: (_tier_of(means[i][0], centers), -area[i], i),
    )
    passages: List[Passage] = []
    remap = np.zeros(n, dtype=np.int32)
    for rank, i in enumerate(order):
        remap[i] = rank
        mask = comp == i
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y0, y1 = np.argmax(rows), h - np.argmax(rows[::-1])
        x0, x1 = np.argmax(cols), w - np.argmax(cols[::-1])
        # The centroid of a concave shape (a C, a ring) falls outside it, which
        # would print the passage's number on top of its neighbour. Snap to the
        # nearest pixel that really belongs to this passage.
        py, px = int(round(cy[i])), int(round(cx[i]))
        if not (0 <= py < h and 0 <= px < w and mask[py, px]):
            pts = np.argwhere(mask)
            d2 = (pts[:, 0] - cy[i]) ** 2 + (pts[:, 1] - cx[i]) ** 2
            py, px = (int(v) for v in pts[int(np.argmin(d2))])
        srgb = tuple(int(c) for c in lab_to_rgb(means[i]))
        lstar = float(means[i][0])
        passages.append(
            Passage(
                id=rank + 1,
                target_hex=rgb_to_hex(srgb),
                area_frac=float(area[i] / total),
                tier=_tier_of(lstar, centers),
                lstar=round(lstar, 2),
                cx=float((px + 0.5) / w),
                cy=float((py + 0.5) / h),
                bbox=(float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)),
                reserved=lstar >= RESERVE_L and _tier_of(lstar, centers) == 0,
            )
        )

    labels = remap[comp]
    raw_adj = _adjacency(labels)
    adjacency = {
        int(a) + 1: {int(b) + 1: int(c) for b, c in nb.items()}
        for a, nb in raw_adj.items()
    }
    return PassageMap(
        passages=tuple(passages),
        labels=labels,
        tier_centers=tuple(float(c) for c in centers),
        tier_fracs=tier_fracs,
        adjacency=adjacency,
    )
