"""Easel mode, the Qt-free half: pick a colour off the reference, and say how to mix it.

This is the tool you use *while painting*, not a document you print. You click
a region of the reference photo; the app finds the patch of similar colour you
meant, averages it, shows it big, and tells you — from the paints you actually
own — what to mix, in what order, and roughly how much.

Three decisions shape it:

* **A region, not a pixel.** A photo is noisy at the pixel level, so a single
  pixel is almost never the colour the painter is looking at. Clicking grows a
  region of similar colour out from the click (compared against the clicked
  colour, on a lightly blurred copy), and the swatch is that region's average.
  Knowing the region's size is also what makes an amount possible: a region's
  share of the picture is its share of the canvas.
* **An order, not just a ratio.** Oil mixing goes wrong in the order more than
  in the proportions: strong pigments (Phthalo, Alizarin) swamp a mix if they
  go in early. The steps start from the base — white first when white
  dominates, since you add colour to white, never white to colour — and put
  strong paints last, "a little at a time".
* **Amounts in things you can see.** Millilitres mean nothing at the easel, so
  an amount is also given as a pile you recognise and as centimetres squeezed
  from a standard tube.

Everything here is pure functions over numpy arrays and the existing models, so
the smoke suite can drive it without a display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .color import hex_to_rgb, lab, lab_to_rgb, lch, rgb_to_hex, rgb_to_lab, temperature
from .media import dilution_phrase
from .models import MixRecipe, Paint
from .params import CM_PER_INCH, UNIT_CM

# Region picking runs on a copy this size — plenty to find a patch of colour,
# and small enough that growing one is instant.
ANALYSIS_EDGE = 360
# Membership is judged on a blurred copy. At one pixel of blur a textured area
# (marble, skin, a painted wall) stays speckled, so the region is a single dot
# at low tolerance and floods half the picture one notch higher; at three it
# grows smoothly as tolerance rises.
BLUR_PX = 3
DEFAULT_TOLERANCE = 10.0            # CIE76 ΔE from the clicked colour
MIN_TOLERANCE, MAX_TOLERANCE = 3.0, 30.0
# The tolerances a click sweeps through to find its natural region.
TOLERANCE_STEPS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 21.0, 24.0)
MIN_REGION = 0.004                  # below this a "region" is really a speck
AUTO_TOLERANCE = 8.0                # where "auto" starts: plainly the same colour family
LEAK_RATIO = 2.0                    # one step more than doubling the region = it leaked
MAX_BIAS = 3                        # how far "smaller" / "larger" can move from auto
SPOT_FRAC = 0.012                   # spot radius, as a share of the image diagonal

# Oil/acrylic coverage. A normal painting layer is ~0.15 mm, so one millilitre
# covers ~60 cm²; the safety factor is there because running out of a mixed
# colour and remixing it never quite matches.
CM2_PER_ML = 60.0
SAFETY = 1.4
MIN_ML = 0.3                        # you cannot usefully mix less than a pea
# A standard 37–40 ml tube squeezes a ~5.6 mm worm: about 0.25 ml per cm.
TUBE_ML_PER_CM = 0.25
# Below this share of a mix, a paint is added "a touch at a time".
TOUCH_SHARE = 0.12
# The neutral surround the big swatch sits on: N5, value 5 of 10. Surrounding
# colour shifts how a colour is seen; a mid grey is the one surround that
# doesn't push it warmer, cooler, lighter or darker.
NEUTRAL_GREY = "#767676"


# --- analysis image -------------------------------------------------------------
@dataclass
class Analysis:
    """The reference, prepared once for picking."""

    rgb: np.ndarray        # (h, w, 3) uint8, analysis resolution
    lab: np.ndarray        # unblurred LAB — what region colours are averaged from
    lab_soft: np.ndarray   # lightly blurred LAB — what region membership is judged on

    @property
    def shape(self) -> Tuple[int, int]:
        return self.lab.shape[:2]


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a.astype(np.float64)
    out = a.astype(np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
        out = (hi - lo) / float(2 * r + 1)
    return out


def prepare(rgb: np.ndarray, edge: int = ANALYSIS_EDGE) -> Analysis:
    """Downsample (area-average) and convert once, so every click is cheap."""
    h, w = rgb.shape[:2]
    scale = min(1.0, edge / float(max(h, w)))
    if scale < 1.0:
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        ys = np.unique(np.linspace(0, h, nh + 1).astype(int))
        xs = np.unique(np.linspace(0, w, nw + 1).astype(int))
        small = np.add.reduceat(rgb.astype(np.float64), ys[:-1], axis=0)
        small /= np.diff(ys).reshape(-1, 1, 1)
        small = np.add.reduceat(small, xs[:-1], axis=1)
        small /= np.diff(xs).reshape(1, -1, 1)
        small = np.clip(np.round(small), 0, 255).astype(np.uint8)
    else:
        small = rgb.astype(np.uint8)
    lab_img = rgb_to_lab(small)
    return Analysis(rgb=small, lab=lab_img, lab_soft=_box_blur(lab_img, BLUR_PX))


# --- picking --------------------------------------------------------------------
@dataclass
class Pick:
    """What a click selected."""

    hex: str
    mask: np.ndarray                      # (h, w) bool at analysis resolution
    area_frac: float                      # share of the picture
    x: float                              # the click, normalized 0..1
    y: float
    bbox: Tuple[float, float, float, float]   # normalized x0, y0, x1, y1
    tolerance: float = 0.0                # what the region was grown at (0 for a spot)


def _to_px(an: Analysis, x: float, y: float) -> Tuple[int, int]:
    h, w = an.shape
    return (min(h - 1, max(0, int(y * h))), min(w - 1, max(0, int(x * w))))


def _grow(similar: np.ndarray, sy: int, sx: int, start: Optional[np.ndarray] = None) -> np.ndarray:
    """The 4-connected part of ``similar`` that contains the seed.

    Vectorized dilation constrained to ``similar``: each pass grows the region by
    a pixel, so there is no Python-level flood fill over thousands of pixels.
    ``start`` resumes from a smaller region already grown at a lower tolerance —
    regions only ever grow as tolerance rises, so a sweep need not start over.
    """
    mask = np.zeros(similar.shape, dtype=bool) if start is None else start.copy()
    mask[sy, sx] = True
    if not similar[sy, sx]:
        return mask
    while True:
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= similar
        if np.array_equal(grown, mask):
            return mask
        mask = grown


def _finish(an: Analysis, mask: np.ndarray, x: float, y: float, tolerance: float = 0.0) -> Pick:
    h, w = an.shape
    mean = an.lab[mask].mean(axis=0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = int(np.argmax(rows)), int(h - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(w - np.argmax(cols[::-1]))
    return Pick(
        hex=rgb_to_hex(tuple(int(c) for c in lab_to_rgb(mean))),
        mask=mask,
        area_frac=float(mask.mean()),
        x=float(x),
        y=float(y),
        bbox=(x0 / w, y0 / h, x1 / w, y1 / h),
        tolerance=float(tolerance),
    )


def _similar(an: Analysis, sy: int, sx: int, tol: float) -> np.ndarray:
    seed = an.lab_soft[sy, sx]
    return np.sum((an.lab_soft - seed) ** 2, axis=2) <= tol * tol


def region_sweep(an: Analysis, x: float, y: float) -> List[Tuple[float, np.ndarray]]:
    """The region around a click at every step of :data:`TOLERANCE_STEPS`."""
    sy, sx = _to_px(an, x, y)
    out: List[Tuple[float, np.ndarray]] = []
    mask: Optional[np.ndarray] = None
    for tol in TOLERANCE_STEPS:
        mask = _grow(_similar(an, sy, sx, tol), sy, sx, mask)
        out.append((tol, mask))
    return out


def natural_index(areas: Sequence[float]) -> int:
    """Which step of a sweep is "the shape you clicked".

    Anchored on :data:`AUTO_TOLERANCE` — colour a painter would call the same
    mix — then corrected in two directions:

    * **Up, out of a speck.** On a textured surface the anchor can be a handful
      of pixels; keep going until the region is at least :data:`MIN_REGION`.
    * **Down, out of a leak.** If the anchor region is more than
      :data:`LEAK_RATIO` times the step below it, the tolerance has just passed
      the contrast at the shape's edge and the region has spilled into its
      neighbour; step back until it hasn't.

    The earlier version chose the step whose area changed least relative to its
    size — which reliably picks a flooded half of the picture, because a huge
    region grows slowly in relative terms. A painter's "this shape" is local.
    """
    n = len(areas)
    i = min(range(n), key=lambda k: abs(TOLERANCE_STEPS[k] - AUTO_TOLERANCE)) if n == len(TOLERANCE_STEPS) else n // 2
    while i < n - 1 and areas[i] < MIN_REGION:
        i += 1
    while i > 0 and areas[i - 1] >= MIN_REGION and areas[i] > LEAK_RATIO * areas[i - 1]:
        i -= 1
    return i


def pick_region(
    an: Analysis, x: float, y: float, bias: int = 0, tolerance: Optional[float] = None
) -> Pick:
    """The patch of similar colour around a click, and its average colour.

    With no ``tolerance`` the region is found automatically (see
    :func:`natural_index`), and ``bias`` moves it that many steps smaller or
    larger — the "region size" control. Passing a ``tolerance`` reproduces an
    earlier pick exactly, which is how a saved swatch redraws its outline.

    Membership is judged against the *clicked* colour rather than grown
    neighbour to neighbour, so a slow gradient can't drag the region across the
    whole picture.
    """
    if tolerance is not None:
        tol = max(MIN_TOLERANCE, min(MAX_TOLERANCE, float(tolerance)))
        sy, sx = _to_px(an, x, y)
        return _finish(an, _grow(_similar(an, sy, sx, tol), sy, sx), x, y, tol)
    sweep = region_sweep(an, x, y)
    i = natural_index([float(m.mean()) for _t, m in sweep])
    i = max(0, min(len(sweep) - 1, i + max(-MAX_BIAS, min(MAX_BIAS, int(bias)))))
    tol, mask = sweep[i]
    return _finish(an, mask, x, y, tol)


def pick_spot(an: Analysis, x: float, y: float, radius_frac: float = SPOT_FRAC) -> Pick:
    """A small brush-sized circle around the click, for details a region would swallow."""
    h, w = an.shape
    sy, sx = _to_px(an, x, y)
    r = max(1.0, radius_frac * math.hypot(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (yy - sy) ** 2 + (xx - sx) ** 2 <= r * r
    return _finish(an, mask, x, y)


def region_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a region — what gets drawn around it on the photo."""
    edge = np.zeros(mask.shape, dtype=bool)
    edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    edge[0, :] |= mask[0, :]
    edge[-1, :] |= mask[-1, :]
    edge[:, 0] |= mask[:, 0]
    edge[:, -1] |= mask[:, -1]
    return edge


def crop_around(rgb: np.ndarray, x: float, y: float, bbox, out_px: int = 128) -> np.ndarray:
    """A square crop centred on the click — the history thumbnail for "where was this?"."""
    from PIL import Image

    h, w = rgb.shape[:2]
    bw = (bbox[2] - bbox[0]) * w
    bh = (bbox[3] - bbox[1]) * h
    side = int(max(0.12 * min(h, w), min(0.5 * min(h, w), 1.3 * max(bw, bh))))
    side = max(4, side)
    cx, cy = int(x * w), int(y * h)
    x0 = min(max(0, cx - side // 2), max(0, w - side))
    y0 = min(max(0, cy - side // 2), max(0, h - side))
    crop = rgb[y0:y0 + side, x0:x0 + side]
    img = Image.fromarray(np.ascontiguousarray(crop)).resize((out_px, out_px), Image.LANCZOS)
    return np.asarray(img)


# --- describing a colour -----------------------------------------------------------
def value_of(hex_: str) -> Tuple[float, int]:
    """(L*, value on a 0–10 painter's scale)."""
    L = lab(hex_to_rgb(hex_))[0]
    return L, int(round(L / 10.0))


def describe(hex_: str) -> str:
    """The colour in words a painter uses: value, chroma, temperature.

    Chroma is worth saying out loud: the commonest mixing mistake is going too
    saturated, and most colours in a photo are far greyer than they look.
    """
    L, C, _h = lch(hex_to_rgb(hex_))
    value = (
        "very dark" if L < 20 else "dark" if L < 35 else "mid-dark" if L < 48
        else "mid" if L < 60 else "light" if L < 78 else "very light"
    )
    chroma = (
        "near-grey" if C < 8 else "muted" if C < 22 else "moderately saturated" if C < 42
        else "intense"
    )
    temp = {"W": "warm", "C": "cool"}.get(temperature(hex_to_rgb(hex_)), "neutral")
    return f"{value}, {chroma}, {temp}"


# --- how much -------------------------------------------------------------------------
def canvas_cm2(width: float, height: float, units: str) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    k = 1.0 if units == UNIT_CM else CM_PER_INCH
    return (width * k) * (height * k)


def amount_ml(area_frac: float, canvas_area_cm2: float) -> float:
    """Paint to mix for a region covering ``area_frac`` of the canvas (oil/acrylic)."""
    if canvas_area_cm2 <= 0:
        return 0.0
    ml = area_frac * canvas_area_cm2 / CM2_PER_ML * SAFETY
    ml = max(MIN_ML, ml)
    return round(ml * 2) / 2 if ml >= 2 else round(ml, 1)


def pile_phrase(ml: float) -> str:
    """An amount as a pile you can picture on the palette."""
    if ml <= 0:
        return ""
    if ml < 0.5:
        return "about a pea"
    if ml < 2.5:
        return "about a hazelnut"
    if ml < 8:
        return "about a walnut"
    if ml < 25:
        return "about a golf ball"
    return "more than a golf ball — mix it in two batches"


def tube_text(ml: float) -> str:
    """Centimetres squeezed from a standard tube."""
    cm = ml / TUBE_ML_PER_CM
    if cm < 0.4:
        return "a tiny touch"
    return f"{round(cm * 2) / 2:g} cm"


# --- mixing order ----------------------------------------------------------------------
def _is_strong(paint: Optional[Paint]) -> bool:
    """A pigment that swamps a mix — the staining ones (Phthalos, Alizarin…)."""
    return bool(paint is not None and paint.staining)


def _is_touch(paint: Optional[Paint], share: float) -> bool:
    """Added gradually: strong pigments, and anything used in only a small share."""
    return _is_strong(paint) or share < TOUCH_SHARE


def ordered_steps(recipe: MixRecipe, palette: Sequence[Paint]):
    """The recipe's paints in the order to add them.

    White first when it dominates — colour goes into white, never the reverse.
    Otherwise the base is the biggest share *that isn't a strong pigment*: you
    never start a mix with Phthalo, however much of it the recipe calls for,
    because you can't take it back out. Strong and small-share paints come last.
    """
    by_hex = {p.hex: p for p in palette}
    steps = list(recipe.steps)
    if not steps:
        return []
    whites = [s for s in steps if getattr(by_hex.get(s.paint_hex), "is_white", False)]
    gentle = [s for s in steps if not _is_strong(by_hex.get(s.paint_hex))]
    if whites and whites[0].weight >= 0.3:
        base = whites[0]
    else:
        base = max(gentle or steps, key=lambda s: (s.weight, s.paint_name))
    rest = [s for s in steps if s is not base]
    rest.sort(key=lambda s: (_is_touch(by_hex.get(s.paint_hex), s.weight),
                             _is_strong(by_hex.get(s.paint_hex)), -s.weight, s.paint_name))
    return [base] + rest


def mixing_steps(
    recipe: MixRecipe, palette: Sequence[Paint], medium: str, total_ml: float
) -> List[str]:
    """Numbered instructions for making the mix, in the order that works."""
    from .pipeline.plan_schedule import mix_parts

    by_hex = {p.hex: p for p in palette}

    def name(step) -> str:
        return by_hex[step.paint_hex].name if step.paint_hex in by_hex else step.paint_name

    order = ordered_steps(recipe, palette)
    if not order:
        return []
    if recipe.layering:        # oil pastel: a surface layering, not a palette mix
        if len(order) == 1:
            return [f"Use {name(order[0])} straight from the stick."]
        base, top = recipe.steps[0], recipe.steps[-1]
        return [
            f"Lay down {name(base)} first.",
            f"Work {name(top)} over it — about {int(round(top.weight * 100))}% coverage.",
            "Blend with a finger or torchon.",
        ]
    if medium == "watercolor":
        parts = mix_parts([s.weight for s in order])
        ratio = " : ".join(f"{p} {name(s)}" for p, s in zip(parts, order))
        return [f"In a well, mix {ratio}.", f"Thin it to a {dilution_phrase(recipe.dilution)}."]

    parts = mix_parts([s.weight for s in order])
    out: List[str] = []
    if len(order) == 1:
        amt = f" ({tube_text(total_ml)})" if total_ml > 0 else ""
        return [f"Use {name(order[0])} straight from the tube{amt}."]
    for i, (s, p) in enumerate(zip(order, parts)):
        paint = by_hex.get(s.paint_hex)
        amt = tube_text(total_ml * s.weight) if total_ml > 0 else f"{p} part{'s' if p != 1 else ''}"
        if i == 0:
            role = "the base" if not getattr(paint, "is_white", False) else "add colour to it, not the other way round"
            out.append(f"Start with {amt} of {name(s)} — {role}.")
        elif _is_strong(paint):
            out.append(
                f"Add {name(s)} a little at a time ({amt} in all) — it's a strong pigment, "
                "so check after each addition; you can't take it back out."
            )
        elif _is_touch(paint, s.weight):
            out.append(f"Add a touch of {name(s)} ({amt}) — only a small amount is needed.")
        else:
            out.append(f"Mix in {amt} of {name(s)}.")
    out.append("Check it against the swatch: squint, match the value first, then the colour.")
    return out


# --- what the swatch view needs ------------------------------------------------------
def paints_per_mix(medium: str) -> int:
    """How many paints a recipe may use.

    Four for oil and acrylic: a muted neutral wants white plus a warm, a cool
    and a modifier, and three paints leave ordinary greys ~ΔE 8 out. Watercolor
    stays at three — lightness comes from water there, and a fourth pigment in a
    wash is mud, not accuracy.
    """
    return 4 if medium in ("oil", "acrylic") else 3


def amount_summary(medium: str, area_frac: float, canvas_area_cm2: float) -> Tuple[float, str]:
    """(millilitres to mix, the sentence that says so) for a region of the canvas."""
    if medium == "oilpastel":
        return 0.0, ""
    if canvas_area_cm2 <= 0:
        return 0.0, "Set the canvas size in Painting Settings to get amounts."
    if medium == "watercolor":
        from .pipeline.plan_schedule import volume_text, wash_volume_ml

        ml = wash_volume_ml(area_frac, canvas_area_cm2, False)
        return ml, f"{volume_text(ml)[0].upper()}{volume_text(ml)[1:]}."
    ml = amount_ml(area_frac, canvas_area_cm2)
    return ml, (f"{ml:g} ml of mixed paint — {pile_phrase(ml)} ({tube_text(ml)} squeezed "
                f"from a standard tube), for {area_frac * 100:.0f}% of the canvas.")


def match_words(delta_e: float) -> str:
    """ΔE in words: how close the mix your palette can make is to the target."""
    if delta_e < 2.0:
        return "indistinguishable"
    if delta_e < 4.0:
        return "very close"
    if delta_e < 6.0:
        return "close"
    if delta_e < 10.0:
        return "noticeably off"
    return "out of reach"


def grey_of(hex_: str) -> str:
    """The neutral grey with the same value — what the swatch looks like with the colour taken away."""
    L = lab(hex_to_rgb(hex_))[0]
    return rgb_to_hex(tuple(int(c) for c in lab_to_rgb(np.array([L, 0.0, 0.0]))))


def greyscale(rgb: np.ndarray) -> np.ndarray:
    """A value-only copy of an image (L* kept, colour removed)."""
    L = rgb_to_lab(rgb)[..., 0]
    zeros = np.zeros_like(L)
    return lab_to_rgb(np.stack([L, zeros, zeros], axis=-1))


# --- one swatch per area ---------------------------------------------------------------
# A re-click is "the same area" when it is the same kind of pick, made at the
# same Size setting, and its region shares most of its pixels with the saved
# one. Overlap, not size, is the test: a textured patch doesn't always regrow to
# the same size from a seed a pixel away (on a real reference a re-click gave
# the same colour over 0.29× the area), but it grows over the same pixels. The
# colour limit is only a cheap pre-filter, loose enough that a small, noisy dark
# area re-clicked a pixel off still counts as itself.
SAME_DE = 10.0
SAME_OVERLAP = 0.5


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Shared pixels as a share of the smaller region."""
    smaller = min(int(a.sum()), int(b.sum()))
    return float((a & b).sum()) / smaller if smaller else 0.0


def same_area(pick: Pick, spot: bool, swatch, saved_mask: np.ndarray, bias: int = 0) -> bool:
    """Whether ``pick`` re-selects the area an existing ``swatch`` already holds.

    ``saved_mask`` is the swatch's region redrawn on the current reference.
    ``swatch`` only needs ``pick``, ``hex`` and ``bias``.
    """
    from .color import delta_e_rgb

    if (swatch.pick == "spot") != spot:
        return False
    if not spot and int(getattr(swatch, "bias", 0)) != int(bias):
        return False        # a deliberate Size change is a different pick
    if delta_e_rgb(hex_to_rgb(pick.hex), hex_to_rgb(swatch.hex)) >= SAME_DE:
        return False
    return _overlap(pick.mask, saved_mask) >= SAME_OVERLAP


def find_same(an: Analysis, pick: Pick, spot: bool, bias: int, swatches):
    """The saved swatch this pick duplicates, if any.

    Filters on colour first, so only a handful of saved regions are ever
    redrawn — a painting with a hundred swatches still answers instantly.
    """
    from .color import delta_e_rgb

    for sw in swatches:
        if (sw.pick == "spot") != spot:
            continue
        if delta_e_rgb(hex_to_rgb(pick.hex), hex_to_rgb(sw.hex)) >= SAME_DE:
            continue
        saved = (pick_spot(an, sw.x, sw.y) if sw.pick == "spot"
                 else pick_region(an, sw.x, sw.y, tolerance=sw.tolerance))
        if same_area(pick, spot, sw, saved.mask, bias):
            return sw
    return None


def next_swatch_name(names) -> str:
    """"Swatch N" one past the highest number in use.

    Counting the swatches instead reuses a name as soon as one is deleted — seven
    swatches with one deleted made the next one a second "Swatch 6".
    """
    import re

    used = [int(m.group(1)) for n in names if (m := re.fullmatch(r"Swatch (\d+)", n or ""))]
    return f"Swatch {max(used, default=0) + 1}"
