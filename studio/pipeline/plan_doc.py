"""Plan stage C — compose the watercolor painting plan ``Scene``.

The plan is the app's second document. Where the board is a reference sheet of
colours, this is an ordered set of instructions for painting one specific
photo: the rules that keep a beginner out of trouble, a numbered map of the
shapes, the whites to leave bare, then pass after pass of "mix this much of
this, put it here, let it dry".

Like the board, compose is cheap and always re-runs, so everything cosmetic
lives here — section toggles, the selected-pass highlight, and paint *names*,
which are resolved from the current palette by hex at this stage so renaming a
paint never invalidates a cached stage.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..color import hex_to_rgb, rgb_to_hex
from ..models import MixRecipe, Passage, WashPass
from ..params import CM_PER_INCH, Params, UNIT_CM
from ..scene import (
    ImagePatch,
    LAYER_HEADER,
    LAYER_NOTES,
    LAYER_PLAN_MAP,
    LAYER_PLAN_PASSES,
    LAYER_PLAN_RULES,
    LAYER_PLAN_TIPS,
    Layer,
    Scene,
    Stroke,
    Text,
)
from .. import tips as tips_mod
from .board import ATTRIBUTION, BOARD_W, GAP, PAD, BoardLayout, _rect, _rect_pts
from .plan_regions import PassageMap
from .plan_schedule import dilution_label, parts_text, reserved_passages

MAP_W = 560                 # the big plan map, in scene units
THUMB_W = 150               # per-pass mini-map column width
PASS_H = 170                # one wash's block; the thumbnail is fitted to it
UPSCALE = 3                 # nearest-neighbour, so passage edges stay crisp
LINE = 16.0
# Rough advance width per character at a given font size, used only to keep a
# long generated sentence from running off the page. Sans-serif averages a bit
# over half the point size; erring wide truncates a little early, never late.
_CHAR_W = 0.52


def _clip(s: str, width: float, size: float) -> str:
    """Truncate ``s`` so it fits ``width`` scene units at ``size``."""
    n = max(8, int(width / (size * _CHAR_W)))
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;·") + "…"


# --- raster helpers -------------------------------------------------------------
def _fill_table(
    pmap: PassageMap, recipes: Sequence[MixRecipe], paper_hex: str
) -> np.ndarray:
    """One RGB row per passage: the colour that passage will actually be.

    Reserved passages take the paper colour — the map should show the painting
    as planned, and a reserved shape is planned to stay bare.
    """
    paper = np.array(hex_to_rgb(paper_hex), dtype=np.uint8)
    rows = []
    for i, p in enumerate(pmap.passages):
        if p.reserved:
            rows.append(paper)
        else:
            hex_ = recipes[i].mixed_hex if i < len(recipes) else p.target_hex
            rows.append(np.array(hex_to_rgb(hex_), dtype=np.uint8))
    return np.array(rows or [paper], dtype=np.uint8)


def _map_raster(
    pmap: PassageMap,
    fills: np.ndarray,
    ink_hex: str,
    paper_hex: str,
    highlight: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Flat-filled passage map with drawn borders, optionally dimmed outside ``highlight``."""
    labels = pmap.labels
    out = fills[np.clip(labels, 0, len(fills) - 1)].astype(np.float64)
    if highlight is not None:
        keep = np.isin(labels + 1, np.asarray(list(highlight), dtype=np.int64))
        paper = np.array(hex_to_rgb(paper_hex), dtype=np.float64)
        out[~keep] = out[~keep] * 0.22 + paper * 0.78
    border = np.zeros(labels.shape, dtype=bool)
    border[:-1, :] |= labels[:-1, :] != labels[1:, :]
    border[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    border[1:, :] |= labels[1:, :] != labels[:-1, :]
    border[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    out[border] = np.array(hex_to_rgb(ink_hex), dtype=np.float64)
    img = np.clip(out, 0, 255).astype(np.uint8)
    return np.repeat(np.repeat(img, UPSCALE, axis=0), UPSCALE, axis=1)


# --- the composer -----------------------------------------------------------------
def build_plan(
    params: Params,
    pmap: Optional[PassageMap],
    recipes: Sequence[MixRecipe],
    passes: Sequence[WashPass],
    working_rgb: Optional[np.ndarray],
) -> Tuple[Scene, BoardLayout]:
    ink = params.ink
    names_by_hex = {p.hex: p.name for p in params.palette}
    layers = {
        name: Layer(name=name)
        for name in (LAYER_HEADER, LAYER_PLAN_RULES, LAYER_PLAN_MAP,
                     LAYER_PLAN_PASSES, LAYER_PLAN_TIPS, LAYER_NOTES)
    }
    layers[LAYER_PLAN_MAP].visible = params.show_plan_map
    layers[LAYER_PLAN_PASSES].visible = params.show_plan_passes
    layers[LAYER_PLAN_TIPS].visible = params.show_plan_tips
    layout = BoardLayout()
    patches: List[ImagePatch] = []
    y = float(PAD)

    def text(layer, x, ty, s, size=12.0, anchor="start", opacity=1.0, color=None):
        layers[layer].texts.append(
            Text(x=x, y=ty, text=s, size=size, color=color or ink, anchor=anchor, opacity=opacity)
        )

    def title(layer, s, ty) -> float:
        text(layer, PAD, ty + 14, s.upper(), size=13.0, opacity=0.65)
        return ty + 24

    # ----- header ---------------------------------------------------------------
    head = params.board_title or "Watercolor Painting Plan"
    text(LAYER_HEADER, PAD, y + 26, head, size=26.0)
    unit = "cm" if params.canvas_units == UNIT_CM else "in"
    n_paint = len([p for p in (pmap.passages if pmap else ()) if not p.reserved])
    sub = (
        f"{len(passes)} wash{'es' if len(passes) != 1 else ''} · {n_paint} passages · "
        f"{len(params.palette)} paints · paper {params.paper_w:g} × {params.paper_h:g} {unit}"
    )
    text(LAYER_HEADER, PAD, y + 48, sub, size=12.5, opacity=0.7)
    y += 64

    if pmap is None or working_rgb is None:
        text(LAYER_HEADER, PAD, y + 20,
             "Open a reference photo to build a painting plan.", size=14.0, opacity=0.8)
        text(LAYER_HEADER, PAD, y + 44,
             "The plan reads the photo's shapes and values, then tells you what to mix "
             "and in what order.", size=11.5, opacity=0.6)
        y += 70
        return _finish(params, layers, patches, layout, y)

    # ----- the rules that keep you out of trouble --------------------------------
    y = title(LAYER_PLAN_RULES, "Before you start", y)
    for line in tips_mod.GOLDEN_RULES:
        text(LAYER_PLAN_RULES, PAD, y + 13, "·  " + line, size=11.5, opacity=0.85)
        y += LINE + 1
    y += GAP - 8

    # ----- reference + numbered map ------------------------------------------------
    if params.show_plan_map:
        y = title(LAYER_PLAN_MAP, "The plan — reference and passages", y)
        ih, iw = working_rgb.shape[:2]
        ph = MAP_W * ih / float(iw)
        # Left: the photo. Right: the same shapes as they are planned to be painted.
        patches.append(ImagePatch(rgb=working_rgb, x=PAD, y=y, w=MAP_W, h=ph))
        layers[LAYER_PLAN_MAP].strokes.append(
            Stroke(points=_rect_pts(PAD, y, MAP_W, ph), closed=True, width=1.0, color=ink)
        )
        layout.photo_rect = (float(PAD), y, float(MAP_W), ph)
        layout.photo_img_size = (iw, ih)

        fills = _fill_table(pmap, recipes, params.background)
        highlight = None
        if 0 <= params.selected_pass < len(passes):
            highlight = passes[params.selected_pass].passage_ids
        mx = PAD + MAP_W + 24
        patches.append(
            ImagePatch(
                rgb=_map_raster(pmap, fills, ink, params.background, highlight),
                x=mx, y=y, w=MAP_W, h=ph,
            )
        )
        layers[LAYER_PLAN_MAP].strokes.append(
            Stroke(points=_rect_pts(mx, y, MAP_W, ph), closed=True, width=1.0, color=ink)
        )
        # Numbers sit on the map, at each passage's centroid.
        for p in pmap.passages:
            lx = mx + p.cx * MAP_W
            ly = y + p.cy * ph
            layers[LAYER_PLAN_MAP].strokes.append(
                Stroke(points=_circle(lx, ly - 4, 9.5), closed=True, width=0.0,
                       color=ink, fill=params.background, fill_opacity=0.82)
            )
            text(LAYER_PLAN_MAP, lx, ly, str(p.id), size=11.0, anchor="middle")
        text(LAYER_PLAN_MAP, PAD, y + ph + 15,
             "Your reference.", size=10.5, opacity=0.6)
        text(LAYER_PLAN_MAP, mx, y + ph + 15,
             "Every numbered shape is one passage; the colour is the mix you will make.",
             size=10.5, opacity=0.6)
        y += ph + 30

        # Value plan: how many washes deep the painting is, and how it divides.
        text(LAYER_PLAN_MAP, PAD, y + 12, "Value plan", size=11.5, opacity=0.7)
        bar_y = y + 20
        bw = (BOARD_W - 2 * PAD)
        x = float(PAD)
        for t, frac in enumerate(pmap.tier_fracs):
            seg = bw * frac
            L = pmap.tier_centers[t]
            g = int(round(max(0.0, min(100.0, L)) / 100.0 * 255))
            layers[LAYER_PLAN_MAP].strokes.append(
                _rect(x, bar_y, max(1.0, seg), 26, rgb_to_hex((g, g, g)), ink, 0.8)
            )
            if seg > 46:
                text(LAYER_PLAN_MAP, x + seg / 2, bar_y + 17,
                     f"{t + 1}", size=11.0, anchor="middle",
                     color="#1b1b1b" if L > 55 else "#f4f2ec")
            x += seg
        text(LAYER_PLAN_MAP, PAD, bar_y + 42,
             f"{len(pmap.tier_fracs)} value tiers, lightest to darkest — "
             "roughly how many layers deep the painting goes.", size=10.5, opacity=0.6)
        y = bar_y + 58 + GAP

        # Whites to keep.
        reserved = reserved_passages(pmap.passages)
        text(LAYER_PLAN_MAP, PAD, y + 12, "Leave these bare", size=11.5, opacity=0.7)
        if reserved:
            ids = ", ".join(str(p.id) for p in reserved)
            pct = sum(p.area_frac for p in reserved) * 100
            text(LAYER_PLAN_MAP, PAD, y + 30,
                 f"Passage{'s' if len(reserved) != 1 else ''} {ids} — about {pct:.0f}% of the "
                 "paper. Mask them or paint around them before the first wash.", size=11.5)
        else:
            text(LAYER_PLAN_MAP, PAD, y + 30,
                 "Nothing in this photo is light enough to leave as bare paper — "
                 "keep the first wash pale and it will read as your white.",
                 size=11.5, opacity=0.85)
        y += 46 + GAP

    # ----- the passes -----------------------------------------------------------------
    if params.show_plan_passes:
        y = title(LAYER_PLAN_PASSES, "The washes, in order", y)
        pl = layers[LAYER_PLAN_PASSES]
        if not passes:
            text(LAYER_PLAN_PASSES, PAD, y + 18,
                 "No paintable passages — every shape came out light enough to reserve.",
                 size=12.0, opacity=0.7)
            y += 36
        fills = _fill_table(pmap, recipes, params.background)
        for wp in passes:
            by = y
            if wp.index - 1 == params.selected_pass:
                pl.strokes.append(
                    _rect(PAD - 8, by - 6, BOARD_W - 2 * PAD + 16, PASS_H - 4, None, ink, 1.8)
                )
            layout.pass_rects.append(
                ((float(PAD), by, BOARD_W - 2.0 * PAD, PASS_H - 8.0), wp.index - 1)
            )
            # Mini-map: where this wash goes. Fitted inside the row — a tall
            # portrait reference would otherwise run into the pass below it.
            lh, lw = pmap.labels.shape[:2]
            tw, th = float(THUMB_W), THUMB_W * lh / float(lw)
            if th > PASS_H - 14:
                th = float(PASS_H - 14)
                tw = th * lw / float(lh)
            patches.append(
                ImagePatch(
                    rgb=_map_raster(pmap, fills, ink, params.background, wp.passage_ids),
                    x=PAD, y=by, w=tw, h=th,
                )
            )
            pl.strokes.append(
                Stroke(points=_rect_pts(PAD, by, tw, th), closed=True, width=0.8, color=ink)
            )
            # Number, swatches, coverage.
            cx = PAD + THUMB_W + 18
            text(LAYER_PLAN_PASSES, cx, by + 26, f"{wp.index}", size=30.0, opacity=0.85)
            pl.strokes.append(_rect(cx + 40, by + 2, 34, 30, wp.recipe.target_hex, ink, 0.8))
            pl.strokes.append(_rect(cx + 76, by + 2, 34, 30, wp.recipe.mixed_hex, ink, 0.8))
            text(LAYER_PLAN_PASSES, cx + 40, by + 46, "target / mix", size=9.0, opacity=0.55)
            text(LAYER_PLAN_PASSES, cx, by + 66,
                 f"shape{'s' if len(wp.passage_ids) != 1 else ''} "
                 f"{', '.join(str(i) for i in wp.passage_ids)}", size=10.5, opacity=0.7)
            text(LAYER_PLAN_PASSES, cx, by + 82,
                 f"{wp.area_frac * 100:.0f}% of the paper", size=10.5, opacity=0.7)
            if not wp.recipe.achievable:
                # Short on purpose: this column is only ~130 units wide before
                # the instruction text starts.
                text(LAYER_PLAN_PASSES, cx, by + 100, "▲ closest match", size=10.0, opacity=0.7)

            # The instructions. One idea per line, each clipped to the column.
            tx = cx + 130
            avail = BOARD_W - PAD - tx
            ty = by + 14
            text(LAYER_PLAN_PASSES, tx, ty,
                 _clip(parts_text(wp.recipe, names_by_hex), avail, 14.0), size=14.0)
            ty += 20
            text(LAYER_PLAN_PASSES, tx, ty,
                 _clip(f"{wp.volume_text} · {wp.strength_text}", avail, 11.5),
                 size=11.5, opacity=0.9)
            ty += LINE
            text(LAYER_PLAN_PASSES, tx, ty,
                 _clip(f"{wp.wet} — {wp.technique}", avail, 11.0), size=11.0, opacity=0.85)
            ty += LINE
            if wp.edges:
                text(LAYER_PLAN_PASSES, tx, ty, _clip("Edges — " + wp.edges, avail, 11.0),
                     size=11.0, opacity=0.8)
                ty += LINE
            text(LAYER_PLAN_PASSES, tx, ty, _clip(wp.why, avail, 11.0), size=11.0, opacity=0.7)
            ty += LINE
            if wp.tip:
                text(LAYER_PLAN_PASSES, tx, ty, _clip("Tip — " + wp.tip, avail, 10.5),
                     size=10.5, opacity=0.62)
                ty += LINE
            text(LAYER_PLAN_PASSES, tx, ty,
                 "▸ Let this dry completely before the next wash."
                 if wp.index < len(passes) else "▸ Let it dry, then stop. Resist tidying.",
                 size=11.0, opacity=0.8)
            y = by + PASS_H
        y += GAP

        # Extra colours the artist eyedropped or added by hand.
        extra = list(recipes[len(pmap.passages):])
        if extra:
            text(LAYER_PLAN_PASSES, PAD, y + 12, "Colours you picked", size=11.5, opacity=0.7)
            ey = y + 30
            for r in extra:
                pl.strokes.append(_rect(PAD, ey - 12, 26, 16, r.target_hex, ink, 0.8))
                mix = "  +  ".join(
                    f"{int(round(s.weight * 100))}% {names_by_hex.get(s.paint_hex, s.paint_name)}"
                    for s in r.steps
                )
                text(LAYER_PLAN_PASSES, PAD + 34, ey,
                     f"{mix} · {dilution_label(r)}", size=11.0)
                ey += LINE + 2
            y = ey + GAP

    # ----- test strip + glossary ---------------------------------------------------------
    if params.show_plan_tips:
        if passes:
            y = title(LAYER_PLAN_TIPS, "Test strip — paint these first, let them dry", y)
            tl = layers[LAYER_PLAN_TIPS]
            sw, sh, gap = 64, 48, 10
            per_row = max(1, (BOARD_W - 2 * PAD + gap) // (sw + gap))
            for i, wp in enumerate(passes):
                sx = PAD + (i % per_row) * (sw + gap)
                sy = y + (i // per_row) * (sh + 34)
                tl.strokes.append(_rect(sx, sy, sw, sh, wp.recipe.mixed_hex, ink, 0.8))
                text(LAYER_PLAN_TIPS, sx + sw / 2, sy + sh + 14, f"wash {wp.index}",
                     size=10.0, anchor="middle", opacity=0.75)
            rows = (len(passes) + per_row - 1) // per_row
            y += rows * (sh + 34) + 4
            text(LAYER_PLAN_TIPS, PAD, y + 12,
                 "Watercolor dries 20–30% lighter than it looks wet. Compare these dry, "
                 "not wet, and adjust the strength before you commit.", size=10.5, opacity=0.6)
            y += 20 + GAP

        y = title(LAYER_PLAN_TIPS, "Words used in this plan", y)
        col_w = (BOARD_W - 2 * PAD) / 2.0
        for i, (term, definition) in enumerate(tips_mod.GLOSSARY):
            gx = PAD + (i % 2) * col_w
            gy = y + (i // 2) * (LINE + 2) + 12
            text(LAYER_PLAN_TIPS, gx, gy, f"{term} — {definition}", size=10.5, opacity=0.8)
        y += math.ceil(len(tips_mod.GLOSSARY) / 2.0) * (LINE + 2) + 12 + GAP

    # ----- footer ------------------------------------------------------------------------
    text(LAYER_NOTES, PAD, y + 13,
         "Passages, values and volumes are read from a photograph — they are a starting "
         "point, not a rule. Trust your eyes at the paper.", size=10.5, opacity=0.6)
    y += LINE
    text(LAYER_NOTES, PAD, y + 13, ATTRIBUTION, size=10.0, opacity=0.55)
    y += LINE
    return _finish(params, layers, patches, layout, y)


def _circle(cx: float, cy: float, r: float, n: int = 16) -> List[Tuple[float, float]]:
    return [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _finish(params, layers, patches, layout, y) -> Tuple[Scene, BoardLayout]:
    scene = Scene(
        width=BOARD_W,
        height=int(math.ceil(y + PAD)),
        background=params.background,
        layers=list(layers.values()),
        patches=patches,
        canvas_units=params.canvas_units,
    )
    if params.card_w > 0:
        w_in = params.card_w / CM_PER_INCH if params.canvas_units == UNIT_CM else params.card_w
        scene.inches_per_px = w_in / scene.width
    return scene, layout
