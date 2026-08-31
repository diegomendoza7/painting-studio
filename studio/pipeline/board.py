"""Stage 4 — compose the palette board ``Scene`` (+ its hit-test layout).

The board is the app's document: a printable study sheet. Top to bottom —
header, the artist's paints as swatch cards, the color wheel with harmony
overlays and the palette's mixable gamut, a value scale, the reference photo,
one recipe row per target, the value ladders, and a notes footer (which
carries the Mixbox attribution required by its license).

Compose is cheap and always re-runs, so everything cosmetic lives here:
visibility toggles, the selected-target highlight, harmony scheme, titles —
and, crucially, paint *names*, which are resolved from the current palette by
hex at this stage so renaming a paint never invalidates any cached stage.

``build_board`` also returns a :class:`BoardLayout` of named rects produced
by the same arithmetic that placed the content — the UI hit-tests against it,
so clicks can never drift from the picture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import mixer
from ..color import dilute, hex_to_rgb, lab_to_rgb, lch, lstar, rgb_to_hex, temperature
from ..media import spec
from ..models import MixRecipe, Paint, Target, ValueLadder
from ..params import CM_PER_INCH, Params, UNIT_CM
from ..scene import (
    ImagePatch,
    LAYER_HEADER,
    LAYER_LADDERS,
    LAYER_NOTES,
    LAYER_RECIPES,
    LAYER_REFERENCE,
    LAYER_SWATCHES,
    LAYER_WHEEL,
    Layer,
    Scene,
    Stroke,
    Text,
)

BOARD_W = 1400
PAD = 36
GAP = 26

Rect = Tuple[float, float, float, float]  # x, y, w, h

# Swatch cards.
CARD_W, CARD_H, CARD_GAP = 210, 150, 18
MASS_H, STRIP_H = 84, 26
# Wheel.
WHEEL_R = 150
WHEEL_INNER = 110
# Recipe rows / ladders.
ROW_H = 84
LADDER_SQ, LADDER_GAP = 52, 8

ATTRIBUTION = "Pigment mixing: Mixbox © Secret Weapons (scrtwpns.com) — CC BY-NC 4.0"


@dataclass
class BoardLayout:
    """Hit-test geometry, in scene coordinates."""

    photo_rect: Optional[Rect] = None            # where the photo is drawn
    photo_img_size: Tuple[int, int] = (0, 0)     # source pixels (w, h)
    swatch_rects: List[Tuple[Rect, int]] = field(default_factory=list)   # (rect, palette idx)
    target_rects: List[Tuple[Rect, int]] = field(default_factory=list)   # (rect, target idx)
    wheel: Optional[Tuple[float, float, float]] = None                   # cx, cy, r
    pass_rects: List[Tuple[Rect, int]] = field(default_factory=list)     # plan: (rect, pass idx)


# --- small geometry helpers ---------------------------------------------------
def _rect_pts(x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _rect(x, y, w, h, fill, ink, width=1.0, fill_opacity=1.0, opacity=1.0) -> Stroke:
    return Stroke(
        points=_rect_pts(x, y, w, h), closed=True, width=width, color=ink,
        fill=fill, fill_opacity=fill_opacity, opacity=opacity,
    )


def _circle_pts(cx: float, cy: float, r: float, n: int = 28) -> List[Tuple[float, float]]:
    return [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _wheel_xy(cx: float, cy: float, hue_deg: float, radius: float) -> Tuple[float, float]:
    a = math.radians(hue_deg)
    return (cx + radius * math.cos(a), cy - radius * math.sin(a))  # y axis points down


def _lch_rgb(L: float, C: float, h_deg: float) -> str:
    a = C * math.cos(math.radians(h_deg))
    b = C * math.sin(math.radians(h_deg))
    return rgb_to_hex(tuple(int(v) for v in lab_to_rgb(np.array([L, a, b]))))


def _hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain convex hull (deterministic, tiny)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def resolve_anchor(params: Params, recipes: Sequence[MixRecipe]) -> Tuple[str, str]:
    """(hex, label) the harmony overlays pivot on."""
    if params.anchor_hex:
        return params.anchor_hex, "anchor"
    if recipes:
        return recipes[0].target_hex, recipes[0].target_label or "first target"
    for p in params.palette:
        if lch(hex_to_rgb(p.hex))[1] >= 15.0:
            return p.hex, p.name
    if params.palette:
        return params.palette[0].hex, params.palette[0].name
    return "", ""


# --- the composer ----------------------------------------------------------------
def build_board(
    params: Params,
    recipes: Sequence[MixRecipe],
    ladders: Sequence[ValueLadder],
    working_rgb: Optional[np.ndarray],
    auto_targets: Sequence[Target],
) -> Tuple[Scene, BoardLayout]:
    ink = params.ink
    md = spec(params.effective_medium())
    names_by_hex = {p.hex: p.name for p in params.palette}

    def pname(hex_: str, fallback: str = "?") -> str:
        return names_by_hex.get(hex_, fallback)

    layers = {
        name: Layer(name=name)
        for name in (
            LAYER_HEADER, LAYER_SWATCHES, LAYER_WHEEL, LAYER_REFERENCE,
            LAYER_RECIPES, LAYER_LADDERS, LAYER_NOTES,
        )
    }
    layers[LAYER_SWATCHES].visible = params.show_palette
    layers[LAYER_WHEEL].visible = params.show_wheel
    layers[LAYER_REFERENCE].visible = params.show_photo
    layers[LAYER_RECIPES].visible = params.show_recipes
    layers[LAYER_LADDERS].visible = params.show_ladders
    layers[LAYER_NOTES].visible = params.show_notes
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

    # ----- header -----------------------------------------------------------------
    head = params.board_title or f"{md.label} Palette Board"
    text(LAYER_HEADER, PAD, y + 26, head, size=26.0)
    n_targets = len(recipes)
    text(
        LAYER_HEADER, PAD, y + 48,
        f"{md.label} · {len(params.palette)} paints · {n_targets} target color{'s' if n_targets != 1 else ''}",
        size=12.5, opacity=0.7,
    )
    y += 64

    # ----- palette swatch cards ------------------------------------------------------
    if params.show_palette and params.palette:
        y = title(LAYER_SWATCHES, "Your palette", y)
        white = next((p for p in params.palette if p.is_white), None)
        white_rgb = hex_to_rgb(white.hex) if white else (255, 255, 255)
        per_row = max(1, (BOARD_W - 2 * PAD + CARD_GAP) // (CARD_W + CARD_GAP))
        for i, p in enumerate(params.palette):
            cx = PAD + (i % per_row) * (CARD_W + CARD_GAP)
            cy = y + (i // per_row) * (CARD_H + CARD_GAP)
            rgb = hex_to_rgb(p.hex)
            layers[LAYER_SWATCHES].strokes.append(_rect(cx, cy, CARD_W, MASS_H, p.hex, ink, 1.0))
            layout.swatch_rects.append(((cx, cy, CARD_W, MASS_H), i))
            # Medium-appropriate secondary strips under the masstone.
            if md.dilutes:
                strip_cols = [rgb_to_hex(dilute(rgb, d)) for d in (0.7, 0.45, 0.25)]
            elif md.layered:
                strip_cols = [rgb_to_hex(mixer.lerp(rgb, (246, 244, 239), t)) for t in (0.3, 0.55, 0.8)]
            else:
                strip_cols = [rgb_to_hex(mixer.lerp(rgb, white_rgb, t)) for t in (0.25, 0.5, 0.75)]
            sw = CARD_W / 3.0
            for k, col in enumerate(strip_cols):
                layers[LAYER_SWATCHES].strokes.append(
                    _rect(cx + k * sw, cy + MASS_H, sw, STRIP_H, col, ink, 0.6)
                )
            text(LAYER_SWATCHES, cx, cy + MASS_H + STRIP_H + 16, p.name, size=11.5)
            L, C, _h = lch(rgb)
            glyphs = []
            if p.opacity:
                glyphs.append({"transparent": "T", "semi": "S", "opaque": "O"}[p.opacity])
            if p.staining:
                glyphs.append("St")
            if p.granulating:
                glyphs.append("Gr")
            t_letter = temperature(rgb)
            meta = f"L {int(round(L))}" + (f" · {t_letter}" if t_letter else "") + (
                f" · {' '.join(glyphs)}" if glyphs else ""
            )
            text(LAYER_SWATCHES, cx, cy + MASS_H + STRIP_H + 30, meta, size=10.0, opacity=0.65)
        rows = (len(params.palette) + per_row - 1) // per_row
        y += rows * (CARD_H + CARD_GAP) - CARD_GAP + GAP

    # ----- color wheel + harmony + gamut + value strip --------------------------------
    if params.show_wheel:
        y = title(LAYER_WHEEL, "Color wheel · harmony · value", y)
        cx, cy = PAD + WHEEL_R + 10, y + WHEEL_R + 10
        layout.wheel = (cx, cy, float(WHEEL_R))
        wl = layers[LAYER_WHEEL]
        # 24 hue sectors.
        for i in range(24):
            h0, h1 = i * 15.0, (i + 1) * 15.0
            outer = [_wheel_xy(cx, cy, h, WHEEL_R) for h in (h0, h0 + 5, h0 + 10, h1)]
            inner = [_wheel_xy(cx, cy, h, WHEEL_INNER) for h in (h1, h0 + 10, h0 + 5, h0)]
            wl.strokes.append(
                Stroke(points=outer + inner, closed=True, width=0.0,
                       color=ink, fill=_lch_rgb(65.0, 55.0, h0 + 7.5))
            )
        # Paint positions (and the mixable gamut hull beneath the dots).
        dot_at = {}
        for i, p in enumerate(params.palette):
            L, C, h = lch(hex_to_rgb(p.hex))
            rad = min(1.0, C / 80.0) * (WHEEL_INNER - 14)
            dot_at[i] = _wheel_xy(cx, cy, h, rad)
        if params.show_gamut and len(params.palette) >= 2:
            pts = list(dot_at.values())
            for i in range(len(params.palette)):
                for j in range(i + 1, len(params.palette)):
                    mixed = mixer.lerp(hex_to_rgb(params.palette[i].hex), hex_to_rgb(params.palette[j].hex), 0.5)
                    L, C, h = lch(mixed)
                    pts.append(_wheel_xy(cx, cy, h, min(1.0, C / 80.0) * (WHEEL_INNER - 14)))
            hull = _hull(pts)
            if len(hull) >= 3:
                wl.strokes.append(
                    Stroke(points=hull, closed=True, width=1.0, color=ink,
                           fill=ink, fill_opacity=0.06, opacity=0.8)
                )
        for i, p in enumerate(params.palette):
            px, py = dot_at[i]
            wl.strokes.append(
                Stroke(points=_circle_pts(px, py, 8, 12), closed=True,
                       width=1.2, color=ink, fill=p.hex)
            )
        # Harmony overlay around the anchor.
        anchor_hex, _anchor_label = resolve_anchor(params, recipes)
        if anchor_hex and params.harmony_scheme != "none":
            _L, _C, ha = lch(hex_to_rgb(anchor_hex))
            ring_r = (WHEEL_R + WHEEL_INNER) / 2.0
            ax, ay = _wheel_xy(cx, cy, ha, ring_r)
            wl.strokes.append(
                Stroke(points=_circle_pts(ax, ay, 11, 16), closed=True, width=2.2, color=ink)
            )
            partners = {
                "complementary": (180.0,),
                "analogous": (-30.0, 30.0),
                "triadic": (120.0, 240.0),
                "split": (150.0, 210.0),
            }.get(params.harmony_scheme, ())
            for dh in partners:
                qx, qy = _wheel_xy(cx, cy, ha + dh, ring_r)
                wl.strokes.append(Stroke(points=[(ax, ay), (qx, qy)], width=1.4, color=ink, opacity=0.7))
                wl.strokes.append(Stroke(points=_circle_pts(qx, qy, 7, 12), closed=True, width=1.6, color=ink))
            if params.harmony_scheme == "triadic" and len(partners) == 2:
                p1 = _wheel_xy(cx, cy, ha + 120.0, ring_r)
                p2 = _wheel_xy(cx, cy, ha + 240.0, ring_r)
                wl.strokes.append(Stroke(points=[p1, p2], width=1.4, color=ink, opacity=0.7))
            text(LAYER_WHEEL, cx, cy + WHEEL_R + 24, f"{params.harmony_scheme} harmony", size=10.5,
                 anchor="middle", opacity=0.65)
        # Value strip + temperature letters, paints sorted light -> dark.
        if params.show_value_strip and params.palette:
            sx0 = cx + WHEEL_R + 46
            text(LAYER_WHEEL, sx0, y + 12, "Value scale (light → dark)", size=11.0, opacity=0.65)
            order = sorted(range(len(params.palette)),
                           key=lambda i: (-lstar(hex_to_rgb(params.palette[i].hex)), i))
            per_row = max(1, int((BOARD_W - PAD - sx0) // 50))
            for rank, i in enumerate(order):
                p = params.palette[i]
                px = sx0 + (rank % per_row) * 50
                py = y + 22 + (rank // per_row) * 78
                layers[LAYER_WHEEL].strokes.append(_rect(px, py, 44, 44, p.hex, ink, 0.8))
                t_letter = temperature(hex_to_rgb(p.hex)) or "·"
                text(LAYER_WHEEL, px + 22, py + 58, t_letter, size=9.5, anchor="middle", opacity=0.6)
        y += 2 * WHEEL_R + 20 + GAP + 14

    # ----- reference photo -----------------------------------------------------------
    if params.show_photo and working_rgb is not None:
        y = title(LAYER_REFERENCE, "Reference", y)
        ih, iw = working_rgb.shape[:2]
        scale = min((BOARD_W - 2 * PAD) / iw, 300.0 / ih, 1.0)
        pw, ph = iw * scale, ih * scale
        patches.append(ImagePatch(rgb=working_rgb, x=PAD, y=y, w=pw, h=ph))
        layers[LAYER_REFERENCE].strokes.append(
            Stroke(points=_rect_pts(PAD, y, pw, ph), closed=True, width=1.0, color=ink)
        )
        layout.photo_rect = (float(PAD), y, pw, ph)
        layout.photo_img_size = (iw, ih)
        text(LAYER_REFERENCE, PAD + pw + 14, y + 14,
             "Eyedropper: click the photo to add a target color.", size=10.5, opacity=0.6)
        y += ph + GAP

    # ----- targets & recipes -----------------------------------------------------------
    if params.show_recipes:
        y = title(LAYER_RECIPES, "Targets & mixing recipes", y)
        rl = layers[LAYER_RECIPES]
        if not recipes:
            text(LAYER_RECIPES, PAD, y + 18,
                 "No targets yet — load a reference photo or use “Add color…”.", size=12.0, opacity=0.7)
            y += 36
        for idx, r in enumerate(recipes):
            ry = y + idx * ROW_H
            if idx == params.selected_target:
                rl.strokes.append(
                    _rect(PAD - 8, ry - 6, BOARD_W - 2 * PAD + 16, ROW_H - 4, None, ink, 1.8)
                )
            rl.strokes.append(_rect(PAD, ry, 56, 44, r.target_hex, ink, 1.0))
            text(LAYER_RECIPES, PAD + 56 + 12, ry + 28, "→", size=14.0, anchor="middle", opacity=0.6)
            rl.strokes.append(_rect(PAD + 80, ry, 56, 44, r.mixed_hex, ink, 1.0))
            layout.target_rects.append(((float(PAD), ry, BOARD_W - 2.0 * PAD, ROW_H - 8.0), idx))
            bx = PAD + 160
            # Line 1: label + honesty readout. Line 2: the recipe. Line 3: the
            # technique note. Stacked, so no line can ever collide with another.
            label = r.target_label or f"Target {idx + 1}"
            flag = "" if r.achievable else "  ▲ out of gamut — nearest mix shown"
            text(LAYER_RECIPES, bx, ry + 12, f"{label} · ΔE {r.delta_e:.1f}{flag}", size=11.0, opacity=0.7)
            # Weight bar beside line 1: ingredient shares as colored segments.
            bar_x = bx + 320
            if not r.layering:
                wx = bar_x
                for s in r.steps:
                    seg = 200.0 * s.weight
                    rl.strokes.append(_rect(wx, ry + 2, seg, 12, s.paint_hex, ink, 0.6))
                    wx += seg
            else:
                seg = 200.0 / max(1, len(r.steps))
                for k, s in enumerate(r.steps):
                    rl.strokes.append(_rect(bar_x + k * seg, ry + 2, seg, 12, s.paint_hex, ink, 0.6))
            if r.layering:
                if len(r.steps) == 1:
                    main = f"{pname(r.steps[0].paint_hex, r.steps[0].paint_name)} straight from the stick"
                else:
                    base, top = r.steps[0], r.steps[-1]
                    main = (
                        f"Lay down {pname(base.paint_hex, base.paint_name)}, then "
                        f"{pname(top.paint_hex, top.paint_name)} over (~{int(round(top.weight * 100))}% coverage)"
                    )
            else:
                main = "  +  ".join(
                    f"{int(round(s.weight * 100))}% {pname(s.paint_hex, s.paint_name)}" for s in r.steps
                )
            text(LAYER_RECIPES, bx, ry + 34, main, size=12.5)
            if r.note:
                text(LAYER_RECIPES, bx, ry + 52, r.note, size=10.5, opacity=0.65)
        y += max(1, len(recipes)) * ROW_H if recipes else 0
        y += GAP

    # ----- value ladders ------------------------------------------------------------------
    if params.show_ladders and ladders:
        y = title(LAYER_LADDERS, "Value ladders (light → dark)", y)
        ll = layers[LAYER_LADDERS]
        for li, ladder in enumerate(ladders):
            ly = y + 6
            label = recipes[li].target_label if li < len(recipes) else ""
            ll.strokes.append(_rect(PAD, ly + 14, 20, 20, ladder.target_hex, ink, 0.8))
            text(LAYER_LADDERS, PAD + 28, ly + 29, label or f"Target {li + 1}", size=11.0)
            x0 = PAD + 150
            for si, step in enumerate(ladder.steps):
                sx = x0 + si * (LADDER_SQ + LADDER_GAP)
                ll.strokes.append(_rect(sx, ly, LADDER_SQ, LADDER_SQ, step.hex, ink, 0.8))
                t = step.recipe_text
                if "{}" in t:
                    t = t.format(pname(step.paint_hex, "?"))
                # Three stagger levels: captions on the same level are three
                # squares apart, so even long stick names can't collide.
                ty = ly + LADDER_SQ + 13 + (si % 3) * 13
                text(LAYER_LADDERS, sx + LADDER_SQ / 2, ty, t, size=9.0, anchor="middle", opacity=0.75)
            text(LAYER_LADDERS, x0, ly + LADDER_SQ + 57, ladder.note, size=10.0, opacity=0.6)
            y += LADDER_SQ + 76
        y += GAP

    # ----- notes footer ----------------------------------------------------------------------
    if params.show_notes:
        y = title(LAYER_NOTES, f"{md.label} technique notes", y)
        for line in md.techniques:
            text(LAYER_NOTES, PAD, y + 13, "·  " + line, size=11.0, opacity=0.8)
            y += 17
        text(LAYER_NOTES, PAD, y + 13, "Mixes are estimates — test on scrap before committing.", size=10.5, opacity=0.6)
        y += 17
        text(LAYER_NOTES, PAD, y + 13, ATTRIBUTION, size=10.0, opacity=0.55)
        y += 17

    scene = Scene(
        width=BOARD_W,
        height=int(math.ceil(y + PAD)),
        background=params.background,
        layers=list(layers.values()),
        patches=patches if params.show_photo else [],
        canvas_units=params.canvas_units,
    )
    if params.card_w > 0:
        w_in = params.card_w / CM_PER_INCH if params.canvas_units == UNIT_CM else params.card_w
        scene.inches_per_px = w_in / scene.width
    return scene, layout
