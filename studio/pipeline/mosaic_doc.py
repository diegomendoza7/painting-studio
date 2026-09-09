"""Mosaic stage B — compose the mosaic guide, and the to-scale grid sheet.

Two documents come out of here.

:func:`build_mosaic` is the reference sheet, composed on every render: the
photo, the mosaic as it will actually look in the artist's own paints, the
numbered grid, and a legend that gives each colour a real recipe. Cheap, so all
the cosmetics live here — section toggles, the selected-colour highlight, and
paint *names*, resolved from the current palette by hex at this stage so
renaming a paint never invalidates a cached stage.

:func:`build_grid_sheet` is the sheet you actually paint from: numbered cells
and nothing else, carrying a physical scale so it prints at true tessera size
across as many pages as it takes. It is built **only on export** — a 40×40 grid
is 1600 rectangles plus 1600 labels, which has no business in the live render
path.

Under the ``"flow"`` layout the guide is a **cartoon** — every stone outlined at
true size with its colour number inside — because a course that follows the
drawing cannot be addressed by row and column. Under ``"grid"`` it stays a
lettered lattice you can rule onto paper.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..color import hex_to_rgb, rgb_to_hex
from ..media import dilution_short
from ..models import MixRecipe
from ..params import CM_PER_INCH, Params, UNIT_CM
from ..scene import (
    ImagePatch,
    LAYER_HEADER,
    LAYER_MOSAIC_GRID,
    LAYER_MOSAIC_LEGEND,
    LAYER_MOSAIC_TESSERAE,
    LAYER_NOTES,
    Layer,
    Scene,
    Stroke,
    Text,
)
from .. import tips as tips_mod
from .board import ATTRIBUTION, BOARD_W, GAP, PAD, BoardLayout, _rect, _rect_pts
from .mosaic_flow import MIN_GROUT
from .mosaic_grid import LAYOUT_FLOW, MosaicGrid
from .plan_schedule import parts_text

MM_PER_INCH = 25.4
PANE_W = 620                # photo / preview panes, side by side
MAX_CELL = 36.0             # on-screen grid cell (scene units)
GUTTER = 26.0               # room for the row letters down the left
LEGEND_H = 34.0
LINE = 16.0
SHEET_CELL = 40.0           # grid-sheet cell; the physical scale is derived from it
_JITTER_SEED = 1607
_CHAR_W = 0.52


def _clip(s: str, width: float, size: float) -> str:
    n = max(6, int(width / (size * _CHAR_W)))
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;·") + "…"


def _dim(hex_: str, paper: str, keep: float = 0.18) -> str:
    """Fade a colour back toward the paper, for the un-selected tesserae."""
    a, b = hex_to_rgb(hex_), hex_to_rgb(paper)
    return rgb_to_hex(tuple(a[i] * keep + b[i] * (1.0 - keep) for i in range(3)))


def col_label(i: int) -> str:
    return str(i + 1)


def row_label(i: int) -> str:
    """A..Z, then AA, AB… so a tall grid never repeats a row name."""
    out = ""
    n = i
    while True:
        out = chr(ord("A") + n % 26) + out
        n = n // 26 - 1
        if n < 0:
            return out


def cell_colors(grid: MosaicGrid, recipes: Sequence[MixRecipe]) -> Tuple[str, ...]:
    """The colour each index will actually be painted — the achievable mix."""
    return tuple(
        recipes[i].mixed_hex if i < len(recipes) else c
        for i, c in enumerate(grid.colors)
    )


# --- tesserae ------------------------------------------------------------------
def _jitter_field(n: int) -> np.ndarray:
    """(n, 4, 2) corner offsets in [-1, 1], stable for a given stone count."""
    rng = np.random.default_rng(_JITTER_SEED)
    return rng.uniform(-1.0, 1.0, size=(max(1, n), 4, 2))


def draw_grout(grout_frac: float) -> float:
    """The grout actually drawn.

    Floored at :data:`~studio.pipeline.mosaic_flow.MIN_GROUT` for two reasons
    that happen to agree: the flow layout lets footprints overlap slightly so
    the courses pack, and this margin is what keeps the *painted* squares clear
    of each other — and in watercolor two wet squares that touch bleed into one,
    so a visible gap is a requirement of the medium, not a style choice.
    """
    return max(MIN_GROUT, min(0.6, float(grout_frac)))


def tesserae_strokes(
    grid: MosaicGrid,
    colors: Sequence[str],
    x: float,
    y: float,
    w: float,
    h: float,
    grout_frac: float,
    jitter: float,
    paper: str,
    highlight: int = -1,
    outline_only: bool = False,
    ink: str = "#1b1b1b",
) -> List[Stroke]:
    """One stone per tessera: its footprint, inset by the grout and roughened.

    ``jitter`` is a fraction of the *grout*, so a corner can move at most half
    the gap it was given and a stone can never cross out of its own footprint —
    the gap narrows but never closes, at any setting.
    """
    g = draw_grout(grout_frac)
    j = max(0.0, min(1.0, jitter))
    field = _jitter_field(len(grid.tesserae))
    out: List[Stroke] = []
    for i, te in enumerate(grid.tesserae):
        pts = [(x + px * w, y + py * h) for px, py in te.points]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        # Half the grout is the slack each corner has inside its footprint.
        edge = math.dist(pts[0], pts[1]) or 1.0
        amp = j * (g / 2.0) * edge
        d = field[i]
        shrunk = [
            (mx + (px - mx) * (1.0 - g) + d[c][0] * amp,
             my + (py - my) * (1.0 - g) + d[c][1] * amp)
            for c, (px, py) in enumerate(pts)
        ]
        if outline_only:
            out.append(Stroke(points=shrunk, closed=True, width=0.7, color=ink))
            continue
        fill = colors[te.color] if te.color < len(colors) else paper
        if highlight >= 0 and te.color != highlight:
            fill = _dim(fill, paper)
        out.append(Stroke(points=shrunk, closed=True, width=0.0, color=fill, fill=fill))
    return out


def cartoon_strokes(
    grid: MosaicGrid, x: float, y: float, w: float, h: float, grout_frac: float,
    ink: str, texts, highlight: int = -1,
) -> List[Stroke]:
    """The mosaicist's cartoon: every stone outlined, with its colour number.

    This is what replaces the lettered grid when the courses follow the drawing
    — you cannot address a flowing course by row and column, but you can trace
    the stones, which is how a mosaic is actually transferred.
    """
    g = draw_grout(grout_frac)
    out: List[Stroke] = []
    for te in grid.tesserae:
        pts = [(x + px * w, y + py * h) for px, py in te.points]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        shrunk = [(mx + (px - mx) * (1.0 - g), my + (py - my) * (1.0 - g)) for px, py in pts]
        on = highlight < 0 or te.color == highlight
        out.append(Stroke(points=shrunk, closed=True, width=0.7, color=ink,
                          fill=ink if (highlight >= 0 and on) else None,
                          fill_opacity=0.10, opacity=1.0 if on else 0.4))
        # Sized per stone: the filling passes cut smaller tesserae, and one
        # number size for all of them would either overflow the small ones or
        # look lost in the full-size ones.
        size = max(5.0, min(13.0, math.dist(pts[0], pts[1]) * (1.0 - g) * 0.5))
        texts(mx, my + size * 0.36, str(te.color + 1), size=size, anchor="middle",
              opacity=1.0 if on else 0.35)
    return out


def _numbered_grid(
    grid: MosaicGrid, layer: Layer, texts, x: float, y: float, cell: float, ink: str,
    highlight: int = -1,
) -> None:
    """The lattice layout's sheet: plain cells, numbers, row and column labels."""
    if grid.cells is None:
        return
    label_size = min(13.0, max(6.0, cell * 0.42))
    for c in range(grid.cols):
        texts(x + c * cell + cell / 2, y - 6, col_label(c), size=min(10.0, label_size),
              anchor="middle", opacity=0.6)
    for r in range(grid.rows):
        texts(x - 8, y + r * cell + cell / 2 + label_size * 0.35, row_label(r),
              size=min(10.0, label_size), anchor="end", opacity=0.6)
    for r in range(grid.rows):
        for c in range(grid.cols):
            idx = int(grid.cells[r, c])
            cx, cy = x + c * cell, y + r * cell
            on = highlight < 0 or idx == highlight
            if highlight >= 0 and on:
                layer.strokes.append(_rect(cx, cy, cell, cell, ink, ink, 0.6,
                                           fill_opacity=0.10))
            else:
                layer.strokes.append(_rect(cx, cy, cell, cell, None, ink, 0.6,
                                           opacity=0.55 if not on else 1.0))
            texts(cx + cell / 2, cy + cell / 2 + label_size * 0.36, str(idx + 1),
                  size=label_size, anchor="middle", opacity=1.0 if on else 0.35)


# --- the reference document ---------------------------------------------------
def build_mosaic(
    params: Params,
    grid: Optional[MosaicGrid],
    recipes: Sequence[MixRecipe],
    working_rgb: Optional[np.ndarray],
) -> Tuple[Scene, BoardLayout]:
    ink = params.ink
    names_by_hex = {p.hex: p.name for p in params.palette}
    layers = {
        name: Layer(name=name)
        for name in (LAYER_HEADER, LAYER_MOSAIC_TESSERAE, LAYER_MOSAIC_GRID,
                     LAYER_MOSAIC_LEGEND, LAYER_NOTES)
    }
    layers[LAYER_MOSAIC_TESSERAE].visible = params.show_mosaic_preview
    layers[LAYER_MOSAIC_GRID].visible = params.show_mosaic_grid
    layers[LAYER_MOSAIC_LEGEND].visible = params.show_mosaic_legend
    layout = BoardLayout()
    patches: List[ImagePatch] = []
    y = float(PAD)

    def text(layer, x, ty, s, size=12.0, anchor="start", opacity=1.0, color=None):
        layers[layer].texts.append(
            Text(x=x, y=ty, text=s, size=size, color=color or ink, anchor=anchor,
                 opacity=opacity)
        )

    def title(layer, s, ty) -> float:
        text(layer, PAD, ty + 14, s.upper(), size=13.0, opacity=0.65)
        return ty + 24

    # ----- header ---------------------------------------------------------------
    text(LAYER_HEADER, PAD, y + 26, params.board_title or "Mosaic Guide", size=26.0)
    if grid is None or working_rgb is None:
        text(LAYER_HEADER, PAD, y + 48,
             "Open a reference photo to build a mosaic guide.", size=14.0, opacity=0.8)
        text(LAYER_HEADER, PAD, y + 70,
             "The photo becomes a grid of tesserae, each one a square of watercolor "
             "mixed from your own paints.", size=11.5, opacity=0.6)
        return _finish(params, layers, patches, layout, y + 92)

    unit, per = _unit_and_size(params)
    fw, fh = grid.cols * per, grid.rows * per
    text(LAYER_HEADER, PAD, y + 48,
         f"{grid.total} tesserae · {len(grid.colors)} colours · "
         f"{params.tessera_mm:g} mm each · finished size {fw:.0f} × {fh:.0f} {unit}",
         size=12.5, opacity=0.7)
    y += 64

    # ----- photo + mosaic preview ------------------------------------------------
    if params.show_mosaic_preview:
        y = title(LAYER_MOSAIC_TESSERAE, "The photo, and the mosaic in your paints", y)
        ih, iw = working_rgb.shape[:2]
        ph = PANE_W * ih / float(iw)
        patches.append(ImagePatch(rgb=working_rgb, x=PAD, y=y, w=PANE_W, h=ph))
        layers[LAYER_MOSAIC_TESSERAE].strokes.append(
            Stroke(points=_rect_pts(PAD, y, PANE_W, ph), closed=True, width=1.0, color=ink)
        )
        layout.photo_rect = (float(PAD), y, float(PANE_W), ph)
        layout.photo_img_size = (iw, ih)

        mx = PAD + PANE_W + 24
        mh = PANE_W * grid.rows / float(grid.cols)
        layers[LAYER_MOSAIC_TESSERAE].strokes.extend(
            tesserae_strokes(
                grid, cell_colors(grid, recipes), mx, y, float(PANE_W), mh,
                params.grout_frac, params.tessera_jitter, params.background,
                params.selected_color,
            )
        )
        text(LAYER_MOSAIC_TESSERAE, PAD, y + max(ph, mh) + 15, "Your reference.",
             size=10.5, opacity=0.6)
        text(LAYER_MOSAIC_TESSERAE, mx, y + max(ph, mh) + 15,
             "Every square is one brushstroke; the gaps are bare paper.",
             size=10.5, opacity=0.6)
        y += max(ph, mh) + 30 + GAP

    # ----- the numbered grid -------------------------------------------------------
    if params.show_mosaic_grid:
        gl = layers[LAYER_MOSAIC_GRID]
        emit = lambda tx, ty, ts, **kw: text(LAYER_MOSAIC_GRID, tx, ty, ts, **kw)  # noqa: E731
        cell = min(MAX_CELL, (BOARD_W - 2 * PAD - GUTTER) / grid.cols)
        if grid.layout == LAYOUT_FLOW:
            y = title(LAYER_MOSAIC_GRID, "The cartoon — paint by number", y)
            cw = grid.cols * cell
            ch = grid.rows * cell
            gl.strokes.extend(cartoon_strokes(
                grid, PAD, y + 12, cw, ch, params.grout_frac, ink, emit,
                params.selected_color))
            y += 12 + ch + 18
            note = ("Trace or print this at scale, then paint one stone at a time. "
                    "There are no rows to count — the courses follow the drawing, "
                    "which is what makes it read as a picture even before it has "
                    "any colour.")
        else:
            y = title(LAYER_MOSAIC_GRID, "The grid — paint by number", y)
            _numbered_grid(grid, gl, emit, PAD + GUTTER, y + 12, cell, ink,
                           params.selected_color)
            y += 12 + grid.rows * cell + 18
            note = "Rows are lettered, columns numbered."
        text(LAYER_MOSAIC_GRID, PAD, y + 12,
             _clip(note, BOARD_W - 2 * PAD, 10.5), size=10.5, opacity=0.6)
        text(LAYER_MOSAIC_GRID, PAD, y + 12 + LINE,
             "Export the grid to scale to print it at its true size across several "
             "sheets you tape together.", size=10.5, opacity=0.6)
        y += 20 + LINE + GAP

    # ----- legend --------------------------------------------------------------------
    if params.show_mosaic_legend:
        y = title(LAYER_MOSAIC_LEGEND, "Your colours — mix these", y)
        ll = layers[LAYER_MOSAIC_LEGEND]
        mixed = cell_colors(grid, recipes)
        for i, target in enumerate(grid.colors):
            ry = y + i * LEGEND_H
            if i == params.selected_color:
                ll.strokes.append(
                    _rect(PAD - 8, ry - 4, BOARD_W - 2 * PAD + 16, LEGEND_H - 2,
                          None, ink, 1.8)
                )
            layout.color_rects.append(
                ((float(PAD), ry - 4, BOARD_W - 2.0 * PAD, LEGEND_H - 2.0), i)
            )
            text(LAYER_MOSAIC_LEGEND, PAD + 10, ry + 19, str(i + 1), size=15.0,
                 anchor="middle")
            ll.strokes.append(_rect(PAD + 26, ry, 34, 26, target, ink, 0.8))
            ll.strokes.append(_rect(PAD + 64, ry, 34, 26, mixed[i], ink, 0.8))
            r = recipes[i] if i < len(recipes) else None
            if r is None:
                continue
            text(LAYER_MOSAIC_LEGEND, PAD + 112, ry + 18,
                 _clip(parts_text(r, names_by_hex), 520, 12.5), size=12.5)
            text(LAYER_MOSAIC_LEGEND, PAD + 648, ry + 18,
                 dilution_short(r.dilution), size=11.0, opacity=0.8)
            text(LAYER_MOSAIC_LEGEND, PAD + 790, ry + 18,
                 f"{grid.counts[i]} tiles · {grid.fraction(i) * 100:.0f}%",
                 size=11.0, opacity=0.75)
            flag = "" if r.achievable else "  ▲ closest match"
            text(LAYER_MOSAIC_LEGEND, PAD + 960, ry + 18,
                 f"ΔE {r.delta_e:.1f}{flag}", size=11.0, opacity=0.7)
        y += len(grid.colors) * LEGEND_H + 8
        text(LAYER_MOSAIC_LEGEND, PAD, y + 12,
             "Left swatch is the photo's colour, right is what your palette reaches. "
             "Colours are numbered lightest first, so painting in order is also "
             "painting light to dark.", size=10.5, opacity=0.6)
        y += 20 + GAP

    # ----- how to paint it -------------------------------------------------------------
    y = title(LAYER_NOTES, "How to paint it", y)
    for line in tips_mod.MOSAIC_NOTES:
        text(LAYER_NOTES, PAD, y + 13, "·  " + _clip(line, BOARD_W - 2 * PAD - 14, 11.5),
             size=11.5, opacity=0.85)
        y += LINE + 1
    y += 6
    text(LAYER_NOTES, PAD, y + 13, ATTRIBUTION, size=10.0, opacity=0.55)
    y += LINE
    return _finish(params, layers, patches, layout, y)


# --- the to-scale sheet ------------------------------------------------------------
def build_grid_sheet(params: Params, grid: MosaicGrid) -> Scene:
    """The numbered grid alone, carrying a true physical scale.

    Built on export only. ``inches_per_px`` is derived from ``tessera_mm`` over
    :data:`SHEET_CELL`, so one scene cell prints as one painted square whatever
    the paper.
    """
    ink = params.ink
    layer = Layer(name=LAYER_MOSAIC_GRID)
    head = Layer(name=LAYER_HEADER)
    x = float(PAD) + GUTTER
    y = float(PAD) + 34.0

    head.texts.append(Text(
        x=PAD, y=float(PAD) + 16, size=13.0, color=ink,
        text=(f"{params.board_title or 'Mosaic Guide'} — "
              f"{grid.cols} × {grid.rows} tesserae at {params.tessera_mm:g} mm"),
    ))

    def texts(tx, ty, s, size=12.0, anchor="start", opacity=1.0):
        layer.texts.append(Text(x=tx, y=ty, text=s, size=size, color=ink,
                                anchor=anchor, opacity=opacity))

    if grid.layout == LAYOUT_FLOW:
        x = float(PAD)
        layer.strokes.extend(cartoon_strokes(
            grid, x, y, grid.cols * SHEET_CELL, grid.rows * SHEET_CELL,
            params.grout_frac, ink, texts))
    else:
        _numbered_grid(grid, layer, texts, x, y, SHEET_CELL, ink)
    width = int(math.ceil(x + grid.cols * SHEET_CELL + PAD))
    height = int(math.ceil(y + grid.rows * SHEET_CELL + PAD))
    scene = Scene(
        width=width, height=height, background=params.background,
        layers=[head, layer], canvas_units=params.canvas_units,
    )
    # One SHEET_CELL == one tessera, physically.
    scene.inches_per_px = (float(params.tessera_mm) / MM_PER_INCH) / SHEET_CELL
    return scene


# --- helpers -----------------------------------------------------------------------
def _unit_and_size(params: Params) -> Tuple[str, float]:
    """The label and per-tessera size for the finished-dimensions readout."""
    if params.canvas_units == UNIT_CM:
        return "cm", float(params.tessera_mm) / 10.0
    return "in", float(params.tessera_mm) / MM_PER_INCH


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
