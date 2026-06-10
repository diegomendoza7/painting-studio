"""Stage 3 — value ladders: how to shade each target lighter and darker.

Each ladder is ``ladder_steps`` swatches, strictly light -> dark, centered on
the solved mix, with a per-step recipe in the medium's own terms: tints toward
white for oil/acrylic, thinner washes for watercolor, stick layerings for oil
pastel. Darkening never reaches for pure black unless the artist asks
(``shade_mode="black"``) — the default partner is a transparent dark or the
complement, the way painters keep shadow color alive.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .. import mixer
from ..color import RGB, dilute, hex_to_rgb, lch, lstar, rgb_to_hex
from ..media import dilution_short, ladder_note, pressure_phrase, spec
from ..models import LadderStep, MixRecipe, Paint, ValueLadder

_PAPER: RGB = (246, 244, 239)

# Tint/shade strengths per side-count: 2 extra steps -> these mix fractions.
_LIGHT_T = {1: (0.45,), 2: (0.55, 0.28), 3: (0.65, 0.45, 0.22)}
_DARK_T = {1: (0.35,), 2: (0.22, 0.45), 3: (0.18, 0.38, 0.58)}
_WASH_D = {1: (0.45,), 2: (0.3, 0.6), 3: (0.22, 0.45, 0.7)}


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def pick_shade_partner(
    base_rgb: RGB, palette: Sequence[Paint], mode: str
) -> Optional[Paint]:
    """The paint to darken with, per ``shade_mode``. Deterministic."""
    base_l, _c, base_h = lch(base_rgb)
    cands = [p for p in palette if not p.is_white and lstar(hex_to_rgb(p.hex)) < base_l - 2.0]
    if not cands:
        return None

    def by_l(p: Paint) -> float:
        return lstar(hex_to_rgb(p.hex))

    def comp_dist(p: Paint) -> float:
        return _hue_dist(lch(hex_to_rgb(p.hex))[2], (base_h + 180.0) % 360.0)

    if mode == "black":
        darks = sorted(cands, key=lambda p: (lch(hex_to_rgb(p.hex))[1], by_l(p), p.name))
        return darks[0]
    if mode == "umber":
        earths = [
            p for p in cands
            if 20.0 <= lch(hex_to_rgb(p.hex))[2] <= 95.0 and by_l(p) < 50.0
        ]
        if earths:
            return sorted(earths, key=lambda p: (by_l(p), p.name))[0]
        mode = "auto"
    if mode == "complement":
        return sorted(cands, key=lambda p: (comp_dist(p), by_l(p), p.name))[0]
    # auto: prefer transparent darks, then the complement-closest among them.
    transparent_darks = [p for p in cands if p.opacity == "transparent" and by_l(p) < 42.0]
    pool = transparent_darks or [p for p in cands if by_l(p) < base_l - 12.0] or cands
    return sorted(pool, key=lambda p: (comp_dist(p), by_l(p), p.name))[0]


def _white_paint(palette: Sequence[Paint]) -> Optional[Paint]:
    for p in palette:
        if p.is_white:
            return p
    return None


def _build_one(
    recipe: MixRecipe,
    palette: Sequence[Paint],
    medium: str,
    steps_n: int,
    shade_mode: str,
) -> ValueLadder:
    md = spec(medium)
    base_rgb = hex_to_rgb(recipe.mixed_hex)
    n_light = (steps_n - 1) // 2
    n_dark = steps_n - 1 - n_light
    steps: List[Tuple[RGB, str, str]] = []  # (color, text template, paint hex)

    white = _white_paint(palette)
    partner = pick_shade_partner(base_rgb, palette, shade_mode)

    # Texts use "{}" where a paint name belongs; the board fills it in from the
    # *current* palette (by ``paint_hex``), so renames never touch this stage.
    # --- lighter side ---------------------------------------------------------
    if md.dilutes:
        for d in _WASH_D[n_light]:
            steps.append((dilute(base_rgb, d), f"thin to {dilution_short(d)}", ""))
    elif md.layered:
        top_rgb = hex_to_rgb(white.hex) if white else _PAPER
        top_hex = white.hex if white else ""
        for t in _LIGHT_T[n_light]:
            text = "{} over, " + pressure_phrase(t) if white else f"lighter pressure, let paper show ({pressure_phrase(t)})"
            steps.append((mixer.lerp(base_rgb, top_rgb, t), text, top_hex))
    else:
        w_rgb = hex_to_rgb(white.hex) if white else (255, 255, 255)
        w_hex = white.hex if white else ""
        for t in _LIGHT_T[n_light]:
            text = f"+{int(round(t * 100))}% " + ("{}" if white else "white")
            steps.append((mixer.lerp(base_rgb, w_rgb, t), text, w_hex))

    # --- the mix itself ---------------------------------------------------------
    base_text = f"the mix, {dilution_short(recipe.dilution)}" if md.dilutes else "the recipe mix"
    steps.append((base_rgb, base_text, ""))

    # --- darker side --------------------------------------------------------------
    if partner is None:
        # Nothing darker available: deepen with itself (pigment load / pressure).
        for t in _DARK_T[n_dark]:
            steps.append((base_rgb, "no darker paint in palette", ""))
    else:
        p_rgb = hex_to_rgb(partner.hex)
        for t in _DARK_T[n_dark]:
            if md.layered:
                text = "{} over, " + pressure_phrase(t)
            else:
                text = f"+{int(round(t * 100))}% " + "{}"
            steps.append((mixer.lerp(base_rgb, p_rgb, t), text, partner.hex))

    # Strict light -> dark ordering, whatever the mixes did.
    steps.sort(key=lambda s: -lstar(s[0]))
    return ValueLadder(
        target_hex=recipe.target_hex,
        steps=tuple(LadderStep(rgb_to_hex(rgb), text, phex) for rgb, text, phex in steps),
        note=ladder_note(medium),
    )


def build_ladders(
    recipes: Sequence[MixRecipe],
    palette: Sequence[Paint],
    medium: str,
    steps_n: int,
    shade_mode: str,
) -> Tuple[ValueLadder, ...]:
    steps_n = max(5, min(7, int(steps_n)))
    return tuple(_build_one(r, palette, medium, steps_n, shade_mode) for r in recipes)
