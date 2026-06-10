"""Medium-specific rules and technique language.

One :class:`MediumSpec` per medium says how mixes may be built (white allowed?
dilution? surface layering?) and supplies the guidance lines the board prints.
The note generators here turn a solved recipe's numbers into the words an
artist actually uses at the easel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .models import MixStep, Paint


@dataclass(frozen=True)
class MediumSpec:
    key: str
    label: str
    allows_white: bool   # white paint may appear inside a palette mix
    dilutes: bool        # lighten via water dilution (paper is the white)
    layered: bool        # "mixing" = ordered layering on the surface (oil pastel)
    lighten: str         # one-line lighten guidance
    darken: str          # one-line darken guidance
    techniques: Tuple[str, ...]


SPECS: dict[str, MediumSpec] = {
    "watercolor": MediumSpec(
        key="watercolor",
        label="Watercolor",
        allows_white=False,
        dilutes=True,
        layered=False,
        lighten="Lighten by dilution — more water; the paper is your white.",
        darken="Darken with a stronger pigment load or a staining dark, never black alone.",
        techniques=(
            "Flat and graded washes for large value masses.",
            "Glaze in layers — let each wash dry before the next.",
            "Wet-on-wet for soft edges; drybrush for broken texture.",
            "Lift lights with a damp brush before staining colors set.",
        ),
    ),
    "acrylic": MediumSpec(
        key="acrylic",
        label="Acrylic",
        allows_white=True,
        dilutes=False,
        layered=False,
        lighten="Lighten with titanium white; keep mixes fresh — it dries fast.",
        darken="Darken with burnt umber or the complement; black deadens chroma.",
        techniques=(
            "Acrylic dries slightly darker — mix a touch lighter than the target.",
            "Build in opaque layers; earlier passes are quickly safe to paint over.",
            "Glaze with medium for transparent shifts; scumble lights over dry darks.",
            "Drybrush for texture; keep a mist bottle to extend palette open time.",
        ),
    ),
    "oil": MediumSpec(
        key="oil",
        label="Oil",
        allows_white=True,
        dilutes=False,
        layered=False,
        lighten="Lighten with white — guard against chalkiness; a warm light pigment can help.",
        darken="Darken with transparent darks (umber, alizarin + ultramarine), not black.",
        techniques=(
            "Fat over lean: each layer slightly oilier than the one below.",
            "Alla prima: finish wet-in-wet in one sitting for fresh color.",
            "Glaze transparent darks over dry paint for depth in the shadows.",
            "Scumble or impasto the lights; keep shadows thin and transparent.",
        ),
    ),
    "oilpastel": MediumSpec(
        key="oilpastel",
        label="Oil Pastel",
        allows_white=True,
        dilutes=False,
        layered=True,
        lighten="Layer a white or lighter stick over with light pressure, or let paper show.",
        darken="Layer a darker stick over; burnish to fuse the layers.",
        techniques=(
            "Mix on the surface: lay the base color, layer the second over it.",
            "Pressure is your mixing ratio — press harder for more of the top color.",
            "Blend with a finger or torchon; burnish to fuse and gloss.",
            "Sgraffito: scratch through a top layer to reveal the color beneath.",
        ),
    ),
}

MEDIA: Tuple[str, ...] = tuple(SPECS)


def spec(medium: str) -> MediumSpec:
    return SPECS[medium]


def dilution_phrase(d: float) -> str:
    """Turn an optical-thickness fraction into wash language."""
    if d >= 0.85:
        return "full strength (just enough water to flow)"
    if d >= 0.55:
        return "creamy wash (about 1 part paint : 1 water)"
    if d >= 0.30:
        return "fluid wash (about 1 part paint : 3 water)"
    return "pale wash (1 part paint : 5+ water)"


def dilution_short(d: float) -> str:
    """Compact wash label for ladder-step captions."""
    if d >= 0.85:
        return "full strength"
    if d >= 0.55:
        return "1:1 wash"
    if d >= 0.30:
        return "1:3 wash"
    return "1:5 wash"


def pressure_phrase(t: float) -> str:
    """Oil pastel: top-layer dominance -> pressure wording."""
    if t >= 0.75:
        return "heavy pressure"
    if t >= 0.6:
        return "medium pressure"
    return "light pressure"


def recipe_note(
    medium: str,
    steps: Sequence[MixStep],
    palette: Sequence[Paint],
    dilution: float,
    layering: bool,
) -> str:
    """Medium-aware guidance line for one solved recipe."""
    by_hex = {p.hex: p for p in palette}
    parts: list[str] = []
    if medium == "watercolor":
        parts.append(dilution_phrase(dilution))
        stainers = [s.paint_name for s in steps if by_hex.get(s.paint_hex, Paint("", "")).staining]
        if stainers:
            parts.append(f"{', '.join(stainers)} stain{'s' if len(stainers) == 1 else ''} — plan the wash, lifts poorly")
    elif medium == "oil":
        white_w = sum(s.weight for s in steps if by_hex.get(s.paint_hex, Paint("", "")).is_white)
        if white_w > 0.4:
            parts.append("white-heavy tint — guard against chalkiness (a touch of yellow ochre keeps it warm)")
        if any(by_hex.get(s.paint_hex, Paint("", "")).opacity == "transparent" for s in steps):
            parts.append("transparent ingredients — also works as a glaze over dry paint")
    elif medium == "acrylic":
        parts.append("dries a touch darker — mix slightly lighter than the target")
    elif medium == "oilpastel" and layering and len(steps) >= 2:
        t = steps[-1].weight
        parts.append(f"{pressure_phrase(t)} on the top stick; blend with a finger or torchon")
    return "; ".join(parts)


def ladder_note(medium: str) -> str:
    s = spec(medium)
    return f"{s.label}: {s.lighten.rstrip('.')} · {s.darken.rstrip('.')}"
