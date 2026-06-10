"""Core domain models.

Everything here is a *frozen* dataclass: instances sit inside ``Params``
tuples, feed the pipeline's stage cache keys, and snapshot into undo history,
all of which need value equality and hashability. Only ``Paint`` and
``Target`` are persisted (inside ``Params``); recipes and ladders are derived
pipeline outputs and are never serialized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Tuple

# Paint opacity classes (informational; drives technique notes + swatch glyphs).
OPACITY_TRANSPARENT = "transparent"
OPACITY_SEMI = "semi"
OPACITY_OPAQUE = "opaque"


@dataclass(frozen=True)
class Paint:
    """One physical paint: a tube color or a pastel stick."""

    name: str
    hex: str
    is_white: bool = False        # the palette's white (excluded from watercolor mixes)
    opacity: str = ""             # "" | transparent | semi | opaque
    staining: bool = False
    granulating: bool = False


@dataclass(frozen=True)
class Target:
    """A color the artist wants to achieve."""

    hex: str
    label: str = ""
    source: str = "manual"        # manual | eyedrop | auto (auto = pipeline output only)


@dataclass(frozen=True)
class MixStep:
    """One ingredient of a mix (or one layer of an oil-pastel build, bottom->top)."""

    paint_name: str
    paint_hex: str
    weight: float                 # fraction of the mix, display-rounded to 5%


@dataclass(frozen=True)
class MixRecipe:
    """How to reach a target color from the available palette."""

    target_hex: str
    target_label: str
    steps: Tuple[MixStep, ...]
    mixed_hex: str                # what the recipe actually produces
    delta_e: float                # CIEDE2000(target, mixed) for the printed recipe
    dilution: float = 1.0         # watercolor: optical thickness (1 = masstone)
    layering: bool = False        # oil pastel: steps are a surface layering order
    note: str = ""                # medium-specific technique guidance
    achievable: bool = True       # delta_e within the achievability threshold


@dataclass(frozen=True)
class LadderStep:
    hex: str
    recipe_text: str              # "+25% {}" / "thin: pale wash" — "{}" is filled
    paint_hex: str = ""           # ... with the *current* name of this paint at
    #                               compose time, so renaming a paint never
    #                               invalidates a cached ladder.


@dataclass(frozen=True)
class ValueLadder:
    """Light -> dark shading steps for one target, with per-step recipes."""

    target_hex: str
    steps: Tuple[LadderStep, ...]
    note: str = ""                # one-line medium technique reminder


# --- JSON helpers (session / presets) ---------------------------------------
_PAINT_FIELDS = {f.name for f in fields(Paint)}
_TARGET_FIELDS = {f.name for f in fields(Target)}


def paint_to_dict(p: Paint) -> dict:
    return asdict(p)


def paint_from_dict(d: dict) -> Paint:
    return Paint(**{k: v for k, v in d.items() if k in _PAINT_FIELDS})


def target_to_dict(t: Target) -> dict:
    return asdict(t)


def target_from_dict(d: dict) -> Target:
    return Target(**{k: v for k, v in d.items() if k in _TARGET_FIELDS})
