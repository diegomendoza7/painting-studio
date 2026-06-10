"""All user-tunable parameters in one place.

``Params`` is a plain dataclass (tuple-valued fields keep it value-comparable
and cheap to snapshot for undo). The pipeline derives per-stage cache keys
from subsets of these fields so that an edit only recomputes the stages it can
actually affect — an eyedropper pick re-solves mixes but never re-extracts the
photo; renaming a paint recomputes nothing at all.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from .media import MEDIA  # noqa: F401  (re-exported: the canonical medium list)
from .models import Paint, Target
from .palettes import OIL_STARTER, STARTERS

# Physical card measurement units.
UNIT_IN = "in"
UNIT_CM = "cm"
CM_PER_INCH = 2.54

SHADE_MODES = ("auto", "complement", "umber", "black")
HARMONY_SCHEMES = ("none", "complementary", "analogous", "triadic", "split")


@dataclass
class Params:
    """The complete adjustable state of the palette board."""

    # --- document ----------------------------------------------------------
    medium: str = "oil"
    max_edge: int = 1280            # longest edge of the working photo (px)

    # --- the artist's available paints (edited via the palette panel) ------
    palette: Tuple[Paint, ...] = OIL_STARTER

    # --- targets ------------------------------------------------------------
    extract_enabled: bool = True    # auto-extract a palette from the photo
    extract_n: int = 6              # number of auto-extracted colors (3..12)
    manual_targets: Tuple[Target, ...] = ()

    # --- mix solver ----------------------------------------------------------
    max_paints: int = 3             # paints per mix: 2 | 3

    # --- value ladder --------------------------------------------------------
    ladder_steps: int = 5           # 5..7 steps light -> dark
    shade_mode: str = "auto"        # auto | complement | umber | black

    # --- wheel / harmony (compose-only) --------------------------------------
    harmony_scheme: str = "complementary"
    anchor_hex: str = ""            # "" -> first target, else first chromatic paint
    show_gamut: bool = True

    # --- board regions (compose-only) ----------------------------------------
    show_photo: bool = True
    show_palette: bool = True
    show_wheel: bool = True
    show_recipes: bool = True
    show_ladders: bool = True
    show_value_strip: bool = True
    show_notes: bool = True
    board_title: str = ""

    # --- physical card size (compose-only; drives to-scale export) -----------
    card_w: float = 0.0             # width in canvas_units; 0 = unset (px export)
    canvas_units: str = UNIT_IN

    # --- style / transient ----------------------------------------------------
    background: str = "#f6f3ea"     # warm paper card
    ink: str = "#1b1b1b"
    eyedropper_mode: bool = False   # transient UI state; belongs in no cache key
    selected_target: int = -1       # board highlight only (compose-only)

    # ----------------------------------------------------------------------
    def copy(self, **changes) -> "Params":
        return replace(self, **changes)

    # Cache keys: each tuple captures *only* the fields a stage depends on,
    # plus the key of the stage above it (chained by the pipeline). Two Params
    # with equal keys for a stage must produce identical output there.
    def norm_key(self) -> tuple:
        return (self.max_edge,)

    def extract_key(self) -> tuple:
        return (self.extract_enabled, self.extract_n)

    def palette_mix_sig(self) -> tuple:
        # Only what changes a mix's *color math*: hex, whiteness, opacity class.
        # A paint's name is read at compose time, so renaming recomputes nothing.
        return tuple((p.hex, p.is_white, p.opacity, p.staining) for p in self.palette)

    def solve_key(self) -> tuple:
        return (
            self.medium,
            self.palette_mix_sig(),
            tuple(t.hex for t in self.manual_targets),
            self.max_paints,
        )

    def ladder_key(self) -> tuple:
        return (self.ladder_steps, self.shade_mode)


# Built-in presets: a sensible starting kit per medium. Each switches the
# medium *and* loads its starter palette; board/solver style is left alone.
PRESETS: dict[str, dict] = {
    "Oil — split primary 8": dict(medium="oil", palette=STARTERS["oil"]),
    "Watercolor — 12": dict(medium="watercolor", palette=STARTERS["watercolor"]),
    "Acrylic — basic 6": dict(medium="acrylic", palette=STARTERS["acrylic"]),
    "Oil pastel — 24": dict(medium="oilpastel", palette=STARTERS["oilpastel"], max_paints=2),
}


def apply_preset(params: Params, name: str) -> Params:
    """Return a copy of ``params`` with the named built-in preset applied."""
    changes = PRESETS.get(name)
    return params.copy(**changes) if changes else params


# A user-saved preset captures the board/solver "look" — never the medium, the
# palette, or the picked targets, so applying one can't silently swap the
# artist's physical paints or throw away their color picks.
PRESET_FIELDS: tuple[str, ...] = (
    "max_paints", "ladder_steps", "shade_mode",
    "harmony_scheme", "show_gamut",
    "show_photo", "show_palette", "show_wheel", "show_recipes",
    "show_ladders", "show_value_strip", "show_notes",
    "extract_enabled", "extract_n",
)


def preset_to_dict(params: Params) -> dict:
    """The PRESET_FIELDS of ``params`` as a JSON-able dict (a saved user preset)."""
    return {k: getattr(params, k) for k in PRESET_FIELDS}


def apply_preset_values(params: Params, values: dict) -> Params:
    """Apply a saved preset dict onto ``params`` (only the known PRESET_FIELDS)."""
    changes = {k: v for k, v in values.items() if k in PRESET_FIELDS}
    return params.copy(**changes) if changes else params
