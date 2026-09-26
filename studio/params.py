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

# The two document types. "board" is the palette study sheet; "plan" is the
# ordered watercolor painting plan for the loaded photo.
DOC_BOARD = "board"
DOC_PLAN = "plan"
DOC_MOSAIC = "mosaic"
DOC_MODES = (DOC_BOARD, DOC_PLAN, DOC_MOSAIC)
# The two watercolor documents: both always solve and speak watercolor.
WATERCOLOR_DOCS = (DOC_PLAN, DOC_MOSAIC)


@dataclass
class Params:
    """The complete adjustable state of the palette board."""

    # --- document ----------------------------------------------------------
    doc_mode: str = DOC_BOARD
    medium: str = "oil"
    max_edge: int = 1280            # longest edge of the working photo (px)

    # --- the artist's available paints (edited via the palette panel) ------
    palette: Tuple[Paint, ...] = OIL_STARTER

    # --- targets ------------------------------------------------------------
    extract_enabled: bool = True    # auto-extract a palette from the photo
    extract_n: int = 6              # number of auto-extracted colors (3..12)
    manual_targets: Tuple[Target, ...] = ()

    # --- mix solver ----------------------------------------------------------
    max_paints: int = 3             # paints per mix: 2 | 3 | 4

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

    # --- painting plan --------------------------------------------------------
    wash_tiers: int = 4             # value layers, light -> dark (3..6)
    plan_colors: int = 7            # colour clusters the passages are cut from (3..12)
    min_passage_pct: float = 1.5    # merge any shape smaller than this % of the image
    # Paper size drives the wash volumes — non-zero by default so the plan can
    # answer "how much?" the moment it opens.
    paper_w: float = 10.0           # in canvas_units
    paper_h: float = 7.0
    show_plan_map: bool = True
    show_plan_passes: bool = True
    show_plan_tips: bool = True
    selected_pass: int = -1         # plan highlight only (compose-only)

    # --- mosaic guide ----------------------------------------------------------
    tessera_layout: str = "flow"    # "flow" = courses follow the drawing; "grid" = lattice
    tiles_x: int = 40               # tesserae across; rows follow the photo's aspect
    mosaic_colors: int = 12         # colours the mosaic is reduced to (4..20)
    tessera_mm: float = 12.0        # painted size of one square — drives the to-scale print
    grout_frac: float = 0.12        # gap between tesserae, as a fraction of a stone
    # Jitter is a fraction of the *grout*, never of the cell: two neighbours can
    # each move a corner by at most half the gap before they touch, so this
    # makes containment true by construction. It also means the grout is what
    # buys the irregularity — a tight grout can only ever look machine-cut.
    tessera_jitter: float = 0.85
    show_mosaic_preview: bool = True
    show_mosaic_grid: bool = True
    show_mosaic_legend: bool = True
    selected_color: int = -1        # mosaic highlight only (compose-only)

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

    # The painting plan and the mosaic guide *are* the watercolor coach, so they
    # always solve and speak watercolor whatever the medium selector says.
    # Everything that branches on the medium reads this, never ``self.medium``.
    def effective_medium(self) -> str:
        return "watercolor" if self.doc_mode in WATERCOLOR_DOCS else self.medium

    # Cache keys: each tuple captures *only* the fields a stage depends on,
    # plus the key of the stage above it (chained by the pipeline). Two Params
    # with equal keys for a stage must produce identical output there.
    def norm_key(self) -> tuple:
        return (self.max_edge,)

    def extract_key(self) -> tuple:
        # doc_mode belongs here: plan mode skips extraction entirely, so without
        # it a plan run would poison the cache with an empty result and the next
        # board run would silently show no targets.
        return (self.doc_mode, self.extract_enabled, self.extract_n)

    def passage_key(self) -> tuple:
        # Segmentation is plan-only; in every other document the key collapses
        # to a constant so their settings can never re-segment.
        if self.doc_mode != DOC_PLAN:
            return ("off",)
        return (DOC_PLAN, self.wash_tiers, self.plan_colors, round(self.min_passage_pct, 3))

    def mosaic_key(self) -> tuple:
        # Only what changes the *cells*. Grout, jitter and tessera size are
        # drawing decisions, so they stay out and recompute nothing.
        if self.doc_mode != DOC_MOSAIC:
            return ("off",)
        return (DOC_MOSAIC, self.tessera_layout, self.tiles_x, self.mosaic_colors)

    def palette_mix_sig(self) -> tuple:
        # Only what changes a mix's *color math*: hex, whiteness, opacity class.
        # A paint's name is read at compose time, so renaming recomputes nothing.
        return tuple((p.hex, p.is_white, p.opacity, p.staining) for p in self.palette)

    def solve_key(self) -> tuple:
        return (
            self.effective_medium(),
            self.palette_mix_sig(),
            tuple(t.hex for t in self.manual_targets),
            self.max_paints,
        )

    def ladder_key(self) -> tuple:
        return (self.ladder_steps, self.shade_mode)

    def schedule_key(self) -> tuple:
        # Paper size only scales the wash volumes, so it re-runs the schedule
        # but never the segmentation or the solve above it.
        return (round(self.paper_w, 3), round(self.paper_h, 3), self.canvas_units)


# Built-in presets: a sensible starting kit per medium. Each switches the
# medium *and* loads its starter palette; board/solver style is left alone.
PRESETS: dict[str, dict] = {
    "Oil — split primary 8": dict(medium="oil", palette=STARTERS["oil"], doc_mode=DOC_BOARD),
    "Watercolor — 12": dict(medium="watercolor", palette=STARTERS["watercolor"], doc_mode=DOC_BOARD),
    "Acrylic — basic 6": dict(medium="acrylic", palette=STARTERS["acrylic"], doc_mode=DOC_BOARD),
    "Oil pastel — 24": dict(medium="oilpastel", palette=STARTERS["oilpastel"], max_paints=2,
                            doc_mode=DOC_BOARD),
    "Watercolor painting plan": dict(medium="watercolor", palette=STARTERS["watercolor"],
                                     doc_mode=DOC_PLAN),
    "Watercolor mosaic guide": dict(medium="watercolor", palette=STARTERS["watercolor"],
                                    doc_mode=DOC_MOSAIC),
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
    "wash_tiers", "plan_colors", "min_passage_pct",
    "show_plan_map", "show_plan_passes", "show_plan_tips",
    "tessera_layout", "tiles_x", "mosaic_colors", "grout_frac", "tessera_jitter",
    "show_mosaic_preview", "show_mosaic_grid", "show_mosaic_legend",
)


def preset_to_dict(params: Params) -> dict:
    """The PRESET_FIELDS of ``params`` as a JSON-able dict (a saved user preset)."""
    return {k: getattr(params, k) for k in PRESET_FIELDS}


def apply_preset_values(params: Params, values: dict) -> Params:
    """Apply a saved preset dict onto ``params`` (only the known PRESET_FIELDS)."""
    changes = {k: v for k, v in values.items() if k in PRESET_FIELDS}
    return params.copy(**changes) if changes else params
