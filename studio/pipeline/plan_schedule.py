"""Plan stage B — order the passages into numbered wash passes.

This is where the plan stops being analysis and becomes instructions. The
constraints are watercolor's own, and they are hard ones:

* **Light before dark.** Passes are emitted in value-tier order, lightest
  first, because a transparent medium can only ever go darker.
* **One mix, several shapes.** Passages in the same tier whose colours are
  within ``GROUP_DE`` share a pass — that is one well of paint covering
  several places on the paper, which is how a wash is actually laid.
* **Dry between passes.** Every pass is a glaze over what is already down, so
  each one carries an explicit drying break.
* **Whites are reserved, not painted.** Passages light enough to be paper are
  dropped from the schedule and listed separately.

It also answers "how much?" — a wash volume derived from the passage's share
of the real paper, deliberately generous, because running out of a mix
halfway through a wash is unrecoverable.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .. import tips as tips_mod
from ..color import delta_e_rgb, hex_to_rgb
from ..media import dilution_phrase, dilution_short
from ..models import MixRecipe, Passage, WashPass
from ..params import CM_PER_INCH, UNIT_CM

# Two passages this close in colour are the same mix, so they share a pass.
GROUP_DE = 12.0
# Fluid a single wash lays down, per cm² of paper. Empirical, generous.
ML_PER_CM2 = 0.045
WET_ON_WET_FACTOR = 1.35        # pre-wetting the paper costs extra volume
SAFETY = 1.5                    # always mix more than the arithmetic says
MIN_ML = 3.0                    # below this you cannot usefully mix a wash
BRUSH_LOAD_ML = 0.6             # a loaded size-10 round
TSP_ML = 4.93
# A pass at least this big, in the lightest tiers, is worth pre-wetting.
WET_AREA = 0.18


def paper_area_cm2(paper_w: float, paper_h: float, units: str) -> float:
    """Paper area in cm², from a width/height given in ``units``."""
    if paper_w <= 0 or paper_h <= 0:
        return 0.0
    w, h = float(paper_w), float(paper_h)
    if units != UNIT_CM:
        w, h = w * CM_PER_INCH, h * CM_PER_INCH
    return w * h


def mix_parts(weights: Sequence[float]) -> Tuple[int, ...]:
    """Smallest whole-number parts that reproduce ``weights``.

    Artists measure by eye in parts ("three of this to one of that"), never in
    percentages, so the printed ratio is searched over small denominators and
    scored on how far the rounded ratio drifts from the solved weights.
    """
    if len(weights) <= 1:
        return (1,)
    best: Tuple[float, int, Tuple[int, ...]] | None = None
    for total in range(2, 13):
        parts = tuple(max(1, int(round(w * total))) for w in weights)
        s = sum(parts)
        err = sum(abs(w - p / s) for w, p in zip(weights, parts))
        cand = (round(err, 6), total, parts)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[2] if best else tuple(1 for _ in weights)


def parts_text(recipe: MixRecipe, names_by_hex: Dict[str, str]) -> str:
    """The recipe as a spoken ratio of parts.

    Called at *compose* time, not from the schedule stage: it is the one piece
    of the plan that needs paint names, and resolving them by hex against the
    live palette is what lets a rename cost nothing.
    """
    def nm(step) -> str:
        return names_by_hex.get(step.paint_hex, step.paint_name)

    if len(recipe.steps) == 1:
        return f"{nm(recipe.steps[0])} on its own"
    parts = mix_parts([s.weight for s in recipe.steps])
    return " : ".join(
        f"{p} part{'s' if p != 1 else ''} {nm(s)}" for p, s in zip(parts, recipe.steps)
    )


def wash_volume_ml(area_frac: float, paper_cm2: float, wet_on_wet: bool) -> float:
    """How much wash to mix for a passage covering ``area_frac`` of the paper."""
    if paper_cm2 <= 0:
        return 0.0
    ml = area_frac * paper_cm2 * ML_PER_CM2 * SAFETY
    if wet_on_wet:
        ml *= WET_ON_WET_FACTOR
    ml = max(MIN_ML, ml)
    # Round to something a person can aim at in a mixing well.
    if ml < 10.0:
        return round(ml * 2.0) / 2.0
    if ml < 30.0:
        return float(round(ml))
    return float(int(round(ml / 5.0)) * 5)


def volume_text(ml: float) -> str:
    """Wash volume in units you can judge without a measuring cylinder."""
    if ml <= 0:
        return "set a paper size to get wash volumes"
    loads = max(1, int(round(ml / BRUSH_LOAD_ML)))
    amount = f"{ml:g} ml"
    if ml >= 2 * TSP_ML:
        amount += f" (about {ml / TSP_ML:.0f} tsp)"
    return f"mix ~{amount} — roughly {loads} full brush-loads"


# --- grouping ------------------------------------------------------------------
def _group_tier(members: Sequence[Tuple[Passage, MixRecipe]]) -> List[List[int]]:
    """Cluster one tier's passages into shared mixes, greedily by colour.

    Largest passage first, so each group is anchored on the shape whose colour
    matters most; the recipe that gets printed is that shape's own solved mix,
    never an average of several.
    """
    order = sorted(range(len(members)), key=lambda i: (-members[i][0].area_frac, i))
    groups: List[List[int]] = []
    for i in order:
        placed = False
        rgb_i = hex_to_rgb(members[i][0].target_hex)
        for g in groups:
            if delta_e_rgb(hex_to_rgb(members[g[0]][0].target_hex), rgb_i) < GROUP_DE:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def _edges_text(
    ids: Sequence[int], painted: set, reserved: set, adjacency: Dict[int, Dict[int, int]]
) -> str:
    """Where this pass's edges go soft, stay crisp, or must stop at bare paper."""
    id_set = set(ids)
    neighbours = {b for a in ids for b in adjacency.get(a, {})}
    whites = sorted(neighbours & reserved)
    earlier = sorted((neighbours & painted) - reserved)
    later = sorted(neighbours - painted - id_set - reserved)
    internal = any(b in id_set for a in ids for b in adjacency.get(a, {}) if b != a)
    out: List[str] = []
    if internal:
        out.append("shapes in this pass touch — lay them as one continuous wash")
    if whites:
        out.append(
            f"runs up against bare paper at {', '.join(str(i) for i in whites[:4])} — "
            "keep the brush off it"
        )
    if earlier:
        out.append(
            f"meets {'passage' if len(earlier) == 1 else 'passages'} "
            f"{', '.join(str(i) for i in earlier[:4])} (already dry) — that edge will be crisp"
        )
    if later:
        out.append(
            f"stop cleanly against {', '.join(str(i) for i in later[:4])}; those come later"
        )
    return "; ".join(out[:2])


def _why(index: int, n_passes: int, tier: int, area_frac: float) -> str:
    if index == 1:
        return (
            "The underwash. It sets the lightest note in the painting and everything "
            "else is glazed on top of it."
        )
    if index == n_passes and n_passes > 1:
        return (
            "The darks. They are what make the painting read, and they only work once "
            "everything beneath them is bone dry."
        )
    if area_frac >= 0.20:
        return "A big mid-value mass — this pass does most of the drawing."
    return "Mid-values. These separate the shapes from each other."


def _tags(index: int, n_passes: int, wet_on_wet: bool, area_frac: float, recipe: MixRecipe,
          staining: bool, granulating: bool) -> List[str]:
    tags: List[str] = []
    if index == 1:
        tags.append("first-wash")
    if wet_on_wet:
        tags.append("wet-on-wet")
    else:
        tags.append("wet-on-dry")
    if index == n_passes and n_passes > 1:
        tags.append("darks")
    if index > 1:
        tags.append("glaze")
    if staining:
        tags.append("staining")
    if granulating:
        tags.append("granulating")
    tags.append("large-wash" if area_frac >= 0.20 else "small-wash")
    tags.extend(("drying", "value", "hard-edge", "soft-edge", "lifting"))
    return tags


# --- public entry point ------------------------------------------------------------
def build_schedule(
    passages: Sequence[Passage],
    recipes: Sequence[MixRecipe],
    adjacency: Dict[int, Dict[int, int]],
    palette: Sequence,
    paper_w: float,
    paper_h: float,
    units: str,
) -> Tuple[WashPass, ...]:
    """Turn passages + their solved mixes into an ordered set of wash passes.

    ``recipes`` is index-aligned with ``passages`` (the plan solves the passage
    colours first, so passage *i* is recipe *i*).
    """
    paper_cm2 = paper_area_cm2(paper_w, paper_h, units)
    traits = {p.hex: p for p in palette}
    paintable = [
        (p, recipes[i]) for i, p in enumerate(passages)
        if i < len(recipes) and not p.reserved
    ]
    reserved_ids = {p.id for p in passages if p.reserved}
    if not paintable:
        return ()

    # Bucket by value tier, then split each tier into shared mixes.
    by_tier: Dict[int, List[Tuple[Passage, MixRecipe]]] = {}
    for pair in paintable:
        by_tier.setdefault(pair[0].tier, []).append(pair)

    planned: List[Tuple[int, float, float, List[Tuple[Passage, MixRecipe]]]] = []
    for tier in sorted(by_tier):
        members = by_tier[tier]
        for g in _group_tier(members):
            group = [members[i] for i in g]
            area = sum(p.area_frac for p, _r in group)
            mean_l = sum(p.lstar * p.area_frac for p, _r in group) / max(area, 1e-9)
            planned.append((tier, area, mean_l, group))
    # Strictly lightest-first. A tier is only a grouping device — it can span a
    # wide band of values, so ordering on the group's own mean L* is what
    # actually guarantees no wash ever goes down over a darker one.
    planned.sort(key=lambda t: (-t[2], -t[1], min(p.id for p, _r in t[3])))

    out: List[WashPass] = []
    painted: set = set()
    used_tips: List[str] = []
    n_passes = len(planned)
    for n, (tier, area, _mean_l, group) in enumerate(planned, start=1):
        anchor = max(group, key=lambda pr: (pr[0].area_frac, -pr[0].id))
        recipe = anchor[1]
        ids = tuple(sorted(p.id for p, _r in group))
        wet_on_wet = area >= WET_AREA and tier <= 1
        ml = wash_volume_ml(area, paper_cm2, wet_on_wet)
        staining = any(getattr(traits.get(s.paint_hex), "staining", False) for s in recipe.steps)
        granulating = any(
            getattr(traits.get(s.paint_hex), "granulating", False) for s in recipe.steps
        )
        technique = (
            "Wet the shape with clean water first, let the shine dull, then drop the colour in."
            if wet_on_wet
            else "Paint straight onto dry paper, one loaded stroke at a time."
        )
        out.append(
            WashPass(
                index=n,
                tier=tier,
                passage_ids=ids,
                recipe=recipe,
                area_frac=round(area, 4),
                volume_ml=ml,
                volume_text=volume_text(ml),
                strength_text=dilution_phrase(recipe.dilution),
                wet="wet-on-wet" if wet_on_wet else "wet-on-dry",
                technique=technique,
                edges=_edges_text(ids, painted, reserved_ids, adjacency),
                why=_why(n, n_passes, tier, area),
                tip=tips_mod.pick(
                    _tags(n, n_passes, wet_on_wet, area, recipe, staining, granulating),
                    used_tips,
                ),
            )
        )
        painted.update(ids)
    return tuple(out)


def reserved_passages(passages: Sequence[Passage]) -> Tuple[Passage, ...]:
    """The shapes that stay bare paper, lightest first."""
    return tuple(sorted((p for p in passages if p.reserved), key=lambda p: (-p.lstar, p.id)))


def dilution_label(recipe: MixRecipe) -> str:
    return dilution_short(recipe.dilution)
