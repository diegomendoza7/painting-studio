"""Stage 2 — solve mixing recipes: target color -> paints + proportions.

For brush media (oil, acrylic, watercolor) the solver searches combinations of
1-4 palette paints, mixed in Mixbox's pigment latent space, for the weights
that minimize CIEDE2000 to the target. Watercolor never mixes with white —
each candidate is instead *diluted* (Beer-Lambert over paper white) to match
the target's lightness before scoring. Oil pastel doesn't mix on a palette at
all, so its "recipes" are ordered two-stick layerings.

Everything is deterministic: fixed grids, no RNG, explicit tie-breaking —
the same palette and target always print the same recipe.
"""
from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple

from .. import mixer
from ..color import RGB, delta_e_2000, hex_to_rgb, lab, lab_from_linear, linear3, rgb_to_hex
from ..media import recipe_note, spec
from ..models import MixRecipe, MixStep, Paint, Target

# A printed recipe is "achievable" when its CIEDE2000 to the target is within
# this. ~2 is a just-noticeable difference; 6 is a workable easel match.
ACHIEVABLE_DE = 6.0

_PAIR_GRID = tuple(w / 10.0 for w in range(1, 10))            # 0.1 .. 0.9
# The full simplex at step 0.1, every weight >= 0.1 — symmetric, so no slot is
# capped. The earlier grid capped whichever paint came first in the palette at
# 60%, and since the palette lists Titanium White first, a three-paint oil tint
# could never be mostly white: a light grey came back as yellow + blue + red with
# no white at all.
_TRIPLE_GRID = tuple(
    (a / 10.0, b / 10.0, (10 - a - b) / 10.0)
    for a in range(1, 9)
    for b in range(1, 10 - a)
)
# Four-paint mixes: every weight >= 0.1 in 0.1 steps. Neutrals are where a
# fourth paint earns its place — a muted grey wants white plus a warm and a cool
# plus a modifier, and on the oil starter palette three paints can't get closer
# than ΔE ~8 to an ordinary mid grey while four reach ~2.
_QUAD_GRID = tuple(
    (a / 10.0, b / 10.0, c / 10.0, (10 - a - b - c) / 10.0)
    for a in range(1, 8)
    for b in range(1, 9 - a)
    for c in range(1, 10 - a - b)
)
# Four-paint combinations grow fast (a 15-tube palette has 1,365), so a fourth
# paint is only tried on top of the best three-paint mixes. That keeps a click
# at the easel well under a tenth of a second on any real palette.
_QUAD_SEEDS = 20
_TOP_K = 10
_MIN_W = 0.05
_PASTEL_DOMINANCE = (0.5, 0.65, 0.8)

ProgressCB = Optional[Callable[[str, float], None]]


# --- watercolor dilution fit -------------------------------------------------
def _fit_dilution(mass_lin: Tuple[float, float, float], target_l: float) -> float:
    """Optical thickness d so the wash's L* matches the target's (bisection).

    L* is monotone decreasing in d (more paint = darker), so plain bisection
    on [0.02, 1] converges in 8 steps. Returns 1.0 when the masstone is
    already darker than needed (then pigment load, not water, sets the value).
    """
    if lab_from_linear(mass_lin)[0] >= target_l:
        return 1.0
    lo, hi = 0.02, 1.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        l_mid = lab_from_linear((mass_lin[0] ** mid, mass_lin[1] ** mid, mass_lin[2] ** mid))[0]
        if l_mid > target_l:
            lo = mid  # too light -> thicker paint
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _diluted(rgb: RGB, d: float) -> RGB:
    from ..color import dilute

    return rgb if d >= 0.999 else dilute(rgb, d)


# --- scoring -----------------------------------------------------------------
def _score(
    latents: Sequence[tuple],
    weights: Sequence[float],
    target_lab: tuple,
    dilutes: bool,
) -> Tuple[float, RGB, float]:
    """(delta_e, produced_rgb, dilution) for one candidate mix."""
    rgb = mixer.mix(latents, weights)
    d = 1.0
    if dilutes:
        d = _fit_dilution(linear3(rgb), target_lab[0])
        rgb = _diluted(rgb, d)
    return delta_e_2000(target_lab, lab(rgb)), rgb, d


def _renormalized(weights: Sequence[float]) -> Tuple[float, ...]:
    s = sum(weights)
    return tuple(w / s for w in weights)


def _round_weights(weights: Sequence[float]) -> Tuple[float, ...]:
    """Round to 5% steps, dumping the drift on the largest weight."""
    r = [round(w * 20.0) / 20.0 for w in weights]
    drift = 1.0 - sum(r)
    r[max(range(len(r)), key=lambda i: r[i])] += drift
    return tuple(round(w, 4) for w in r)


# --- brush-media search -------------------------------------------------------
def _solve_brush(
    target_rgb: RGB, paints: Sequence[Tuple[int, Paint, tuple]], max_paints: int, dilutes: bool
) -> Tuple[Tuple[int, ...], Tuple[float, ...], float, RGB, float]:
    """Search 1–4 paint mixes; returns (paint indices, weights, dE, rgb, dilution)."""
    target_lab = lab(target_rgb)
    # Candidate = (dE, n_paints, indices, weights, rgb, dilution); ordering on
    # the first four keeps every comparison deterministic.
    cands: List[tuple] = []

    def consider(idxs: Tuple[int, ...], weights: Tuple[float, ...]) -> None:
        de, rgb, d = _score([paints[i][2] for i in idxs], weights, target_lab, dilutes)
        cands.append((round(de, 6), len(idxs), idxs, weights, rgb, d))

    for i in range(len(paints)):
        consider((i,), (1.0,))
    if max_paints >= 2:
        for i, j in itertools.combinations(range(len(paints)), 2):
            for w in _PAIR_GRID:
                consider((i, j), (w, 1.0 - w))
    if max_paints >= 3:
        for i, j, k in itertools.combinations(range(len(paints)), 3):
            for w in _TRIPLE_GRID:
                consider((i, j, k), w)
    if max_paints >= 4 and len(paints) >= 4:
        seeds = sorted((c for c in cands if c[1] == 3), key=lambda c: c[:4])[:_QUAD_SEEDS]
        tried = set()
        for seed in seeds:
            for extra in range(len(paints)):
                idxs = tuple(sorted(seed[2] + (extra,)))
                if extra in seed[2] or idxs in tried:
                    continue
                tried.add(idxs)
                for w in _QUAD_GRID:
                    consider(idxs, w)

    cands.sort(key=lambda c: c[:4])
    best = cands[0]
    for cand in cands[:_TOP_K]:
        refined = _refine(cand, paints, target_lab, dilutes)
        if refined[:4] < best[:4]:
            best = refined
    _, _, idxs, weights, rgb, d = best
    return idxs, weights, best[0], rgb, d


def _refine(cand: tuple, paints, target_lab: tuple, dilutes: bool) -> tuple:
    """Deterministic local grid refinement around one coarse candidate."""
    de, n, idxs, weights, rgb, d = cand
    latents = [paints[i][2] for i in idxs]
    best = cand
    if n == 2:
        w0 = best[3][0]
        for span, step in ((0.08, 0.02), (0.02, 0.005)):
            w0 = best[3][0]
            for k in range(-int(span / step), int(span / step) + 1):
                w = min(1.0 - _MIN_W, max(_MIN_W, w0 + k * step))
                de2, rgb2, d2 = _score(latents, (w, 1.0 - w), target_lab, dilutes)
                cand2 = (round(de2, 6), n, idxs, (w, 1.0 - w), rgb2, d2)
                if cand2[:4] < best[:4]:
                    best = cand2
    elif n >= 3:
        # Coordinate refinement over the first n-1 weights; the last takes the
        # remainder, so the weights always sum to one.
        for step in (0.025, 0.00625):
            for coord in range(n - 1):
                w = list(best[3])
                for k in range(-3, 4):
                    trial = list(w)
                    trial[coord] = max(_MIN_W, w[coord] + k * step)
                    trial[-1] = 1.0 - sum(trial[:-1])
                    if trial[-1] < _MIN_W:
                        continue
                    tw = tuple(trial)
                    de2, rgb2, d2 = _score(latents, tw, target_lab, dilutes)
                    cand2 = (round(de2, 6), n, idxs, tw, rgb2, d2)
                    if cand2[:4] < best[:4]:
                        best = cand2
    return best


# --- oil pastel layering search ------------------------------------------------
def _solve_pastel(
    target_rgb: RGB, paints: Sequence[Tuple[int, Paint, tuple]]
) -> Tuple[Tuple[int, ...], Tuple[float, ...], float, RGB]:
    """Singles + ordered (base, top) layerings scored with mixbox.lerp."""
    target_lab = lab(target_rgb)
    best = None
    for i, (_, paint, _z) in enumerate(paints):
        rgb = hex_to_rgb(paint.hex)
        de = delta_e_2000(target_lab, lab(rgb))
        cand = (round(de, 6), 1, (i,), (1.0,), rgb)
        if best is None or cand[:4] < best[:4]:
            best = cand
    rgbs = [hex_to_rgb(p.hex) for _, p, _z in paints]
    for i in range(len(paints)):
        for j in range(len(paints)):
            if i == j:
                continue
            for t in _PASTEL_DOMINANCE:
                rgb = mixer.lerp(rgbs[i], rgbs[j], t)
                de = delta_e_2000(target_lab, lab(rgb))
                cand = (round(de, 6), 2, (i, j), (1.0 - t, t), rgb)
                if cand[:4] < best[:4]:
                    best = cand
    # Refine the dominance of the best pair on a 0.05 grid.
    if best[1] == 2:
        i, j = best[2]
        for t in (k / 20.0 for k in range(5, 18)):
            rgb = mixer.lerp(rgbs[i], rgbs[j], t)
            de = delta_e_2000(target_lab, lab(rgb))
            cand = (round(de, 6), 2, (i, j), (1.0 - t, t), rgb)
            if cand[:4] < best[:4]:
                best = cand
    de, _n, idxs, weights, rgb = best
    return idxs, weights, de, rgb


# --- public entry point ----------------------------------------------------------
def solve_targets(
    targets: Sequence[Target],
    palette: Sequence[Paint],
    medium: str,
    max_paints: int,
    progress: ProgressCB = None,
) -> Tuple[MixRecipe, ...]:
    """One :class:`MixRecipe` per target, in target order."""
    md = spec(medium)
    pool = [p for p in palette if not (md.dilutes and p.is_white)]  # watercolor: paper is the white
    if not pool or not targets:
        return ()
    indexed = [(i, p, None if md.layered else mixer.latent_hex(p.hex)) for i, p in enumerate(pool)]

    recipes: List[MixRecipe] = []
    for ti, target in enumerate(targets):
        if progress:
            progress(f"Solving mixes {ti + 1}/{len(targets)}", 0.25 + 0.55 * ti / len(targets))
        target_rgb = hex_to_rgb(target.hex)

        if md.layered:
            idxs, weights, de, rgb = _solve_pastel(target_rgb, indexed)
            d = 1.0
        else:
            idxs, weights, de, rgb, d = _solve_brush(target_rgb, indexed, max_paints, md.dilutes)
            # Print weights the artist can actually measure, and re-score the
            # rounded recipe so the printed delta-E is honest.
            weights = _renormalized(_round_weights(weights))
            target_lab = lab(target_rgb)
            de, rgb, d = _score([indexed[i][2] for i in idxs], weights, target_lab, md.dilutes)

        steps = [MixStep(indexed[i][1].name, indexed[i][1].hex, round(w, 4)) for i, w in zip(idxs, weights)]
        if not md.layered:
            # Largest share first; white last — it reads as "tint to finish".
            steps.sort(key=lambda s: (indexed_by_hex(indexed, s.paint_hex).is_white, -s.weight, s.paint_name))
        note = recipe_note(medium, steps, palette, d, md.layered)
        recipes.append(
            MixRecipe(
                target_hex=target.hex,
                target_label=target.label,
                steps=tuple(steps),
                mixed_hex=rgb_to_hex(rgb),
                delta_e=round(de, 2),
                dilution=round(d, 3),
                layering=md.layered,
                note=note,
                achievable=de <= ACHIEVABLE_DE,
            )
        )
    return tuple(recipes)


def indexed_by_hex(indexed, hex_: str) -> Paint:
    for _, p, _z in indexed:
        if p.hex == hex_:
            return p
    return Paint("", hex_)
