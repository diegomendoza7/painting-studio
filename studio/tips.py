"""Watercolor teaching content — the "learn as you go" half of the plan.

Pure data plus deterministic selectors. The painting plan prints a few
headline rules once, then attaches *one* contextual tip to each wash pass,
chosen from the situation that pass is actually in (first wash, a reserved
white, a staining pigment, a granulating one, a big wet-on-wet lay-in, the
final darks…). Selection is deterministic and de-duplicating, so a plan reads
like a lesson rather than the same sentence six times.

Kept separate from :mod:`studio.media` on purpose: ``media`` describes how a
medium *behaves* (and is consumed by the solver), this describes how to
*learn* it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Printed once, at the top of every plan. The things that bite beginners.
GOLDEN_RULES: Tuple[str, ...] = (
    "Work light to dark. You can always add another wash; you can never take one back.",
    "The paper is your white — anything meant to stay white must be left unpainted.",
    "Let every wash dry completely before glazing the next, or the two become one.",
    "Mix more wash than you think you need. A wash re-mixed mid-passage never matches.",
    "Everything dries 20–30% lighter than it looks wet, so mix stronger than the target.",
)

# Contextual tips, keyed by the situation the schedule detects.
TIPS: Dict[str, Tuple[str, ...]] = {
    "first-wash": (
        "This first wash sets the key for everything above it. Keep it paler than feels "
        "right — you will be glazing over it several times.",
        "Tilt the board slightly toward you and keep a wet bead running along the bottom "
        "edge of the wash; that bead is what stops streaks.",
    ),
    "wet-on-wet": (
        "Wet the area with clean water first, wait until the shine just dulls, then drop "
        "the color in. Too wet and it blooms, too dry and it streaks.",
        "Wet-on-wet does most of the work by itself. Place the color and leave it alone — "
        "brushing it around is what kills the softness.",
    ),
    "wet-on-dry": (
        "Paint this onto bone-dry paper for crisp edges. Touch it with the back of your "
        "hand: if it feels cool, it is still damp.",
        "On dry paper the shape you paint is the shape you get, so load the brush fully "
        "and commit to the edge in one stroke.",
    ),
    "glaze": (
        "A glaze is a transparent wash over a dry one. Move quickly and do not scrub — "
        "rewetting the layer beneath will lift it and muddy both.",
        "Glazes multiply: a pale wash over another pale wash reads darker than either. "
        "Two thin glazes almost always beat one heavy one.",
    ),
    "darks": (
        "Darks are what make a watercolor read. Mix them thicker than feels safe — a timid "
        "dark is the most common beginner mistake.",
        "Build darks from two or three transparent pigments rather than black; mixed darks "
        "keep their color and stay alive.",
    ),
    "reserve": (
        "Plan these whites before your first brushstroke — mask them, or simply paint "
        "around them. Recovering a white later never quite works.",
        "The brightest white in the painting should be untouched paper. Keeping one is "
        "what makes everything else look luminous.",
    ),
    "staining": (
        "A staining pigment sinks into the fibres and will not lift cleanly. Commit to "
        "this shape, or swap in a non-staining color.",
    ),
    "granulating": (
        "Granulating pigment settles into the paper's tooth and gives texture for free. "
        "Lay this passage flat and let it dry undisturbed to keep the speckle.",
    ),
    "soft-edge": (
        "Touch the two colors together while both are still wet and let them mingle on "
        "their own. Soft edges are made by timing, not by blending.",
    ),
    "hard-edge": (
        "Let the neighbouring shape dry before you paint this one. Hard and soft edges in "
        "the same painting are what make it look deliberate.",
    ),
    "large-wash": (
        "Mix the whole volume in a well before you start. Running out halfway through a "
        "big wash is unrecoverable.",
        "Use the biggest brush you can stand for this one. Big shapes painted with a small "
        "brush go patchy.",
    ),
    "small-wash": (
        "Small shapes dry fast — load the brush once and place the color rather than "
        "working it back and forth.",
    ),
    "drying": (
        "Do not chase a half-dry wash with a brush. Backruns (those cauliflower blooms) "
        "happen when wet meets damp. Walk away and let it go.",
    ),
    "value": (
        "Squint at your reference. Squinting throws away detail and shows you the value "
        "masses you are actually painting.",
    ),
}

# Printed on the mosaic guide. A mosaic is an unusually forgiving first
# watercolor — no edges to control, no wet-in-wet timing — so the advice is
# about order and patience rather than brush handling.
MOSAIC_NOTES: Tuple[str, ...] = (
    "Rule the grid onto your paper first, lightly, in pencil — or print the "
    "to-scale sheet and trace it on a window.",
    "The gaps between squares are the grout: leave them as bare paper. Never "
    "let two wet squares touch, or they will bleed into one.",
    "Work one colour at a time, in number order. Number 1 is the palest, so "
    "you are always painting light before dark.",
    "Paint every square of a colour before you rinse the well — remixing a "
    "colour halfway through will not match, and here the mismatch is visible.",
    "Skip a square if its neighbour is still wet and come back to it. Working "
    "in a checkerboard is faster than waiting.",
    "Each square is one loaded brushstroke. Let it dry flat rather than "
    "brushing it out — the small pooling at the edges is what makes it look "
    "like stone.",
)

# Printed at the foot of the plan so the vocabulary above is never a mystery.
GLOSSARY: Tuple[Tuple[str, str], ...] = (
    ("Wash", "A layer of diluted paint laid over an area, flat or graded."),
    ("Glaze", "A transparent wash laid over an earlier one that has dried completely."),
    ("Wet-on-wet", "Colour dropped into paper that is already damp — soft, bleeding edges."),
    ("Wet-on-dry", "Colour laid on dry paper — crisp, hard edges."),
    ("Reserve", "To leave paper unpainted (or masked) so it stays white."),
    ("Lift", "To pick colour back off with a damp brush or tissue while it is still wet."),
    ("Staining", "Pigment that sinks into the fibres and will not lift back off."),
    ("Granulating", "Pigment that settles into the paper's texture, giving a speckled grain."),
    ("Backrun", "The cauliflower-shaped bloom you get when wet paint runs into damp paint."),
    ("Value", "How light or dark a colour is, ignoring its hue. The bones of the painting."),
    ("Masstone", "The colour of the paint straight from the pan, barely diluted."),
)


def pick(tags: Sequence[str], used: List[str]) -> str:
    """One tip for the first tag that still has an unused line.

    ``used`` accumulates across a schedule so passes don't repeat each other;
    when every line for every tag is spent the first tag's first line is
    reused rather than returning nothing.
    """
    for tag in tags:
        for line in TIPS.get(tag, ()):
            if line not in used:
                used.append(line)
                return line
    for tag in tags:
        lines = TIPS.get(tag, ())
        if lines:
            return lines[0]
    return ""
