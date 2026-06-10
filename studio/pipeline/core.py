"""Pipeline orchestration with per-stage caching.

Each stage caches its output keyed by *only* the parameters it depends on,
chained to the key of the stage above it. An eyedropper pick re-solves mixes
and re-ladders but never re-extracts the photo; toggling a board section is
compose-only and recomputes nothing; renaming a paint recomputes nothing
(names are resolved at compose time).

Unlike bargue, the pipeline runs happily with *no source image* — the
artist's palette plus hand-picked targets already make a valid board. The
photo stages simply no-op until a reference is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..models import MixRecipe, Target, ValueLadder
from ..params import Params
from ..scene import Scene
from . import board, extract as extract_mod, ladder as ladder_mod, solver
from .board import BoardLayout
from .loader import LoadedImage, normalize

ProgressCB = Optional[Callable[[str, float], None]]


@dataclass
class Result:
    scene: Scene
    layout: BoardLayout
    working_rgb: Optional[np.ndarray]
    auto_targets: Tuple[Target, ...]
    recipes: Tuple[MixRecipe, ...]
    ladders: Tuple[ValueLadder, ...]
    recomputed: List[str] = field(default_factory=list)


class Pipeline:
    """Holds the source photo (if any) and all cached intermediate results."""

    def __init__(self) -> None:
        self._raw: Optional[LoadedImage] = None
        self._raw_token: Optional[tuple] = None
        self._norm = None      # (key, LoadedImage)
        self._extract = None   # (key, Tuple[Target, ...])
        self._solve = None     # (key, Tuple[MixRecipe, ...])
        self._ladder = None    # (key, Tuple[ValueLadder, ...])

    # -- source management --------------------------------------------------
    def set_source(self, loaded: LoadedImage) -> None:
        self._raw = loaded
        self._raw_token = (loaded.source_path, loaded.page_index, id(loaded))

    def clear_source(self) -> None:
        self._raw = None
        self._raw_token = None

    @property
    def has_source(self) -> bool:
        return self._raw is not None

    # -- cache helper -------------------------------------------------------
    @staticmethod
    def _cached(slot, key, compute):
        if slot is not None and slot[0] == key:
            return slot[1], slot, False
        value = compute()
        return value, (key, value), True

    # -- run ----------------------------------------------------------------
    def run(self, params: Params, progress: ProgressCB = None) -> Result:
        recomputed: List[str] = []

        def report(label: str, frac: float) -> None:
            if progress:
                progress(label, frac)

        # Stage 0 — normalize (photo only; skipped entirely with no source).
        rgb: Optional[np.ndarray] = None
        norm_key: Optional[tuple] = None
        if self._raw is not None:
            norm_key = (self._raw_token, params.norm_key())

            def comp_norm():
                report("Normalizing photo", 0.05)
                return normalize(self._raw, params.max_edge)

            loaded, self._norm, r = self._cached(self._norm, norm_key, comp_norm)
            if r:
                recomputed.append("normalize")
            rgb = loaded.rgb

        # Stage 1 — extract dominant colors from the photo.
        extract_key = (norm_key, params.extract_key())

        def comp_extract() -> Tuple[Target, ...]:
            if rgb is None or not params.extract_enabled:
                return ()
            report("Extracting photo palette", 0.15)
            return extract_mod.extract_targets(rgb, params.extract_n)

        auto_targets, self._extract, r = self._cached(self._extract, extract_key, comp_extract)
        if r:
            recomputed.append("extract")

        # Stage 2 — solve mixing recipes for all targets (auto then manual).
        solve_key = (extract_key, params.solve_key())

        def comp_solve() -> Tuple[MixRecipe, ...]:
            manual = tuple(
                t if t.label else Target(hex=t.hex, label=f"Color {i + 1}", source=t.source)
                for i, t in enumerate(params.manual_targets)
            )
            return solver.solve_targets(
                auto_targets + manual, params.palette, params.medium,
                params.max_paints, progress=report,
            )

        recipes, self._solve, r = self._cached(self._solve, solve_key, comp_solve)
        if r:
            recomputed.append("solve")

        # Stage 3 — value ladders.
        ladder_key = (solve_key, params.ladder_key())

        def comp_ladders() -> Tuple[ValueLadder, ...]:
            report("Building value ladders", 0.85)
            return ladder_mod.build_ladders(
                recipes, params.palette, params.medium,
                params.ladder_steps, params.shade_mode,
            )

        ladders, self._ladder, r = self._cached(self._ladder, ladder_key, comp_ladders)
        if r:
            recomputed.append("ladders")

        # Stage 4 — compose (cheap; always rebuilt from cached results + style).
        report("Composing board", 0.95)
        scene, layout = board.build_board(params, recipes, ladders, rgb, auto_targets)
        report("Done", 1.0)

        return Result(
            scene=scene,
            layout=layout,
            working_rgb=rgb,
            auto_targets=auto_targets,
            recipes=recipes,
            ladders=ladders,
            recomputed=recomputed,
        )
