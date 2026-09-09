"""Pipeline orchestration with per-stage caching.

Each stage caches its output keyed by *only* the parameters it depends on,
chained to the key of the stage above it. An eyedropper pick re-solves mixes
and re-ladders but never re-extracts the photo; toggling a board section is
compose-only and recomputes nothing; renaming a paint recomputes nothing
(names are resolved at compose time); resizing the paper re-costs the washes
but never re-segments the photo.

The app has three documents and they share this one pipeline. Only the target
list differs: in *board* mode the targets are the photo's dominant colours plus
whatever the artist picked, in *plan* mode they are the segmented passages, and
in *mosaic* mode they are the tesserae colour stock. The same solver turns each
into a recipe from the artist's own paints.

Unlike bargue, the pipeline runs happily with *no source image* — the artist's
palette plus hand-picked targets already make a valid board. The photo stages
simply no-op until a reference is loaded (a painting plan, which is about a
specific picture, says so instead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..models import MixRecipe, Target, ValueLadder, WashPass
from ..params import DOC_MOSAIC, DOC_PLAN, Params
from ..scene import Scene
from . import (
    board,
    extract as extract_mod,
    ladder as ladder_mod,
    mosaic_doc,
    mosaic_grid as mosaic_mod,
    plan_doc,
    plan_regions,
    plan_schedule,
    solver,
)
from .board import BoardLayout
from .loader import LoadedImage, normalize
from .mosaic_grid import MosaicGrid
from .plan_regions import PassageMap

ProgressCB = Optional[Callable[[str, float], None]]


@dataclass
class Result:
    scene: Scene
    layout: BoardLayout
    working_rgb: Optional[np.ndarray]
    auto_targets: Tuple[Target, ...]
    recipes: Tuple[MixRecipe, ...]
    ladders: Tuple[ValueLadder, ...]
    passage_map: Optional[PassageMap] = None
    passes: Tuple[WashPass, ...] = ()
    mosaic: Optional[MosaicGrid] = None
    recomputed: List[str] = field(default_factory=list)


class Pipeline:
    """Holds the source photo (if any) and all cached intermediate results."""

    def __init__(self) -> None:
        self._raw: Optional[LoadedImage] = None
        self._raw_token: Optional[tuple] = None
        self._norm = None      # (key, LoadedImage)
        self._extract = None   # (key, Tuple[Target, ...])
        self._passages = None  # (key, Optional[PassageMap])
        self._mosaic = None    # (key, Optional[MosaicGrid])
        self._solve = None     # (key, Tuple[MixRecipe, ...])
        self._ladder = None    # (key, Tuple[ValueLadder, ...])
        self._schedule = None  # (key, Tuple[WashPass, ...])

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
        is_plan = params.doc_mode == DOC_PLAN
        is_mosaic = params.doc_mode == DOC_MOSAIC

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

        # Stage 1 — extract dominant colors from the photo (board targets).
        extract_key = (norm_key, params.extract_key())

        def comp_extract() -> Tuple[Target, ...]:
            if rgb is None or not params.extract_enabled or is_plan or is_mosaic:
                return ()
            report("Extracting photo palette", 0.15)
            return extract_mod.extract_targets(rgb, params.extract_n)

        # A stage that no-ops for this document type reports nothing — the
        # status line should name work that actually happened, so board mode
        # never mentions "passages" and plan mode never mentions "ladders".
        auto_targets, self._extract, r = self._cached(self._extract, extract_key, comp_extract)
        if r and not (is_plan or is_mosaic):
            recomputed.append("extract")

        # Stage 1b — segment the photo into value tiers and paintable passages.
        passage_key = (norm_key, params.passage_key())

        def comp_passages() -> Optional[PassageMap]:
            if rgb is None or not is_plan:
                return None
            report("Finding passages", 0.20)
            return plan_regions.build_passages(
                rgb, params.wash_tiers, params.plan_colors, params.min_passage_pct
            )

        pmap, self._passages, r = self._cached(self._passages, passage_key, comp_passages)
        if r and is_plan:
            recomputed.append("passages")

        # Stage 1c — cut the photo into tesserae and reduce them to a colour stock.
        mosaic_key = (norm_key, params.mosaic_key())

        def comp_mosaic() -> Optional[MosaicGrid]:
            if rgb is None or not is_mosaic:
                return None
            report("Laying tesserae", 0.20)
            return mosaic_mod.build_mosaic_grid(
                rgb, params.tiles_x, params.mosaic_colors, params.tessera_layout
            )

        grid, self._mosaic, r = self._cached(self._mosaic, mosaic_key, comp_mosaic)
        if r and is_mosaic:
            recomputed.append("mosaic")

        # Stage 2 — solve mixing recipes for all targets.
        #
        # Plan mode solves the passage colours (index-aligned, so passage i is
        # recipe i) and appends anything hand-picked; board mode solves the
        # photo's auto palette plus the picks, as before.
        solve_key = (extract_key, passage_key, mosaic_key, params.solve_key())

        def comp_solve() -> Tuple[MixRecipe, ...]:
            manual = tuple(
                t if t.label else Target(hex=t.hex, label=f"Color {i + 1}", source=t.source)
                for i, t in enumerate(params.manual_targets)
            )
            if is_plan:
                head = tuple(
                    Target(hex=p.target_hex, label=f"Passage {p.id}", source="auto")
                    for p in (pmap.passages if pmap else ())
                )
            elif is_mosaic:
                head = tuple(
                    Target(hex=c, label=f"Colour {i + 1}", source="auto")
                    for i, c in enumerate(grid.colors if grid else ())
                )
            else:
                head = auto_targets
            return solver.solve_targets(
                head + manual, params.palette, params.effective_medium(),
                params.max_paints, progress=report,
            )

        recipes, self._solve, r = self._cached(self._solve, solve_key, comp_solve)
        if r:
            recomputed.append("solve")

        # Stage 3 — value ladders (a board section; the plan has passes instead).
        ladder_key = (solve_key, params.ladder_key())

        def comp_ladders() -> Tuple[ValueLadder, ...]:
            if is_plan or is_mosaic:
                return ()
            report("Building value ladders", 0.85)
            return ladder_mod.build_ladders(
                recipes, params.palette, params.effective_medium(),
                params.ladder_steps, params.shade_mode,
            )

        ladders, self._ladder, r = self._cached(self._ladder, ladder_key, comp_ladders)
        if r and not (is_plan or is_mosaic):
            recomputed.append("ladders")

        # Stage 4 — order the passages into wash passes.
        schedule_key = (solve_key, params.schedule_key())

        def comp_schedule() -> Tuple[WashPass, ...]:
            if not is_plan or pmap is None:
                return ()
            report("Ordering the washes", 0.9)
            return plan_schedule.build_schedule(
                pmap.passages, recipes, pmap.adjacency, params.palette,
                params.paper_w, params.paper_h, params.canvas_units,
            )

        passes, self._schedule, r = self._cached(self._schedule, schedule_key, comp_schedule)
        if r and is_plan:
            recomputed.append("schedule")

        # Stage 5 — compose (cheap; always rebuilt from cached results + style).
        report("Composing", 0.95)
        if is_plan:
            scene, layout = plan_doc.build_plan(params, pmap, recipes, passes, rgb)
        elif is_mosaic:
            scene, layout = mosaic_doc.build_mosaic(params, grid, recipes, rgb)
        else:
            scene, layout = board.build_board(params, recipes, ladders, rgb, auto_targets)
        report("Done", 1.0)

        return Result(
            scene=scene,
            layout=layout,
            working_rgb=rgb,
            auto_targets=auto_targets,
            recipes=recipes,
            ladders=ladders,
            passage_map=pmap,
            passes=passes,
            mosaic=grid,
            recomputed=recomputed,
        )
