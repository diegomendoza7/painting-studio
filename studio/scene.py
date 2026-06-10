"""A backend-agnostic description of the palette board.

The pipeline produces a ``Scene`` (pure geometry + style). Two renderers
consume it: ``studio.svg`` (vector export) and ``studio.ui.render`` (QPainter
raster for the live preview and PNG export). Keeping the scene Qt-free means
the whole pipeline is testable without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]

# Canonical layer names (also used as SVG group ids / Inkscape labels).
LAYER_HEADER = "header"
LAYER_SWATCHES = "palette-swatches"
LAYER_WHEEL = "color-wheel"
LAYER_REFERENCE = "reference"
LAYER_RECIPES = "targets-recipes"
LAYER_LADDERS = "value-ladders"
LAYER_NOTES = "notes"

ALL_LAYERS = (
    LAYER_HEADER,
    LAYER_SWATCHES,
    LAYER_WHEEL,
    LAYER_REFERENCE,
    LAYER_RECIPES,
    LAYER_LADDERS,
    LAYER_NOTES,
)


@dataclass
class Stroke:
    """A single polyline or polygon, optionally with interior holes."""

    points: List[Point]
    closed: bool = False
    width: float = 1.0
    color: str = "#1b1b1b"
    fill: Optional[str] = None       # None = no fill (outline only)
    fill_opacity: float = 1.0
    opacity: float = 1.0
    holes: List[List[Point]] = field(default_factory=list)


@dataclass
class Text:
    """A single text annotation."""

    x: float
    y: float                         # baseline y (SVG/Qt convention)
    text: str
    size: float = 14.0               # font size in user units (px)
    color: str = "#1b1b1b"
    anchor: str = "start"            # SVG text-anchor: start | middle | end
    opacity: float = 1.0


@dataclass
class ImagePatch:
    """A raster image placed at a rect inside the board (the reference photo)."""

    rgb: np.ndarray                  # HxWx3 uint8
    x: float
    y: float
    w: float
    h: float
    opacity: float = 1.0


@dataclass
class Layer:
    name: str
    strokes: List[Stroke] = field(default_factory=list)
    texts: List[Text] = field(default_factory=list)
    visible: bool = True


@dataclass
class Scene:
    width: int
    height: int
    background: str = "#f6f3ea"
    layers: List[Layer] = field(default_factory=list)
    # Raster patches drawn beneath the vector layers (the reference photo).
    patches: List[ImagePatch] = field(default_factory=list)
    # Uniform border (scene units) the renderers reserve around the content.
    margin: int = 0
    # Real-world scale: inches per scene pixel, set when the artist gives a
    # card width so the export carries physical units. 0 = unscaled px output.
    inches_per_px: float = 0.0
    canvas_units: str = "in"

    def layer(self, name: str) -> Optional[Layer]:
        for ly in self.layers:
            if ly.name == name:
                return ly
        return None

    def visible_layers(self) -> List[Layer]:
        return [ly for ly in self.layers if ly.visible and (ly.strokes or ly.texts)]
