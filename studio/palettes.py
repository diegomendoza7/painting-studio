"""Built-in starter palettes, one per medium. Pure data.

Hexes are reasonable sRGB approximations of the real tube/stick masstones —
close enough to plan mixes from, and every paint is editable in the UI, so an
artist can match their own tubes exactly.
"""
from __future__ import annotations

from typing import Tuple

from .models import OPACITY_OPAQUE, OPACITY_SEMI, OPACITY_TRANSPARENT, Paint

# Classic split-primary oil palette: a warm + cool of each primary, two earths, white.
OIL_STARTER: Tuple[Paint, ...] = (
    Paint("Titanium White", "#f6f4ef", is_white=True, opacity=OPACITY_OPAQUE),
    Paint("Cadmium Yellow Light", "#f6e14b", opacity=OPACITY_OPAQUE),
    Paint("Yellow Ochre", "#c08a3e", opacity=OPACITY_SEMI),
    Paint("Cadmium Red Light", "#e2422d", opacity=OPACITY_OPAQUE),
    Paint("Alizarin Crimson", "#962139", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("Ultramarine Blue", "#2c3e92", opacity=OPACITY_TRANSPARENT, granulating=True),
    Paint("Phthalo Blue", "#0d4f8b", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("Burnt Umber", "#4f3320", opacity=OPACITY_SEMI),
)

# Typical 12-color watercolor palette. No white: the paper is the white.
WATERCOLOR_STARTER: Tuple[Paint, ...] = (
    Paint("Lemon Yellow", "#f4e04d", opacity=OPACITY_TRANSPARENT),
    Paint("New Gamboge", "#e8a33d", opacity=OPACITY_TRANSPARENT),
    Paint("Yellow Ochre", "#c49441", opacity=OPACITY_SEMI, granulating=True),
    Paint("Burnt Sienna", "#8a4a2a", opacity=OPACITY_TRANSPARENT, granulating=True),
    Paint("Burnt Umber", "#5c4029", opacity=OPACITY_SEMI, granulating=True),
    Paint("Cadmium Red", "#ce3b30", opacity=OPACITY_SEMI),
    Paint("Alizarin Crimson", "#98213f", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("Quinacridone Rose", "#b93a66", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("French Ultramarine", "#2d4290", opacity=OPACITY_TRANSPARENT, granulating=True),
    Paint("Cobalt Blue", "#2e5fa3", opacity=OPACITY_SEMI, granulating=True),
    Paint("Phthalo Blue", "#0e5290", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("Sap Green", "#4e7032", opacity=OPACITY_TRANSPARENT, staining=True),
)

# Basic 6-color acrylic set.
ACRYLIC_STARTER: Tuple[Paint, ...] = (
    Paint("Titanium White", "#f6f4ef", is_white=True, opacity=OPACITY_OPAQUE),
    Paint("Primary Yellow", "#f5d327", opacity=OPACITY_SEMI),
    Paint("Naphthol Red", "#d03a2b", opacity=OPACITY_SEMI),
    Paint("Phthalo Blue", "#14508c", opacity=OPACITY_TRANSPARENT, staining=True),
    Paint("Burnt Umber", "#4f3320", opacity=OPACITY_SEMI),
    Paint("Mars Black", "#2b2b2b", opacity=OPACITY_OPAQUE),
)

# Representative 24-stick oil pastel set ("mixing" happens on the surface).
OILPASTEL_STARTER: Tuple[Paint, ...] = (
    Paint("White", "#f5f2ea", is_white=True, opacity=OPACITY_OPAQUE),
    Paint("Lemon Yellow", "#f2e33b", opacity=OPACITY_OPAQUE),
    Paint("Yellow", "#f5c927", opacity=OPACITY_OPAQUE),
    Paint("Golden Yellow", "#efa32b", opacity=OPACITY_OPAQUE),
    Paint("Orange", "#e87c26", opacity=OPACITY_OPAQUE),
    Paint("Vermilion", "#dd5226", opacity=OPACITY_OPAQUE),
    Paint("Red", "#c93434", opacity=OPACITY_OPAQUE),
    Paint("Carmine", "#a82649", opacity=OPACITY_OPAQUE),
    Paint("Pink", "#e08a9b", opacity=OPACITY_OPAQUE),
    Paint("Magenta", "#a04679", opacity=OPACITY_OPAQUE),
    Paint("Violet", "#6e4488", opacity=OPACITY_OPAQUE),
    Paint("Blue Violet", "#4a4490", opacity=OPACITY_OPAQUE),
    Paint("Ultramarine", "#31479b", opacity=OPACITY_OPAQUE),
    Paint("Blue", "#2a5ca8", opacity=OPACITY_OPAQUE),
    Paint("Sky Blue", "#5c8fc6", opacity=OPACITY_OPAQUE),
    Paint("Turquoise", "#2e8e9c", opacity=OPACITY_OPAQUE),
    Paint("Emerald Green", "#2f8c57", opacity=OPACITY_OPAQUE),
    Paint("Green", "#3a7a3c", opacity=OPACITY_OPAQUE),
    Paint("Yellow Green", "#8fae3c", opacity=OPACITY_OPAQUE),
    Paint("Olive", "#77702f", opacity=OPACITY_OPAQUE),
    Paint("Yellow Ochre", "#c08a3e", opacity=OPACITY_OPAQUE),
    Paint("Brown", "#7a4a28", opacity=OPACITY_OPAQUE),
    Paint("Van Dyke Brown", "#553a26", opacity=OPACITY_OPAQUE),
    Paint("Black", "#262626", opacity=OPACITY_OPAQUE),
)

STARTERS: dict[str, Tuple[Paint, ...]] = {
    "oil": OIL_STARTER,
    "watercolor": WATERCOLOR_STARTER,
    "acrylic": ACRYLIC_STARTER,
    "oilpastel": OILPASTEL_STARTER,
}
