"""Serialize a ``Scene`` to a layered SVG string.

Layers are emitted as Inkscape layer groups (``inkscape:groupmode="layer"``)
so they open as separable, named layers in Inkscape and as named groups in
Illustrator. Only geometry that the scene marks visible is written. Image
patches (the reference photo) are embedded as base64 PNG ``<image>`` elements.
"""
from __future__ import annotations

import base64
import io
from typing import List
from xml.sax.saxutils import escape

import numpy as np

from .params import CM_PER_INCH, UNIT_CM
from .scene import ImagePatch, Point, Scene, Stroke, Text


def _fmt(v: float) -> str:
    # Compact fixed-point; strips a trailing ".0" so paths stay small.
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _ring_d(pts: List[Point], close: bool) -> str:
    out = [f"M {_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
    for x, y in pts[1:]:
        out.append(f"L {_fmt(x)} {_fmt(y)}")
    if close:
        out.append("Z")
    return " ".join(out)


def _path_d(stroke: Stroke) -> str:
    pts = stroke.points
    if not pts:
        return ""
    parts = [_ring_d(pts, stroke.closed)]
    for hole in stroke.holes:  # interior holes as extra (closed) subpaths
        if len(hole) >= 2:
            parts.append(_ring_d(hole, True))
    return " ".join(parts)


def _stroke_to_path(stroke: Stroke) -> str:
    d = _path_d(stroke)
    if not d:
        return ""
    attrs = [
        f'd="{d}"',
        f'stroke="{stroke.color}"',
        f'stroke-width="{_fmt(stroke.width)}"',
        'stroke-linejoin="round"',
        'stroke-linecap="round"',
    ]
    if stroke.width <= 0:
        attrs[1] = 'stroke="none"'
    if stroke.fill:
        attrs.append(f'fill="{stroke.fill}"')
        if stroke.holes:
            attrs.append('fill-rule="evenodd"')
        if stroke.fill_opacity < 1.0:
            attrs.append(f'fill-opacity="{_fmt(stroke.fill_opacity)}"')
    else:
        attrs.append('fill="none"')
    if stroke.opacity < 1.0:
        attrs.append(f'opacity="{_fmt(stroke.opacity)}"')
    return "  <path " + " ".join(attrs) + " />"


def _text_to_svg(t: Text) -> str:
    attrs = [
        f'x="{_fmt(t.x)}"',
        f'y="{_fmt(t.y)}"',
        f'font-size="{_fmt(t.size)}"',
        'font-family="sans-serif"',
        f'fill="{t.color}"',
        f'text-anchor="{t.anchor}"',
    ]
    if t.opacity < 1.0:
        attrs.append(f'opacity="{_fmt(t.opacity)}"')
    return "  <text " + " ".join(attrs) + f">{escape(t.text)}</text>"


def _png_data_uri(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def _patch_to_svg(p: ImagePatch) -> str:
    uri = _png_data_uri(p.rgb)
    attrs = [
        f'x="{_fmt(p.x)}"',
        f'y="{_fmt(p.y)}"',
        f'width="{_fmt(p.w)}"',
        f'height="{_fmt(p.h)}"',
    ]
    if p.opacity < 1.0:
        attrs.append(f'opacity="{_fmt(p.opacity)}"')
    return "  <image " + " ".join(attrs) + f' xlink:href="{uri}" preserveAspectRatio="none" />'


def scene_to_svg(scene: Scene) -> str:
    w, h = scene.width, scene.height
    m = max(0, int(scene.margin))
    vw, vh = w + 2 * m, h + 2 * m
    # When a real-world scale is known (the artist set a card width), express
    # the document in physical units while keeping the px viewBox — so it
    # opens 1:1 in Inkscape/Illustrator and prints to scale.
    ipp = float(scene.inches_per_px)
    if ipp > 0:
        w_phys, h_phys = vw * ipp, vh * ipp  # true inches
        if scene.canvas_units == UNIT_CM:
            w_phys, h_phys, unit = w_phys * CM_PER_INCH, h_phys * CM_PER_INCH, "cm"
        else:
            unit = "in"
        dim = f'width="{_fmt(w_phys)}{unit}" height="{_fmt(h_phys)}{unit}"'
    else:
        dim = f'width="{vw}" height="{vh}"'
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'{dim} viewBox="{-m} {-m} {vw} {vh}">'
    )
    lines.append(
        f'  <rect x="{-m}" y="{-m}" width="{vw}" height="{vh}" fill="{scene.background}" />'
    )

    for patch in scene.patches:
        lines.append(_patch_to_svg(patch))

    for layer in scene.layers:
        if not layer.visible or (not layer.strokes and not layer.texts):
            continue
        label = escape(layer.name)
        lines.append(
            f'  <g id="{label}" inkscape:groupmode="layer" '
            f'inkscape:label="{label}">'
        )
        for stroke in layer.strokes:
            p = _stroke_to_path(stroke)
            if p:
                lines.append("  " + p)
        for text in layer.texts:
            lines.append("  " + _text_to_svg(text))
        lines.append("  </g>")

    lines.append("</svg>")
    return "\n".join(lines)
