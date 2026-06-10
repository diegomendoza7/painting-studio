"""Raster renderer: draw a ``Scene`` with QPainter.

Used both for the live preview and for PNG export, so what you see is exactly
what gets exported. Kept in the UI package because it depends on Qt; the SVG
renderer (``studio.svg``) is the Qt-free equivalent.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)

from ..scene import Scene, Stroke, Text


def ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    """RGB uint8 HxWx3 -> QImage (copied so it owns its buffer)."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _build_path(stroke: Stroke) -> QPainterPath:
    path = QPainterPath()  # default OddEvenFill makes the hole subpaths knock out
    pts = stroke.points
    if not pts:
        return path
    path.moveTo(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    if stroke.closed:
        path.closeSubpath()
    for hole in stroke.holes:
        if len(hole) >= 2:
            path.moveTo(hole[0][0], hole[0][1])
            for hx, hy in hole[1:]:
                path.lineTo(hx, hy)
            path.closeSubpath()
    return path


def _draw_text(painter: QPainter, t: Text) -> None:
    painter.setOpacity(t.opacity)
    painter.setPen(QColor(t.color))
    font = QFont()
    font.setPixelSize(max(1, int(round(t.size))))
    painter.setFont(font)
    x = t.x
    if t.anchor in ("middle", "end"):
        w = QFontMetricsF(font).horizontalAdvance(t.text)
        x -= w / 2.0 if t.anchor == "middle" else w
    painter.drawText(QPointF(x, t.y), t.text)


def render_scene(scene: Scene, scale: float = 1.0) -> QImage:
    m = max(0, int(scene.margin))
    w = max(1, round((scene.width + 2 * m) * scale))
    h = max(1, round((scene.height + 2 * m) * scale))
    img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(scene.background))

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(scale, scale)
    if m:
        painter.translate(m, m)

    # Raster patches (the reference photo) sit beneath the vector layers.
    for patch in scene.patches:
        painter.setOpacity(float(patch.opacity))
        painter.drawImage(
            QRectF(patch.x, patch.y, patch.w, patch.h), ndarray_to_qimage(patch.rgb)
        )
    painter.setOpacity(1.0)

    for layer in scene.layers:
        if not layer.visible:
            continue
        for stroke in layer.strokes:
            painter.setOpacity(stroke.opacity)
            if stroke.width > 0:
                pen = QPen(QColor(stroke.color))
                pen.setWidthF(stroke.width)
                pen.setJoinStyle(Qt.RoundJoin)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.NoPen)
            if stroke.fill:
                fill = QColor(stroke.fill)
                fill.setAlphaF(stroke.fill_opacity)
                painter.setBrush(QBrush(fill))
            else:
                painter.setBrush(Qt.NoBrush)
            painter.drawPath(_build_path(stroke))
        for text in layer.texts:
            _draw_text(painter, text)
    painter.setOpacity(1.0)
    painter.end()
    return img
