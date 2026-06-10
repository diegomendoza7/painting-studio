"""Center preview: shows the rendered block-in, scaled to fit the viewport.

Zoom is a pure *view* operation — it scales the already-rendered ``QImage`` and
never re-runs the pipeline. Pinch (Mac trackpad), Ctrl/mouse wheel and the
toolbar +/−/Fit buttons all drive it; two-finger scroll pans when zoomed in. The
zoom level is a multiplier on the fit-scale (1.0 = fit), preserved across the
slider-driven re-renders and reset only when a new subject (new working-image
size) arrives.

Also captures clicks (mapped back to source-image coordinates via ``_draw_rect``,
which already accounts for zoom/pan) so SAM precision mode and analysis mode can
be prompted with points.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

MIN_ZOOM = 1.0      # 1.0 == fit-to-viewport (the minimum; you never zoom out past fit)
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25    # multiplicative step per +/− click or wheel notch


class PreviewWidget(QWidget):
    clicked = Signal(float, float, int)  # source x, y, label (1=fg, 0=bg)
    zoom_changed = Signal(float)         # current zoom multiplier (1.0 = fit)

    def __init__(self):
        super().__init__()
        self._image: Optional[QImage] = None
        self._src_w = 1
        self._src_h = 1
        self._margin = 0                  # border baked into the image around the content
        self._flip_h = False              # mirror the view (error-check), view-only
        self._flip_v = False
        self._draw_rect = QRectF()
        self._zoom = 1.0
        self._pan = QPointF()             # image-center offset from the viewport center
        self._placeholder = "Open an image or PDF to begin"
        self.setMinimumSize(360, 360)
        self.setAutoFillBackground(False)

    def set_image(self, image: QImage, src_w: int, src_h: int, margin: int = 0):
        new_dims = (max(1, src_w), max(1, src_h))
        if (self._src_w, self._src_h) != new_dims:
            # New subject (different working-image size): reset the view. Same-size
            # re-renders (slider edits) keep the current zoom/pan so inspection
            # survives live tuning. Keyed on the *content* size, so a guide toggle that
            # only changes the border doesn't reset the view.
            self._zoom = 1.0
            self._pan = QPointF()
            self.zoom_changed.emit(self._zoom)
        self._image = image
        self._src_w, self._src_h = new_dims
        self._margin = max(0, int(margin))
        self.update()

    def set_flip(self, horizontal: bool, vertical: bool):
        """Mirror the preview (the atelier 'flip to check errors' move). Pure view —
        never re-runs the pipeline; clicks are un-mirrored back to image coords."""
        self._flip_h, self._flip_v = bool(horizontal), bool(vertical)
        self.update()

    def clear(self, message: str = "Open an image or PDF to begin"):
        self._image = None
        self._placeholder = message
        self.update()

    # -- zoom API (view-only) ----------------------------------------------
    def set_zoom(self, zoom: float, focal: Optional[QPointF] = None):
        """Set the zoom multiplier (clamped to ``[MIN_ZOOM, MAX_ZOOM]``). When a
        *focal* point (widget coords) is given, the image content under it is kept
        in place — so pinch/wheel zoom toward the cursor or gesture."""
        z = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
        if focal is not None and self._zoom > 0:
            ac = self._avail().center()
            ratio = z / self._zoom
            self._pan = QPointF(
                self._pan.x() + (focal.x() - ac.x() - self._pan.x()) * (1.0 - ratio),
                self._pan.y() + (focal.y() - ac.y() - self._pan.y()) * (1.0 - ratio),
            )
        changed = abs(z - self._zoom) > 1e-9
        self._zoom = z
        if z <= MIN_ZOOM + 1e-9:
            self._pan = QPointF()         # back at fit → recenter
        self.update()
        if changed:
            self.zoom_changed.emit(self._zoom)

    def zoom_in(self):
        self.set_zoom(self._zoom * ZOOM_STEP)

    def zoom_out(self):
        self.set_zoom(self._zoom / ZOOM_STEP)

    def zoom_fit(self):
        self._zoom = MIN_ZOOM
        self._pan = QPointF()
        self.update()
        self.zoom_changed.emit(self._zoom)

    # -- geometry helpers ---------------------------------------------------
    def _avail(self) -> QRectF:
        return QRectF(self.rect()).adjusted(8, 8, -8, -8)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            painter.end()
            return

        avail = self._avail()
        iw, ih = self._image.width(), self._image.height()
        base = min(avail.width() / iw, avail.height() / ih)
        scale = base * self._zoom
        dw, dh = iw * scale, ih * scale

        # Clamp the pan so the image can't be dragged clear of the viewport; when an
        # axis is no larger than the viewport (e.g. at fit) it stays centered there.
        max_x = max(0.0, (dw - avail.width()) / 2.0)
        max_y = max(0.0, (dh - avail.height()) / 2.0)
        px = max(-max_x, min(max_x, self._pan.x()))
        py = max(-max_y, min(max_y, self._pan.y()))
        self._pan = QPointF(px, py)

        cx = avail.x() + avail.width() / 2.0 + px
        cy = avail.y() + avail.height() / 2.0 + py
        self._draw_rect = QRectF(cx - dw / 2.0, cy - dh / 2.0, dw, dh)

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setClipRect(avail)        # keep a zoomed image inside the gutter
        img = self._image
        if self._flip_h or self._flip_v:
            img = img.mirrored(self._flip_h, self._flip_v)
        painter.drawImage(self._draw_rect, img)
        painter.end()

    # -- input: pinch / wheel ----------------------------------------------
    def event(self, e):
        # Mac trackpad pinch arrives as a native zoom gesture; value() is the
        # incremental scale delta and position() its anchor in widget coords.
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:
            self.set_zoom(self._zoom * (1.0 + e.value()), focal=e.position())
            return True
        return super().event(e)

    def wheelEvent(self, e):
        pixel = e.pixelDelta()
        if (e.modifiers() & Qt.ControlModifier) or pixel.isNull():
            # Ctrl-wheel, or a classic mouse wheel (no pixelDelta) → zoom at cursor.
            dy = e.angleDelta().y()
            if dy:
                self.set_zoom(
                    self._zoom * (ZOOM_STEP if dy > 0 else 1.0 / ZOOM_STEP),
                    focal=e.position(),
                )
        else:
            # Trackpad two-finger scroll → pan (a no-op at fit, where pan clamps to 0).
            self._pan = QPointF(self._pan.x() + pixel.x(), self._pan.y() + pixel.y())
            self.update()
        e.accept()

    def mousePressEvent(self, event):
        if self._image is None or self._draw_rect.isEmpty():
            return
        pos = event.position()
        if not self._draw_rect.contains(pos):
            return
        fx = (pos.x() - self._draw_rect.x()) / self._draw_rect.width()
        fy = (pos.y() - self._draw_rect.y()) / self._draw_rect.height()
        # Un-mirror first so a click lands on the image point under the cursor, not its
        # mirror, then undo the margin border (below).
        if self._flip_h:
            fx = 1.0 - fx
        if self._flip_v:
            fy = 1.0 - fy
        # The rendered image carries a border of self._margin around the content, so the
        # drawn rect spans scene coords [-m, src+m]. Undo the border to recover the
        # working-image coordinate under the cursor (m == 0 → fx*src, unchanged).
        m = self._margin
        sx = fx * (self._src_w + 2 * m) - m
        sy = fy * (self._src_h + 2 * m) - m
        label = 1 if event.button() == Qt.LeftButton else 0
        self.clicked.emit(sx, sy, label)
