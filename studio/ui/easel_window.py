"""Easel mode — the window you keep open beside the canvas while you paint.

Click a region of the reference and it fills the window as one big swatch on a
neutral grey, with the recipe from your own palette: what to mix, in what order,
and roughly how much. Click again to go back to the photo, with the region you
matched still outlined. Every swatch is saved to the painting's history as you
go, so a colour you mixed last week is one search away.

Deliberately separate from the document window: this is a live tool, not a
sheet for printing, and it wants the whole screen and very few controls.

Keys: Space back to the photo · Esc back, then clear the selection · ⌘D clear
the selection · V value (colour off) · E switch between Area and Eyedropper ·
⌫ delete swatch · ⌘N new painting · ⌘F search · ⌃⌘F full screen.
Option-click takes the exact pixel when picking areas (and the area average when
using the eyedropper). Clicking the margin around the photo clears the selection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QStandardPaths, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import easel
from ..library import (
    Library,
    NamedPalette,
    Project,
    Swatch,
    new_swatch_id,
    palette_or_first,
)
from ..models import Target
from ..palettes import STARTERS
from ..params import UNIT_CM, UNIT_IN
from ..pipeline.loader import load, normalize
from ..pipeline.plan_schedule import parts_text
from ..pipeline.solver import solve_targets
from .palette_panel import PalettePanel
from .render import ndarray_to_qimage

DISPLAY_EDGE = 1600
LOUPE_HALF = 7          # the loupe shows a 15 × 15 pixel neighbourhood…
LOUPE_CELL = 9          # …each pixel drawn 9 screen pixels wide
CARD_BG = "#f4f2ec"
CARD_INK = "#1b1b1b"
HINT = "Click anywhere to go back to the photo   ·   V value   ·   E eyedropper   ·   ⌫ delete   ·   Esc Esc clears the selection"


def library_root() -> str:
    """``~/Documents/Painting Studio``, unless ``PAINTING_STUDIO_LIBRARY`` says
    otherwise — which is how the test suite keeps its hands off the real one."""
    override = os.environ.get("PAINTING_STUDIO_LIBRARY")
    if override:
        return override
    docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return os.path.join(docs or os.path.expanduser("~/Documents"), "Painting Studio")


# --- the swatch card ---------------------------------------------------------------
@dataclass
class SwatchCard:
    title: str
    target_hex: str
    mixed_hex: str
    delta_e: float
    achievable: bool
    description: str
    value_text: str
    ratio_text: str
    amount_text: str
    steps: List[str] = field(default_factory=list)
    palette_name: str = ""
    crop: Optional[QImage] = None


class EaselCanvas(QWidget):
    """The reference photo, or one colour filling the view."""

    picked = Signal(float, float, bool)   # normalized x, y, Option held
    dismissed = Signal()                  # clicked while a swatch is showing
    empty_clicked = Signal()
    deselected = Signal()                 # clicked the margin around the photo

    def __init__(self):
        super().__init__()
        self._mode = "empty"
        self._photo: Optional[QImage] = None
        self._photo_grey: Optional[QImage] = None
        self._rgb: Optional[np.ndarray] = None
        self._overlay: Optional[QImage] = None
        self._card: Optional[SwatchCard] = None
        self._value = False
        self._rect = QRectF()
        self._placeholder = ""
        self._marker = None                # (x, y) of an exact-pixel pick
        self._loupe_mode = False           # the eyedropper is the active pick
        self._hover = None                 # cursor position the loupe follows
        self.setMinimumSize(420, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # -- state --------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._mode = "empty"
        self._photo = self._photo_grey = None
        self._rgb = None
        self.update()

    def set_photo(self, rgb: np.ndarray) -> None:
        self._rgb = rgb
        self._photo = ndarray_to_qimage(rgb)
        self._photo_grey = None
        self._overlay = None
        self._mode = "photo"
        self.update()

    def set_marker(self, x: float, y: float) -> None:
        """Mark one exact point on the photo — no dimming, just a ring round the pixel."""
        self._overlay = None
        self._marker = (float(x), float(y))
        self.update()

    def set_loupe_mode(self, on: bool) -> None:
        """Magnify the pixels under the cursor, so an exact pick lands where you mean."""
        self._loupe_mode = bool(on)
        if not on:
            self._hover = None
        self.update()

    def set_region(self, mask: Optional[np.ndarray]) -> None:
        """Outline a region on the photo: everything else dimmed, the edge in white."""
        self._marker = None
        if mask is None:
            self._overlay = None
        else:
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[~mask] = (0, 0, 0, 70)
            rgba[easel.region_outline(mask)] = (255, 255, 255, 235)
            self._overlay = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w,
                                   QImage.Format_RGBA8888).copy()
        self.update()

    @property
    def has_region(self) -> bool:
        return self._overlay is not None or self._marker is not None

    def show_photo(self) -> None:
        if self._photo is not None:
            self._mode = "photo"
            self.update()

    def show_swatch(self, card: SwatchCard) -> None:
        self._card = card
        self._mode = "swatch"
        self.update()

    def set_value_mode(self, on: bool) -> None:
        self._value = bool(on)
        if on and self._photo_grey is None and self._rgb is not None:
            self._photo_grey = ndarray_to_qimage(easel.greyscale(self._rgb))
        self.update()

    # -- input ----------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._mode == "swatch":
            self.dismissed.emit()
            return
        if self._mode != "photo" or self._rect.isEmpty():
            self.empty_clicked.emit()
            return
        pos = event.position()
        if not self._rect.contains(pos):
            self.deselected.emit()        # the margin clears the selection
            return
        nx = (pos.x() - self._rect.x()) / self._rect.width()
        ny = (pos.y() - self._rect.y()) / self._rect.height()
        self.picked.emit(nx, ny, bool(event.modifiers() & Qt.AltModifier))

    def mouseMoveEvent(self, event):
        # The loupe shows whenever the next click would be an exact pick: in
        # Eyedropper mode, or with Option held in the other modes.
        alt = bool(event.modifiers() & Qt.AltModifier)
        want = (self._loupe_mode != alt and self._mode == "photo" and self._rgb is not None
                and self._rect.contains(event.position()))
        hover = event.position() if want else None
        if hover is not None or self._hover is not None:
            self._hover = hover
            self.update()

    def leaveEvent(self, _event):
        if self._hover is not None:
            self._hover = None
            self.update()

    # -- painting --------------------------------------------------------------------
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._mode == "swatch" and self._card is not None:
            self._paint_swatch(p)
        elif self._mode == "photo" and self._photo is not None:
            self._paint_photo(p)
        else:
            p.fillRect(self.rect(), QColor("#2b2b2b"))
            p.setPen(QColor("#bbbbbb"))
            f = QFont()
            f.setPixelSize(18)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
        p.end()

    def _paint_photo(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor("#2b2b2b"))
        img = self._photo_grey if (self._value and self._photo_grey is not None) else self._photo
        avail = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        s = min(avail.width() / img.width(), avail.height() / img.height())
        w, h = img.width() * s, img.height() * s
        self._rect = QRectF(avail.center().x() - w / 2, avail.center().y() - h / 2, w, h)
        p.drawImage(self._rect, img)
        if self._overlay is not None:
            p.drawImage(self._rect, self._overlay)
        if self._marker is not None:
            self._paint_marker(p)
        if self._hover is not None:
            self._paint_loupe(p)

    def _to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._rect.x() + x * self._rect.width(), self._rect.y() + y * self._rect.height())

    def _paint_marker(self, p: QPainter) -> None:
        """A ring with ticks round the exact point — visible on light and dark alike."""
        c = self._to_widget(*self._marker)
        for colour, width in (("#000000", 4.0), ("#ffffff", 1.6)):
            p.setPen(QPen(QColor(colour), width))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 9.0, 9.0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p.drawLine(QPointF(c.x() + dx * 12, c.y() + dy * 12),
                           QPointF(c.x() + dx * 18, c.y() + dy * 18))

    def _paint_loupe(self, p: QPainter) -> None:
        """Magnified pixels round the cursor; the outlined centre one is what a click takes."""
        pos = self._hover
        nx = (pos.x() - self._rect.x()) / self._rect.width()
        ny = (pos.y() - self._rect.y()) / self._rect.height()
        patch = easel.loupe_patch(self._rgb, nx, ny, LOUPE_HALF)
        centre = patch[LOUPE_HALF, LOUPE_HALF]
        if self._value:
            patch = easel.greyscale(patch)
        n = patch.shape[0]
        size = n * LOUPE_CELL
        x0, y0 = pos.x() + 24, pos.y() + 24
        if x0 + size > self.width() - 4:
            x0 = pos.x() - 24 - size
        if y0 + size + 30 > self.height() - 4:
            y0 = pos.y() - 24 - size - 30
        box = QRectF(x0, y0, size, size)
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)   # hard pixel edges
        p.drawImage(box, ndarray_to_qimage(patch))
        p.restore()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#f4f2ec"), 2))
        p.drawRect(box)
        cell = QRectF(x0 + LOUPE_HALF * LOUPE_CELL, y0 + LOUPE_HALF * LOUPE_CELL, LOUPE_CELL, LOUPE_CELL)
        p.setPen(QPen(QColor("#000000"), 3))
        p.drawRect(cell)
        p.setPen(QPen(QColor("#ffffff"), 1.2))
        p.drawRect(cell)
        hex_ = "#%02x%02x%02x" % tuple(int(v) for v in centre)
        label = QRectF(x0, y0 + size + 4, size, 24)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 20, 20, 225))
        p.drawRoundedRect(label, 4, 4)
        p.setBrush(QColor(hex_))
        p.drawRect(QRectF(label.x() + 5, label.y() + 5, 14, 14))
        f = QFont()
        f.setPixelSize(12)
        p.setFont(f)
        p.setPen(QColor("#f4f2ec"))
        p.drawText(label.adjusted(26, 0, 0, 0), int(Qt.AlignLeft | Qt.AlignVCenter), f"exact  {hex_}")

    def _paint_swatch(self, p: QPainter) -> None:
        c = self._card
        target = easel.grey_of(c.target_hex) if self._value else c.target_hex
        mixed = easel.grey_of(c.mixed_hex) if self._value else c.mixed_hex
        p.fillRect(self.rect(), QColor(easel.NEUTRAL_GREY))
        m = 28.0
        full = QRectF(self.rect()).adjusted(m, m, -m, -m - 22)
        card_w = max(300.0, min(460.0, full.width() * 0.38))
        left = QRectF(full.x(), full.y(), full.width() - card_w - m, full.height())
        card = QRectF(left.right() + m, full.y(), card_w, full.height())

        # The colour, big. The target fills most of it; the closest mix your
        # palette reaches sits underneath so the gap is visible, not just a number.
        big = QRectF(left.x(), left.y(), left.width(), left.height() * 0.80)
        strip = QRectF(left.x(), big.bottom() + 10, left.width(), left.height() * 0.20 - 10)
        # A thin off-white keyline, like the border on a grey card: without it a
        # mid-grey swatch dissolves into the mid-grey surround.
        p.setPen(QPen(QColor("#ecebe6"), 3))
        p.setBrush(QColor(target))
        p.drawRect(big)
        p.setBrush(QColor(mixed))
        p.drawRect(strip)
        p.setPen(Qt.NoPen)
        label = (f"Closest mix from {c.palette_name}: {easel.match_words(c.delta_e)} "
                 f"(ΔE {c.delta_e:.1f})"
                 + ("" if c.achievable else "  ▲ your palette can't quite reach this"))
        self._label(p, strip.adjusted(12, 0, -12, 0), label, 13, mixed)

        # The recipe card.
        p.setBrush(QColor(CARD_BG))
        p.drawRoundedRect(card, 10, 10)
        inner = card.adjusted(22, 20, -22, -20)
        y = inner.y()
        y += self._text(p, QRectF(inner.x(), y, inner.width(), 200), c.title, 22, bold=True) + 4
        y += self._text(p, QRectF(inner.x(), y, inner.width(), 200), c.description, 14,
                        color="#5a5a5a") + 12
        if c.crop is not None:
            thumb = QRectF(inner.x(), y, 84, 84)
            p.drawImage(thumb, c.crop)
            p.setPen(QPen(QColor("#9a9a9a"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(thumb)
            self._text(p, QRectF(inner.x() + 96, y + 8, inner.width() - 96, 80),
                       c.value_text + "\nwhere it came from", 13, color="#5a5a5a")
            y += 96
        else:
            y += self._text(p, QRectF(inner.x(), y, inner.width(), 60), c.value_text, 13,
                            color="#5a5a5a") + 10
        y += self._text(p, QRectF(inner.x(), y, inner.width(), 30), "MIX", 12,
                        bold=True, color="#7a7a7a") + 4
        y += self._text(p, QRectF(inner.x(), y, inner.width(), 200), c.ratio_text, 19, bold=True) + 8
        if c.amount_text:
            y += self._text(p, QRectF(inner.x(), y, inner.width(), 200), c.amount_text, 14) + 12
        for i, step in enumerate(c.steps, start=1):
            if y > inner.bottom() - 20:
                break
            y += self._text(p, QRectF(inner.x(), y, inner.width(), 400), f"{i}.  {step}", 15) + 7

        hint = QRectF(self.rect()).adjusted(m, self.height() - 30, -m, -6)
        self._text(p, hint, HINT, 12, color="#e6e6e6")

    @staticmethod
    def _text(p: QPainter, rect: QRectF, text: str, px: int, bold: bool = False,
              color: str = CARD_INK) -> float:
        f = QFont()
        f.setPixelSize(px)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(QColor(color))
        flags = int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap)
        used = p.boundingRect(rect, flags, text)
        p.drawText(rect, flags, text)
        return used.height()

    @staticmethod
    def _label(p: QPainter, rect: QRectF, text: str, px: int, behind: str) -> None:
        """Text on a colour, in whichever of black or white reads on it."""
        f = QFont()
        f.setPixelSize(px)
        p.setFont(f)
        L, _v = easel.value_of(behind)
        p.setPen(QColor("#1b1b1b" if L > 55 else "#f4f2ec"))
        p.drawText(rect, int(Qt.AlignLeft | Qt.AlignVCenter), text)


# --- dialogs ------------------------------------------------------------------------
class PaintingDialog(QDialog):
    """New painting, or the settings of an existing one."""

    def __init__(self, parent, palettes: List[str], project: Optional[Project] = None,
                 default: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Painting settings" if project else "New painting")
        self._reference = ""
        form = QFormLayout(self)
        self.name = QLineEdit(project.name if project else "")
        self.name.setPlaceholderText("e.g. Still life with lemons")
        form.addRow("Name", self.name)
        if project is None:
            row = QHBoxLayout()
            self.ref_label = QLabel("(none yet)")
            pick = QPushButton("Choose…")
            pick.clicked.connect(self._choose)
            row.addWidget(self.ref_label, 1)
            row.addWidget(pick)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow("Reference photo", holder)
        size = QHBoxLayout()
        self.w = QDoubleSpinBox()
        self.h = QDoubleSpinBox()
        for sb, v in ((self.w, project.canvas_w if project else 16.0),
                      (self.h, project.canvas_h if project else 20.0)):
            sb.setRange(0.0, 400.0)
            sb.setDecimals(1)
            sb.setValue(v)
        self.units = QComboBox()
        self.units.addItem("in", UNIT_IN)
        self.units.addItem("cm", UNIT_CM)
        self.units.setCurrentIndex(1 if project and project.units == UNIT_CM else 0)
        size.addWidget(self.w)
        size.addWidget(QLabel("×"))
        size.addWidget(self.h)
        size.addWidget(self.units)
        holder = QWidget()
        holder.setLayout(size)
        form.addRow("Canvas", holder)
        form.addRow(QLabel("The canvas size is what turns a region into an amount of paint."))
        self.palette = QComboBox()
        self.palette.addItems(palettes)
        if project and project.palette in palettes:
            self.palette.setCurrentText(project.palette)
        elif default in palettes:
            self.palette.setCurrentText(default)
        form.addRow("Palette", self.palette)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference photo", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.heic)")
        if path:
            self._reference = path
            self.ref_label.setText(os.path.basename(path))
            if not self.name.text().strip():
                self.name.setText(os.path.splitext(os.path.basename(path))[0])

    def values(self) -> dict:
        return {
            "name": self.name.text().strip() or "Untitled",
            "reference": self._reference,
            "canvas_w": float(self.w.value()),
            "canvas_h": float(self.h.value()),
            "units": self.units.currentData(),
            "palette": self.palette.currentText(),
        }


# --- the window ------------------------------------------------------------------------
class EaselWindow(QMainWindow):
    def __init__(self, library: Optional[Library] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Easel — Painting Studio")
        self.resize(1400, 900)
        self.lib = library or Library(library_root())
        self.project: Optional[Project] = None
        self.palette: NamedPalette = palette_or_first(self.lib, self.lib.default_palette_name())
        self.analysis: Optional[easel.Analysis] = None
        self.display_rgb: Optional[np.ndarray] = None
        self.current: Optional[str] = None       # swatch id on show
        self._syncing = False

        self._build()
        self._reload_palettes()
        self._reload_projects(QSettings().value("easel/project", "", str))

    # -- construction ---------------------------------------------------------------
    def _build(self):
        self.canvas = EaselCanvas()
        self.canvas.picked.connect(self._on_picked)
        self.canvas.dismissed.connect(self._back_to_photo)
        self.canvas.empty_clicked.connect(self._on_empty_clicked)
        self.canvas.deselected.connect(self._deselect)

        # History: search, scope, list, details.
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(8, 8, 8, 8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search swatches — name, notes, paint…")
        self.search.textChanged.connect(self._refresh_history)
        self.scope = QComboBox()
        self.scope.addItem("This painting", "this")
        self.scope.addItem("All paintings", "all")
        self.scope.currentIndexChanged.connect(self._refresh_history)
        self.history = QListWidget()
        self.history.setIconSize(QPixmap(96, 48).size())
        self.history.setWordWrap(True)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history.itemClicked.connect(self._on_history_clicked)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name this swatch (e.g. shadow under the chin)")
        self.name_edit.editingFinished.connect(self._save_details)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Tags, comma separated")
        self.tags_edit.editingFinished.connect(self._save_details)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Notes — what worked, what to try")
        self.notes_edit.setMaximumHeight(80)
        self.remix_btn = QPushButton("Re-mix with current palette")
        self.remix_btn.clicked.connect(self._remix)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_current)
        btns = QHBoxLayout()
        btns.addWidget(self.remix_btn, 1)
        btns.addWidget(self.delete_btn)
        lay.addWidget(self.search)
        lay.addWidget(self.scope)
        lay.addWidget(self.history, 1)
        lay.addWidget(QLabel("Selected swatch"))
        lay.addWidget(self.name_edit)
        lay.addWidget(self.tags_edit)
        lay.addWidget(self.notes_edit)
        lay.addLayout(btns)
        side.setMinimumWidth(260)
        side.setMaximumWidth(380)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(side)
        split.addWidget(self.canvas)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        # Palette editor, in a dock that stays out of the way until wanted.
        self.palette_panel = PalettePanel()
        self.palette_panel.palette_changed.connect(self._on_palette_edited)
        self.palette_panel.starter_requested.connect(self._on_palette_starter)
        dock_body = QWidget()
        dl = QVBoxLayout(dock_body)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.palette_panel, 1)
        row = QHBoxLayout()
        save_as = QPushButton("Save as new palette…")
        save_as.clicked.connect(self._palette_save_as)
        remove = QPushButton("Delete palette")
        remove.clicked.connect(self._palette_delete)
        row.addWidget(save_as)
        row.addWidget(remove)
        dl.addLayout(row)
        make_default = QPushButton("Make this my palette")
        make_default.setToolTip("New paintings start with it, and the main window "
                                "falls back to it on Reset.")
        make_default.clicked.connect(self._palette_make_default)
        dl.addWidget(make_default)
        self.palette_dock = QDockWidget("Palette — your paints", self)
        self.palette_dock.setWidget(dock_body)
        self.addDockWidget(Qt.RightDockWidgetArea, self.palette_dock)
        self.palette_dock.hide()

        # Toolbar.
        tb = self.addToolBar("Easel")
        tb.setMovable(False)
        tb.addWidget(QLabel(" Painting "))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(180)
        self.project_combo.currentIndexChanged.connect(self._on_project_chosen)
        tb.addWidget(self.project_combo)
        self.act_new = QAction("New Painting…", self)
        self.act_new.setShortcut(QKeySequence.New)
        self.act_new.triggered.connect(self._new_painting)
        self.act_settings = QAction("Settings…", self)
        self.act_settings.triggered.connect(self._painting_settings)
        self.act_reference = QAction("Reference…", self)
        self.act_reference.triggered.connect(self._change_reference)
        for a in (self.act_new, self.act_settings, self.act_reference):
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" Palette "))
        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumWidth(170)
        self.palette_combo.currentTextChanged.connect(self._on_palette_chosen)
        tb.addWidget(self.palette_combo)
        self.act_edit_palette = QAction("Edit Palette", self)
        self.act_edit_palette.setCheckable(True)
        self.act_edit_palette.toggled.connect(self.palette_dock.setVisible)
        self.palette_dock.visibilityChanged.connect(self.act_edit_palette.setChecked)
        tb.addAction(self.act_edit_palette)
        tb.addSeparator()
        tb.addWidget(QLabel(" Pick "))
        self.pick_mode = QComboBox()
        self.pick_mode.addItem("Area — average", easel.KIND_REGION)
        self.pick_mode.addItem("Eyedropper — exact pixel", easel.KIND_PIXEL)
        self.pick_mode.addItem("Small spot — average", easel.KIND_SPOT)
        self.pick_mode.setToolTip(
            "Area: the average of the patch of similar colour you click.\n"
            "Eyedropper: the exact colour of the pixel under the cursor, no averaging.\n"
            "Small spot: the average of a small circle.\n"
            "Option-click uses the eyedropper from Area (and Area from the eyedropper). "
            "E switches between them.")
        self.pick_mode.currentIndexChanged.connect(self._on_pick_mode)
        tb.addWidget(self.pick_mode)
        tb.addWidget(QLabel(" Size "))
        self.region_bias = QSlider(Qt.Horizontal)
        self.region_bias.setRange(-easel.MAX_BIAS, easel.MAX_BIAS)
        self.region_bias.setValue(0)
        self.region_bias.setFixedWidth(110)
        self.region_bias.setToolTip("Region size: smaller ← auto → larger")
        tb.addWidget(self.region_bias)
        tb.addSeparator()
        self.act_clear = QAction("Clear Selection", self)
        self.act_clear.setShortcut(QKeySequence("Ctrl+D"))    # ⌘D on a Mac
        self.act_clear.setToolTip("Show the whole photo with no region selected "
                                  "(⌘D, Esc, or click the margin around the photo)")
        self.act_clear.triggered.connect(self._deselect)
        tb.addAction(self.act_clear)
        self.act_value = QAction("Value", self)
        self.act_value.setCheckable(True)
        self.act_value.setShortcut(QKeySequence("V"))
        self.act_value.setToolTip("Take the colour away and see value only (V)")
        self.act_value.toggled.connect(self.canvas.set_value_mode)
        tb.addAction(self.act_value)
        self.act_full = QAction("Full Screen", self)
        self.act_full.setShortcut(QKeySequence.FullScreen)
        self.act_full.triggered.connect(
            lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())
        tb.addAction(self.act_full)

        # Keys that don't need a button.
        eyedrop = QAction(self)
        eyedrop.setShortcut(QKeySequence("E"))
        eyedrop.triggered.connect(self._toggle_eyedropper)
        self.addAction(eyedrop)
        for keys, slot in (
            (("Space",), self._back_to_photo),
            (("Esc",), self._escape),
            (("Backspace", "Delete"), self._delete_current),
        ):
            act = QAction(self)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(slot)
            self.addAction(act)
        find = QAction(self)
        find.setShortcut(QKeySequence.Find)
        find.triggered.connect(lambda: (self.search.setFocus(), self.search.selectAll()))
        self.addAction(find)
        self._on_pick_mode()
        self.statusBar().showMessage("Click the reference to see a colour big, with how to mix it.")

    # -- palettes -------------------------------------------------------------------------
    def _reload_palettes(self, select: str = ""):
        self._syncing = True
        names = sorted(self.lib.palettes())
        self.palette_combo.clear()
        self.palette_combo.addItems(names)
        want = select or (self.project.palette if self.project else self.palette.name)
        if want in names:
            self.palette_combo.setCurrentText(want)
        self._syncing = False
        self.palette = palette_or_first(self.lib, self.palette_combo.currentText(), self.palette.medium)
        self.palette_panel.set_palette(self.palette.paints)
        self.palette_dock.setWindowTitle(f"Palette — {self.palette.name}")

    def _on_palette_chosen(self, name: str):
        if self._syncing or not name:
            return
        self.palette = palette_or_first(self.lib, name, self.palette.medium)
        self.palette_panel.set_palette(self.palette.paints)
        self.palette_dock.setWindowTitle(f"Palette — {self.palette.name}")
        if self.project is not None and self.project.palette != name:
            self.project.palette = name
            self.lib.save_project(self.project)
            self.statusBar().showMessage(
                f"{self.project.name} now mixes from {name}. Old swatches keep the recipe "
                "they were mixed with — use Re-mix to update one.")

    def _on_palette_edited(self, paints):
        self.palette = NamedPalette(self.palette.name, self.palette.medium, tuple(paints))
        self.lib.save_palette(self.palette)

    def _on_palette_starter(self):
        starter = STARTERS.get(self.palette.medium)
        if not starter:
            return
        self.palette = NamedPalette(self.palette.name, self.palette.medium, starter)
        self.lib.save_palette(self.palette)
        self.palette_panel.set_palette(starter)

    def _palette_save_as(self):
        name, ok = QInputDialog.getText(self, "Save palette", "Name for this palette:",
                                        text=f"{self.palette.name} (copy)")
        name = name.strip()
        if not ok or not name:
            return
        self.palette = NamedPalette(name, self.palette.medium, self.palette_panel.palette())
        self.lib.save_palette(self.palette)
        self._reload_palettes(select=name)
        self._on_palette_chosen(name)

    def _palette_make_default(self):
        self.lib.set_default_palette(self.palette.name)
        self.statusBar().showMessage(f"“{self.palette.name}” is now your palette.")

    def _palette_delete(self):
        pals = self.lib.palettes()
        if len(pals) <= 1:
            QMessageBox.information(self, "Delete palette", "Keep at least one palette.")
            return
        if QMessageBox.question(
                self, "Delete palette",
                f"Delete the palette “{self.palette.name}”? Swatches already mixed from it "
                "keep their recipes.") != QMessageBox.Yes:
            return
        self.lib.delete_palette(self.palette.name)
        self._reload_palettes()
        self._on_palette_chosen(self.palette_combo.currentText())

    # -- projects ---------------------------------------------------------------------------
    def _reload_projects(self, select: str = ""):
        self._syncing = True
        self.project_combo.clear()
        projects = self.lib.projects()
        for proj in projects:
            self.project_combo.addItem(proj.name, proj.id)
        self._syncing = False
        if not projects:
            self.project = None
            self.canvas.set_placeholder("Start a painting — click here, or New Painting… (⌘N)")
            self._refresh_history()
            return
        idx = self.project_combo.findData(select) if select else 0
        self._syncing = True
        self.project_combo.setCurrentIndex(max(0, idx))
        self._syncing = False
        self._open_project(self.project_combo.currentData())

    def _on_project_chosen(self, _i):
        if not self._syncing and self.project_combo.currentData():
            self._open_project(self.project_combo.currentData())

    def _open_project(self, pid: str):
        self.project = self.lib.load_project(pid)
        QSettings().setValue("easel/project", pid)
        self.current = None
        self._reload_palettes(select=self.project.palette)
        self._load_reference()
        self._refresh_history()
        self.setWindowTitle(f"Easel — {self.project.name}")

    def _load_reference(self):
        path = self.lib.reference_path(self.project) if self.project else None
        if not path:
            self.analysis = self.display_rgb = None
            self.canvas.set_placeholder("Add the reference photo — click here, or Reference…")
            return
        try:
            self.display_rgb = normalize(load(path), DISPLAY_EDGE).rgb
        except Exception as exc:  # noqa: BLE001 - a bad file must not take the window down
            self.analysis = self.display_rgb = None
            self.canvas.set_placeholder(f"Couldn't open the reference: {exc}")
            return
        self.analysis = easel.prepare(self.display_rgb)
        self.canvas.set_photo(self.display_rgb)

    def _new_painting(self):
        dlg = PaintingDialog(self, sorted(self.lib.palettes()),
                             default=self.lib.default_palette_name())
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        proj = self.lib.create_project(v["name"], v["palette"], None,
                                       v["canvas_w"], v["canvas_h"], v["units"])
        imported = False
        if v["reference"]:
            try:
                self.lib.set_reference(proj, v["reference"])
                imported = True
            except ValueError as exc:
                # The painting still exists — only the photo failed; say so and
                # let them pick another rather than losing the painting too.
                QMessageBox.warning(self, "Reference photo", str(exc))
        self._reload_projects(select=proj.id)
        if not imported:
            self._change_reference()

    def _painting_settings(self):
        if self.project is None:
            return self._new_painting()
        dlg = PaintingDialog(self, sorted(self.lib.palettes()), self.project)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        self.project.name = v["name"]
        self.project.canvas_w, self.project.canvas_h = v["canvas_w"], v["canvas_h"]
        self.project.units = v["units"]
        self.project.palette = v["palette"]
        self.lib.save_project(self.project)
        self._reload_projects(select=self.project.id)

    def _change_reference(self):
        if self.project is None:
            return self._new_painting()
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference photo", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.heic)")
        if not path:
            return
        if self.project.swatches and QMessageBox.question(
                self, "Replace reference",
                "This painting already has swatches. They'll keep their colours and recipes, "
                "but their outlines will point at the new photo. Replace it?") != QMessageBox.Yes:
            return
        try:
            self.lib.set_reference(self.project, path)
        except ValueError as exc:
            QMessageBox.warning(self, "Reference", str(exc))
            return
        self._load_reference()

    def _on_empty_clicked(self):
        if self.project is None:
            self._new_painting()
        elif self.analysis is None:
            self._change_reference()

    # -- picking ------------------------------------------------------------------------------
    def _solve(self, hex_: str):
        medium = self.palette.medium
        recipes = solve_targets([Target(hex=hex_)], self.palette.paints, medium,
                                easel.paints_per_mix(medium))
        return recipes[0] if recipes else None

    def _on_pick_mode(self, _i=None):
        kind = self.pick_mode.currentData()
        self.region_bias.setEnabled(kind == easel.KIND_REGION)   # Size only shapes an area
        self.canvas.set_loupe_mode(kind == easel.KIND_PIXEL)

    def _toggle_eyedropper(self):
        """E: flip between averaging an area and the exact-pixel eyedropper."""
        want = (easel.KIND_REGION if self.pick_mode.currentData() == easel.KIND_PIXEL
                else easel.KIND_PIXEL)
        self.pick_mode.setCurrentIndex(self.pick_mode.findData(want))

    def _kind_for_click(self, alt: bool) -> str:
        """The pick a click makes. Option swaps between the two main ones."""
        mode = self.pick_mode.currentData()
        if not alt:
            return mode
        return easel.KIND_REGION if mode == easel.KIND_PIXEL else easel.KIND_PIXEL

    def _mark(self, kind: str, x: float, y: float, mask) -> None:
        """Show a pick on the photo: a ring for an exact pixel, an outline for an area."""
        if kind == easel.KIND_PIXEL:
            self.canvas.set_marker(x, y)
        else:
            self.canvas.set_region(mask)

    def _on_picked(self, nx: float, ny: float, alt: bool):
        if self.project is None or self.analysis is None:
            return
        self._save_details(refresh=False)
        kind = self._kind_for_click(alt)
        if kind == easel.KIND_PIXEL:
            pk = easel.pick_pixel(self.analysis, self.display_rgb, nx, ny)
        elif kind == easel.KIND_SPOT:
            pk = easel.pick_spot(self.analysis, nx, ny)
        else:
            pk = easel.pick_region(self.analysis, nx, ny, bias=self.region_bias.value())
        bias = self.region_bias.value() if kind == easel.KIND_REGION else 0
        # Somewhere already in the history opens that swatch instead of adding a
        # copy — clicking the cheek twice shouldn't give two cheek swatches.
        existing = easel.find_same(self.analysis, pk, kind, bias, self.project.swatches)
        if existing is not None:
            self._mark(kind, existing.x, existing.y, pk.mask)
            self._refresh_history(select=existing.id)
            self._show(existing)
            where = "exact point" if kind == easel.KIND_PIXEL else "area"
            self.statusBar().showMessage(
                f"That {where} is already “{existing.label()}” — showing it. Change Size "
                "or the pick type if you want a different pick there.")
            return
        sw = Swatch(
            id=new_swatch_id(), hex=pk.hex, x=pk.x, y=pk.y, area_frac=pk.area_frac,
            recipe=self._solve(pk.hex), palette=self.palette.name, medium=self.palette.medium,
            pick=kind, tolerance=pk.tolerance, bias=bias,
            name=easel.next_swatch_name(s.name for s in self.project.swatches),
        )
        crop = easel.crop_around(self.display_rgb, pk.x, pk.y, pk.bbox)
        self.lib.add_swatch(self.project, sw, crop)
        self._mark(kind, pk.x, pk.y, pk.mask)
        self._refresh_history(select=sw.id)
        self._show(sw)

    def _show(self, sw: Swatch):
        if self.current and self.current != sw.id:
            self._save_details(refresh=False)     # notes have no editingFinished
        self.current = sw.id
        proj = self.project
        area = easel.canvas_cm2(proj.canvas_w, proj.canvas_h, proj.units)
        if sw.pick == easel.KIND_PIXEL:
            total, amount = 0.0, ("An exact point has no area to size an amount from — mix a "
                                  "little to test it, or pick the Area to see how much you need.")
        else:
            total, amount = easel.amount_summary(sw.medium, sw.area_frac, area)
        # Read names and traits from the palette this swatch was mixed with, if
        # it still exists — whiteness and staining decide the mixing order.
        pal = self.lib.palette(sw.palette) or self.palette
        names = {p.hex: p.name for p in pal.paints}
        recipe = sw.recipe
        L, v = easel.value_of(sw.hex)
        crop_path = self.lib.crop_path(proj, sw)
        how = {easel.KIND_PIXEL: "  ·  exact pixel", easel.KIND_SPOT: "  ·  small spot"}.get(sw.pick, "")
        card = SwatchCard(
            title=sw.label(),
            target_hex=sw.hex,
            mixed_hex=recipe.mixed_hex if recipe else sw.hex,
            delta_e=recipe.delta_e if recipe else 0.0,
            achievable=recipe.achievable if recipe else False,
            description=easel.describe(sw.hex).capitalize(),
            value_text=f"Value {v} of 10  (L* {L:.0f})  ·  {sw.hex}{how}",
            ratio_text=parts_text(recipe, names) if recipe else "No paints in this palette.",
            amount_text=amount,
            steps=easel.mixing_steps(recipe, pal.paints, sw.medium, total) if recipe else [],
            palette_name=sw.palette,
            crop=QImage(crop_path) if crop_path else None,
        )
        self._fill_details(sw)
        self.canvas.show_swatch(card)
        if sw.recipe and sw.palette != self.palette.name:
            self.statusBar().showMessage(
                f"Mixed with “{sw.palette}”. Re-mix to use “{self.palette.name}” instead.")

    def _escape(self):
        """Esc steps back one level: swatch → photo, then selection → none."""
        if self.canvas.mode == "swatch":
            self._back_to_photo()
        else:
            self._deselect()

    def _deselect(self):
        """Clear the selected region so the whole photo shows, undimmed."""
        if self.canvas.mode == "swatch":
            self._back_to_photo()
        self._save_details(refresh=False)
        self.current = None
        self.canvas.set_region(None)
        self.history.clearSelection()
        self.history.setCurrentRow(-1)
        self._fill_details(None)

    def _back_to_photo(self):
        if self.canvas.mode != "swatch":
            return
        sw = self.project.swatch(self.current) if (self.project and self.current) else None
        if sw is not None and self.analysis is not None:
            kind = sw.pick or easel.KIND_REGION
            mask = None if kind == easel.KIND_PIXEL else easel.saved_mask(self.analysis, sw)
            self._mark(kind, sw.x, sw.y, mask)
        self.canvas.show_photo()

    # -- history --------------------------------------------------------------------------------
    def _refresh_history(self, *_args, select: str = ""):
        self.history.clear()
        scope_all = self.scope.currentData() == "all"
        if not scope_all and self.project is None:
            return
        results = self.lib.search(self.search.text(), None if scope_all else self.project.id)
        target_row = -1
        for proj, sw in results:
            n = len(proj.swatches) - proj.swatches.index(sw)
            recipe = (" + ".join(s.paint_name for s in sw.recipe.steps) if sw.recipe else "")
            head = f"{proj.name} — " if scope_all else ""
            how = {easel.KIND_PIXEL: " · exact", easel.KIND_SPOT: " · spot"}.get(sw.pick, "")
            item = QListWidgetItem(self._icon(proj, sw),
                                   f"{head}{sw.label(n)}{how}\n{recipe}\n{sw.created[:10]}")
            item.setData(Qt.UserRole, (proj.id, sw.id))
            item.setToolTip(sw.notes or sw.hex)
            self.history.addItem(item)
            if sw.id == (select or self.current):
                target_row = self.history.count() - 1
        if target_row >= 0:
            self.history.setCurrentRow(target_row)

    def _icon(self, proj: Project, sw: Swatch) -> QIcon:
        pix = QPixmap(96, 48)
        pix.fill(QColor(sw.hex))
        crop = self.lib.crop_path(proj, sw)
        if crop:
            p = QPainter(pix)
            p.drawImage(QRectF(48, 0, 48, 48), QImage(crop))
            p.end()
        return QIcon(pix)

    def _on_history_clicked(self, item: QListWidgetItem):
        pid, sid = item.data(Qt.UserRole)
        if self.project is None or pid != self.project.id:
            idx = self.project_combo.findData(pid)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)   # opens that painting
        sw = self.project.swatch(sid) if self.project else None
        if sw is not None:
            self._show(sw)

    def _fill_details(self, sw: Optional[Swatch]):
        self._syncing = True
        self.name_edit.setText(sw.name if sw else "")
        self.tags_edit.setText(", ".join(sw.tags) if sw else "")
        self.notes_edit.setPlainText(sw.notes if sw else "")
        self._syncing = False

    def _current_swatch(self) -> Optional[Swatch]:
        return self.project.swatch(self.current) if (self.project and self.current) else None

    def _save_details(self, refresh: bool = True):
        sw = self._current_swatch()
        if self._syncing or sw is None:
            return
        name = self.name_edit.text().strip()
        tags = tuple(t.strip() for t in self.tags_edit.text().split(",") if t.strip())
        notes = self.notes_edit.toPlainText().strip()
        if (name, tags, notes) == (sw.name, sw.tags, sw.notes):
            return
        sw.name, sw.tags, sw.notes = name, tags, notes
        self.lib.save_project(self.project)
        if refresh:
            self._refresh_history(select=sw.id)

    def _remix(self):
        sw = self._current_swatch()
        if sw is None:
            return
        sw.recipe = self._solve(sw.hex)
        sw.palette, sw.medium = self.palette.name, self.palette.medium
        self.lib.save_project(self.project)
        self._refresh_history(select=sw.id)
        self._show(sw)
        self.statusBar().showMessage(f"Re-mixed with {self.palette.name}.")

    def _delete_current(self):
        sw = self._current_swatch()
        if sw is None:
            return
        if QMessageBox.question(self, "Delete swatch", f"Delete “{sw.label()}”?") != QMessageBox.Yes:
            return
        self.lib.delete_swatch(self.project, sw.id)
        self.current = None
        self._fill_details(None)
        self._refresh_history()
        self.canvas.set_region(None)
        self.canvas.show_photo()

    # -- shutdown -------------------------------------------------------------------------------
    def closeEvent(self, event):
        # Name/notes are saved on editingFinished; notes have no such signal, so
        # flush them here rather than lose a sentence typed just before closing.
        self._save_details()
        super().closeEvent(event)
