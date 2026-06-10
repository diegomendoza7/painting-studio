"""Left panel — the artist's available paints, editable in place.

One row per paint: a color swatch button (click to repick the hex), the name,
a trait menu (white / opacity class / staining / granulating) and a remove
button. The panel never owns authoritative state: it renders a
``tuple[Paint, ...]`` via :meth:`set_palette` and reports every edit through
the ``palette_changed`` signal, which MainWindow routes into ``Params`` (the
same pattern bargue uses for its per-layer signals).
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..models import OPACITY_OPAQUE, OPACITY_SEMI, OPACITY_TRANSPARENT, Paint


def _swatch_css(hex_: str) -> str:
    return (
        f"QPushButton {{ background-color: {hex_}; border: 1px solid #555; "
        f"border-radius: 3px; }}"
    )


class PalettePanel(QWidget):
    palette_changed = Signal(object)   # tuple[Paint, ...]
    starter_requested = Signal()       # "Load starter set" for the current medium

    def __init__(self):
        super().__init__()
        self._palette: List[Paint] = []
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        header = QLabel("Palette — your paints")
        header.setStyleSheet("font-weight: bold;")
        root.addWidget(header)

        self._rows_host = QWidget()
        self._rows_lay = QVBoxLayout(self._rows_host)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(3)
        self._rows_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll, 1)

        btns = QHBoxLayout()
        self.add_btn = QPushButton("Add paint…")
        self.starter_btn = QPushButton("Starter set")
        self.starter_btn.setToolTip("Replace the palette with the current medium's starter set.")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.starter_btn)
        root.addLayout(btns)

        self.add_btn.clicked.connect(self._on_add)
        self.starter_btn.clicked.connect(self.starter_requested)

    # -- public API ----------------------------------------------------------
    def set_palette(self, palette) -> None:
        """Rebuild the rows from a tuple of Paints (guarded; no signal)."""
        self._loading = True
        self._palette = list(palette)
        # Drop old rows (last item is the stretch).
        while self._rows_lay.count() > 1:
            item = self._rows_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, paint in enumerate(self._palette):
            self._rows_lay.insertWidget(i, self._make_row(i, paint))
        self._loading = False

    def palette(self) -> tuple:
        return tuple(self._palette)

    # -- row construction -------------------------------------------------------
    def _make_row(self, i: int, paint: Paint) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        swatch = QPushButton()
        swatch.setFixedSize(30, 22)
        swatch.setStyleSheet(_swatch_css(paint.hex))
        swatch.setToolTip(f"{paint.hex} — click to change")
        swatch.clicked.connect(lambda _=False, k=i: self._on_pick_color(k))

        name = QLineEdit(paint.name)
        name.setPlaceholderText("Paint name")
        name.editingFinished.connect(lambda k=i, w=name: self._on_rename(k, w.text()))

        traits = QToolButton()
        traits.setText("⋯")
        traits.setToolTip("Traits: white / opacity / staining / granulating")
        traits.setPopupMode(QToolButton.InstantPopup)
        traits.setMenu(self._traits_menu(i, paint, traits))

        remove = QToolButton()
        remove.setText("✕")
        remove.setToolTip("Remove this paint")
        remove.clicked.connect(lambda _=False, k=i: self._on_remove(k))

        lay.addWidget(swatch)
        lay.addWidget(name, 1)
        lay.addWidget(traits)
        lay.addWidget(remove)
        return row

    def _traits_menu(self, i: int, paint: Paint, parent) -> QMenu:
        menu = QMenu(parent)
        white = QAction("Palette white", menu, checkable=True, checked=paint.is_white)
        white.toggled.connect(lambda v, k=i: self._on_trait(k, is_white=v))
        menu.addAction(white)
        menu.addSeparator()
        group = QActionGroup(menu)
        group.setExclusive(True)
        for label, value in (
            ("Opacity unknown", ""),
            ("Transparent", OPACITY_TRANSPARENT),
            ("Semi-opaque", OPACITY_SEMI),
            ("Opaque", OPACITY_OPAQUE),
        ):
            act = QAction(label, menu, checkable=True, checked=paint.opacity == value)
            act.triggered.connect(lambda _=False, k=i, v=value: self._on_trait(k, opacity=v))
            group.addAction(act)
            menu.addAction(act)
        menu.addSeparator()
        staining = QAction("Staining", menu, checkable=True, checked=paint.staining)
        staining.toggled.connect(lambda v, k=i: self._on_trait(k, staining=v))
        menu.addAction(staining)
        gran = QAction("Granulating", menu, checkable=True, checked=paint.granulating)
        gran.toggled.connect(lambda v, k=i: self._on_trait(k, granulating=v))
        menu.addAction(gran)
        return menu

    # -- edits ----------------------------------------------------------------
    def _emit(self):
        if not self._loading:
            self.palette_changed.emit(tuple(self._palette))

    def _replace(self, i: int, **changes):
        if self._loading or not 0 <= i < len(self._palette):
            return
        p = self._palette[i]
        self._palette[i] = Paint(
            name=changes.get("name", p.name),
            hex=changes.get("hex", p.hex),
            is_white=changes.get("is_white", p.is_white),
            opacity=changes.get("opacity", p.opacity),
            staining=changes.get("staining", p.staining),
            granulating=changes.get("granulating", p.granulating),
        )
        self._emit()

    def _on_pick_color(self, i: int):
        if not 0 <= i < len(self._palette):
            return
        color = QColorDialog.getColor(parent=self, title="Paint color")
        if color.isValid():
            self._replace(i, hex=color.name())
            self.set_palette(self._palette)  # refresh the swatch

    def _on_rename(self, i: int, text: str):
        text = text.strip()
        if text and 0 <= i < len(self._palette) and text != self._palette[i].name:
            self._replace(i, name=text)

    def _on_trait(self, i: int, **changes):
        self._replace(i, **changes)

    def _on_remove(self, i: int):
        if self._loading or not 0 <= i < len(self._palette):
            return
        del self._palette[i]
        self.set_palette(self._palette)
        self.palette_changed.emit(tuple(self._palette))

    def _on_add(self):
        color = QColorDialog.getColor(parent=self, title="New paint color")
        if not color.isValid():
            return
        self._palette.append(Paint(name=f"Paint {len(self._palette) + 1}", hex=color.name()))
        self.set_palette(self._palette)
        self.palette_changed.emit(tuple(self._palette))
