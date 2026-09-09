"""Right-hand control panel — every tunable parameter as a live widget.

The panel never holds the authoritative state; it reads its widgets into a
``Params`` via :meth:`collect`, and can be driven from a ``Params`` via
:meth:`apply` (used by Reset, presets, undo/redo and session restore).
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..media import SPECS
from ..pipeline.mosaic_grid import (
    LAYOUT_FLOW,
    LAYOUT_GRID,
    MAX_COLORS,
    MAX_TILES,
    MIN_COLORS,
    MIN_TILES,
)
from ..params import (
    CM_PER_INCH,
    DOC_BOARD,
    DOC_MOSAIC,
    DOC_PLAN,
    WATERCOLOR_DOCS,
    HARMONY_SCHEMES,
    PRESETS,
    SHADE_MODES,
    UNIT_CM,
    UNIT_IN,
    Params,
)


class FloatSlider(QWidget):
    changed = Signal()

    def __init__(self, lo, hi, value, decimals=3, suffix="", log=False):
        super().__init__()
        self._lo, self._hi, self._dec, self._suffix = lo, hi, decimals, suffix
        # Logarithmic mapping gives constant *relative* change per slider step,
        # so small-fraction controls stay finely adjustable at the low end.
        self._log = bool(log) and lo > 0 and hi > lo
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.value_lbl = QLabel()
        self.value_lbl.setMinimumWidth(52)
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value_lbl)
        self.setValue(value)
        self.slider.valueChanged.connect(self._on_slider)

    def _on_slider(self):
        self.value_lbl.setText(self._text())
        self.changed.emit()

    def _text(self):
        return f"{self.value():.{self._dec}f}{self._suffix}"

    def _to_value(self, t: float) -> float:
        if self._log:
            return self._lo * (self._hi / self._lo) ** t
        return self._lo + t * (self._hi - self._lo)

    def _to_t(self, v: float) -> float:
        if self._log:
            v = max(self._lo, min(self._hi, v))
            return math.log(v / self._lo) / math.log(self._hi / self._lo)
        span = self._hi - self._lo
        return 0.0 if span == 0 else (v - self._lo) / span

    def value(self) -> float:
        # Rounded to the decimals actually displayed, so the value read back is
        # the value shown — and so collect(apply(p)) == p exactly. Without this
        # the raw float drifts (0.2 -> 0.2001) and every session restore or undo
        # would report a spurious change.
        return round(self._to_value(self.slider.value() / 1000.0), self._dec)

    def setValue(self, v: float):
        t = self._to_t(v)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(max(0.0, min(1.0, t)) * 1000)))
        self.slider.blockSignals(False)
        self.value_lbl.setText(self._text())


class ControlsPanel(QWidget):
    changed = Signal()
    preset_requested = Signal(str)
    preset_save_requested = Signal()
    preset_delete_requested = Signal(str)
    add_color_requested = Signal()    # "Add color…" — manual target picker
    clear_picked_requested = Signal()  # drop all manual/eyedrop targets

    def __init__(self, params: Params):
        super().__init__()
        self._loading = False
        self._anchor_hex = params.anchor_hex
        self._card_unit = params.canvas_units
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # --- Preset row ------------------------------------------------------
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(QLabel("Preset"))
        self.preset = QComboBox()
        self.preset.addItem("Custom")
        for _name in PRESETS:
            self.preset.addItem(_name)
        preset_row.addWidget(self.preset, 1)
        self.preset_save_btn = QPushButton("Save…")
        self.preset_save_btn.setToolTip("Save the current board/solver settings as a named preset.")
        self.preset_del_btn = QPushButton("Del")
        self.preset_del_btn.setToolTip("Delete the selected user preset.")
        preset_row.addWidget(self.preset_save_btn)
        preset_row.addWidget(self.preset_del_btn)
        prow = QWidget()
        prow.setLayout(preset_row)
        root.addWidget(prow)

        # --- Document type -------------------------------------------------------
        g_doc = QGroupBox("Document")
        f = self._form(g_doc)
        self.doc_mode = QComboBox()
        self.doc_mode.addItem("Palette board", DOC_BOARD)
        self.doc_mode.addItem("Watercolor painting plan", DOC_PLAN)
        self.doc_mode.addItem("Mosaic guide", DOC_MOSAIC)
        self._set_combo_data(self.doc_mode, params.doc_mode)
        self.doc_mode.setToolTip(
            "Palette board: a reference sheet of your colours.\n"
            "Painting plan: numbered washes for the photo you loaded.\n"
            "Mosaic guide: the photo as tesserae, painted square by square."
        )
        f.addRow("Type", self.doc_mode)
        root.addWidget(g_doc)

        # --- Medium ------------------------------------------------------------
        g_medium = QGroupBox("Medium")
        f = self._form(g_medium)
        self.medium = QComboBox()
        for key, sp in SPECS.items():
            self.medium.addItem(sp.label, key)
        self._set_combo_data(self.medium, params.medium)
        self.medium_rule = QLabel()
        self.medium_rule.setWordWrap(True)
        self.medium_rule.setStyleSheet("color: #888; font-size: 11px;")
        f.addRow("Medium", self.medium)
        f.addRow(self.medium_rule)
        root.addWidget(g_medium)
        self._update_medium_rule()

        # --- Painting plan (plan mode only) ----------------------------------------
        g_plan = QGroupBox("Painting plan")
        f = self._form(g_plan)
        self.wash_tiers = QSpinBox()
        self.wash_tiers.setRange(3, 6)
        self.wash_tiers.setValue(params.wash_tiers)
        self.wash_tiers.setToolTip("How many value layers the painting is built from.")
        self.plan_colors = QSpinBox()
        self.plan_colors.setRange(3, 12)
        self.plan_colors.setValue(params.plan_colors)
        self.plan_colors.setToolTip("How finely the photo is cut into colour areas.")
        self.min_passage = QDoubleSpinBox()
        self.min_passage.setRange(0.2, 8.0)
        self.min_passage.setDecimals(1)
        self.min_passage.setSingleStep(0.5)
        self.min_passage.setSuffix(" %")
        self.min_passage.setValue(params.min_passage_pct)
        self.min_passage.setToolTip(
            "Shapes smaller than this share of the picture get folded into their neighbour."
        )
        paper_row = QHBoxLayout()
        self.paper_w = QDoubleSpinBox()
        self.paper_h = QDoubleSpinBox()
        for sb, v in ((self.paper_w, params.paper_w), (self.paper_h, params.paper_h)):
            sb.setRange(0.0, 200.0)
            sb.setDecimals(1)
            sb.setSingleStep(0.5)
            sb.setValue(v)
            sb.setToolTip("Your paper size — this is what turns area into millilitres of wash.")
            paper_row.addWidget(sb)
        pprow = QWidget()
        pprow.setLayout(paper_row)
        self.show_plan_map = QCheckBox("Map & value plan")
        self.show_plan_passes = QCheckBox("The washes")
        self.show_plan_tips = QCheckBox("Test strip & glossary")
        for cb, v in (
            (self.show_plan_map, params.show_plan_map),
            (self.show_plan_passes, params.show_plan_passes),
            (self.show_plan_tips, params.show_plan_tips),
        ):
            cb.setChecked(v)
        f.addRow("Value tiers", self.wash_tiers)
        f.addRow("Colour areas", self.plan_colors)
        f.addRow("Smallest shape", self.min_passage)
        f.addRow("Paper W × H", pprow)
        for cb in (self.show_plan_map, self.show_plan_passes, self.show_plan_tips):
            f.addRow(cb)
        root.addWidget(g_plan)

        # --- Mosaic guide (mosaic mode only) ----------------------------------------
        g_mosaic = QGroupBox("Mosaic")
        f = self._form(g_mosaic)
        self.tessera_layout = QComboBox()
        self.tessera_layout.addItem("Courses follow the drawing", LAYOUT_FLOW)
        self.tessera_layout.addItem("Square grid", LAYOUT_GRID)
        self._set_combo_data(self.tessera_layout, params.tessera_layout)
        self.tessera_layout.setToolTip(
            "Courses: tesserae are laid along the picture's contours, the way a "
            "mosaicist works — the shapes read even with the colour taken away.\n"
            "Square grid: a plain lattice you can rule onto paper by row and column."
        )
        self.tiles_x = QSpinBox()
        self.tiles_x.setRange(MIN_TILES, MAX_TILES)
        self.tiles_x.setValue(params.tiles_x)
        self.tiles_x.setToolTip("Tesserae across. Rows follow the photo's shape.")
        self.mosaic_colors = QSpinBox()
        self.mosaic_colors.setRange(MIN_COLORS, MAX_COLORS)
        self.mosaic_colors.setValue(params.mosaic_colors)
        self.mosaic_colors.setToolTip("How many colours the whole mosaic is reduced to.")
        self.tessera_mm = QDoubleSpinBox()
        self.tessera_mm.setRange(3.0, 60.0)
        self.tessera_mm.setDecimals(1)
        self.tessera_mm.setSingleStep(1.0)
        self.tessera_mm.setSuffix(" mm")
        self.tessera_mm.setValue(params.tessera_mm)
        self.tessera_mm.setToolTip(
            "Painted size of one square. This is what the to-scale grid prints at."
        )
        self.grout_frac = FloatSlider(0.0, 0.30, params.grout_frac, decimals=2)
        self.grout_frac.setToolTip("Gap of bare paper left between squares.")
        self.tessera_jitter = FloatSlider(0.0, 1.0, params.tessera_jitter, decimals=2)
        self.tessera_jitter.setToolTip(
            "How hand-cut the tesserae look. Measured against the grout, so a wider "
            "grout allows more irregularity — squares can never overlap."
        )
        self.show_mosaic_preview = QCheckBox("Photo & mosaic")
        self.show_mosaic_grid = QCheckBox("Numbered grid")
        self.show_mosaic_legend = QCheckBox("Colour legend")
        for cb, v in (
            (self.show_mosaic_preview, params.show_mosaic_preview),
            (self.show_mosaic_grid, params.show_mosaic_grid),
            (self.show_mosaic_legend, params.show_mosaic_legend),
        ):
            cb.setChecked(v)
        f.addRow("Layout", self.tessera_layout)
        f.addRow("Tesserae across", self.tiles_x)
        f.addRow("Colours", self.mosaic_colors)
        f.addRow("Tessera size", self.tessera_mm)
        f.addRow("Grout", self.grout_frac)
        f.addRow("Irregularity", self.tessera_jitter)
        for cb in (self.show_mosaic_preview, self.show_mosaic_grid, self.show_mosaic_legend):
            f.addRow(cb)
        root.addWidget(g_mosaic)

        # --- Targets --------------------------------------------------------------
        g_targets = QGroupBox("Target colors")
        f = self._form(g_targets)
        self.extract_enabled = QCheckBox("Auto palette from photo")
        self.extract_enabled.setChecked(params.extract_enabled)
        self.extract_n = QSpinBox()
        self.extract_n.setRange(3, 12)
        self.extract_n.setValue(params.extract_n)
        btn_row = QHBoxLayout()
        self.add_color_btn = QPushButton("Add color…")
        self.add_color_btn.setToolTip("Pick a target color to mix toward.")
        self.clear_picked_btn = QPushButton("Clear picked")
        self.clear_picked_btn.setToolTip("Remove all hand-picked and eyedropped targets.")
        btn_row.addWidget(self.add_color_btn)
        btn_row.addWidget(self.clear_picked_btn)
        brow = QWidget()
        brow.setLayout(btn_row)
        f.addRow(self.extract_enabled)
        f.addRow("Colors from photo", self.extract_n)
        f.addRow(brow)
        root.addWidget(g_targets)

        # --- Mixing ------------------------------------------------------------------
        g_mix = QGroupBox("Mixing")
        f = self._form(g_mix)
        self.max_paints = QComboBox()
        self.max_paints.addItem("2 paints (cleaner)", 2)
        self.max_paints.addItem("3 paints (closer)", 3)
        self._set_combo_data(self.max_paints, params.max_paints)
        f.addRow("Paints per mix", self.max_paints)
        root.addWidget(g_mix)

        # --- Value ladder ----------------------------------------------------------------
        g_ladder = QGroupBox("Value ladder (shading)")
        f = self._form(g_ladder)
        self.ladder_steps = QSpinBox()
        self.ladder_steps.setRange(5, 7)
        self.ladder_steps.setValue(params.ladder_steps)
        self.shade_mode = QComboBox()
        for mode in SHADE_MODES:
            self.shade_mode.addItem(mode.capitalize(), mode)
        self._set_combo_data(self.shade_mode, params.shade_mode)
        self.shade_mode.setToolTip(
            "What darkens the shadow steps: Auto picks a transparent dark or the "
            "complement; Black only if you insist."
        )
        f.addRow("Steps", self.ladder_steps)
        f.addRow("Darken with", self.shade_mode)
        root.addWidget(g_ladder)

        # --- Wheel & harmony -----------------------------------------------------------------
        g_wheel = QGroupBox("Wheel & harmony")
        f = self._form(g_wheel)
        self.harmony = QComboBox()
        for scheme in HARMONY_SCHEMES:
            self.harmony.addItem(scheme.capitalize(), scheme)
        self._set_combo_data(self.harmony, params.harmony_scheme)
        anchor_row = QHBoxLayout()
        self.anchor_btn = QPushButton()
        self.anchor_btn.setFixedSize(34, 20)
        self.anchor_btn.setToolTip("Harmony anchor color — click to pick; Auto follows the first target.")
        self.anchor_auto_btn = QPushButton("Auto")
        self.anchor_auto_btn.setToolTip("Anchor the harmony on the first target color.")
        anchor_row.addWidget(self.anchor_btn)
        anchor_row.addWidget(self.anchor_auto_btn)
        anchor_row.addStretch(1)
        arow = QWidget()
        arow.setLayout(anchor_row)
        self.show_gamut = QCheckBox("Show mixable gamut")
        self.show_gamut.setChecked(params.show_gamut)
        f.addRow("Scheme", self.harmony)
        f.addRow("Anchor", arow)
        f.addRow(self.show_gamut)
        root.addWidget(g_wheel)
        self._paint_anchor()

        # --- Board sections ------------------------------------------------------------------
        g_board = QGroupBox("Board")
        f = self._form(g_board)
        self.show_photo = QCheckBox("Reference photo")
        self.show_palette = QCheckBox("Palette swatches")
        self.show_wheel = QCheckBox("Color wheel")
        self.show_value_strip = QCheckBox("Value scale")
        self.show_recipes = QCheckBox("Mixing recipes")
        self.show_ladders = QCheckBox("Value ladders")
        self.show_notes = QCheckBox("Technique notes")
        for cb, v in (
            (self.show_photo, params.show_photo),
            (self.show_palette, params.show_palette),
            (self.show_wheel, params.show_wheel),
            (self.show_value_strip, params.show_value_strip),
            (self.show_recipes, params.show_recipes),
            (self.show_ladders, params.show_ladders),
            (self.show_notes, params.show_notes),
        ):
            cb.setChecked(v)
            f.addRow(cb)
        self.board_title = QLineEdit(params.board_title)
        self.board_title.setPlaceholderText("Board title (optional)")
        f.addRow("Title", self.board_title)
        root.addWidget(g_board)

        # --- Card size (print scale) --------------------------------------------------------------
        g_card = QGroupBox("Card size (print)")
        f = self._form(g_card)
        self.card_w = QDoubleSpinBox()
        self.card_w.setRange(0.0, 120.0)
        self.card_w.setDecimals(2)
        self.card_w.setSingleStep(0.5)
        self.card_w.setSpecialValueText("(not set)")
        self.card_w.setValue(params.card_w)
        self.card_units = QComboBox()
        self.card_units.addItem("inches", UNIT_IN)
        self.card_units.addItem("centimetres", UNIT_CM)
        self._set_combo_data(self.card_units, params.canvas_units)
        self.card_w.setToolTip("Physical width of the printed board — sets the SVG/PDF scale.")
        f.addRow("Width", self.card_w)
        f.addRow("Units", self.card_units)
        root.addWidget(g_card)

        # The two documents want different controls; keep both built and just
        # show the relevant set, so collect()/apply() stay mode-agnostic.
        self._g_plan = g_plan
        self._g_mosaic = g_mosaic
        self._g_medium = g_medium
        self._g_targets = g_targets
        self._g_ladder = g_ladder
        self._g_wheel = g_wheel
        self._g_board = g_board

        root.addStretch(1)
        self._wire()
        self._sync_mode()

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _set_combo_data(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _form(self, box):
        f = QFormLayout(box)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return f

    def _paint_anchor(self):
        if self._anchor_hex:
            self.anchor_btn.setStyleSheet(
                f"QPushButton {{ background-color: {self._anchor_hex}; border: 1px solid #555; }}"
            )
            self.anchor_btn.setText("")
        else:
            self.anchor_btn.setStyleSheet("")
            self.anchor_btn.setText("auto")

    def _update_medium_rule(self):
        mode = self.doc_mode.currentData()
        if mode in WATERCOLOR_DOCS:
            what = "painting plan" if mode == DOC_PLAN else "mosaic guide"
            self.medium_rule.setText(
                f"The {what} is watercolor — the paper is your white, and everything "
                "lightens by dilution."
            )
        else:
            sp = SPECS[self.medium.currentData()]
            self.medium_rule.setText(f"{sp.lighten} {sp.darken}")

    def _sync_mode(self):
        """Show only the controls the current document actually uses."""
        mode = self.doc_mode.currentData()
        self._g_plan.setVisible(mode == DOC_PLAN)
        self._g_mosaic.setVisible(mode == DOC_MOSAIC)
        # The two watercolor documents pick the medium for you.
        self.medium.setEnabled(mode not in WATERCOLOR_DOCS)
        for box in (self._g_targets, self._g_ladder, self._g_wheel, self._g_board):
            box.setVisible(mode == DOC_BOARD)
        self._update_medium_rule()

    # -- signal wiring ------------------------------------------------------------
    def _wire(self):
        for combo in (self.doc_mode, self.medium, self.max_paints, self.shade_mode,
                      self.harmony, self.card_units, self.tessera_layout):
            combo.currentIndexChanged.connect(self._emit)
        self.medium.currentIndexChanged.connect(self._update_medium_rule)
        self.doc_mode.currentIndexChanged.connect(self._sync_mode)
        self.card_units.currentIndexChanged.connect(self._on_units_changed)
        for sb in (self.extract_n, self.ladder_steps, self.wash_tiers, self.plan_colors,
                   self.tiles_x, self.mosaic_colors):
            sb.valueChanged.connect(self._emit)
        for sb in (self.card_w, self.min_passage, self.paper_w, self.paper_h,
                   self.tessera_mm):
            sb.valueChanged.connect(self._emit)
        for sl in (self.grout_frac, self.tessera_jitter):
            sl.changed.connect(self._emit)
        for cb in (
            self.extract_enabled, self.show_gamut, self.show_photo, self.show_palette,
            self.show_wheel, self.show_value_strip, self.show_recipes, self.show_ladders,
            self.show_notes, self.show_plan_map, self.show_plan_passes, self.show_plan_tips,
            self.show_mosaic_preview, self.show_mosaic_grid, self.show_mosaic_legend,
        ):
            cb.toggled.connect(self._emit)
        self.board_title.textEdited.connect(self._emit)
        self.anchor_btn.clicked.connect(self._on_pick_anchor)
        self.anchor_auto_btn.clicked.connect(self._on_anchor_auto)
        self.preset.currentTextChanged.connect(self._on_preset)
        self.preset_save_btn.clicked.connect(lambda: self.preset_save_requested.emit())
        self.preset_del_btn.clicked.connect(
            lambda: self.preset_delete_requested.emit(self.preset.currentText()))
        self.add_color_btn.clicked.connect(lambda: self.add_color_requested.emit())
        self.clear_picked_btn.clicked.connect(lambda: self.clear_picked_requested.emit())

    def _emit(self):
        if not self._loading:
            self.changed.emit()

    def _on_preset(self, name: str):
        if self._loading:
            return
        if name and name != "Custom":
            self.preset_requested.emit(name)

    def _on_pick_anchor(self):
        color = QColorDialog.getColor(parent=self, title="Harmony anchor color")
        if color.isValid():
            self._anchor_hex = color.name()
            self._paint_anchor()
            self._emit()

    def _on_anchor_auto(self):
        if self._anchor_hex:
            self._anchor_hex = ""
            self._paint_anchor()
            self._emit()

    def _on_units_changed(self):
        """Converting units keeps the physical size (5 in -> 12.7 cm)."""
        if self._loading:
            return
        new = self.card_units.currentData()
        if new == self._card_unit:
            return
        # Card width and paper size are both in these units, so both convert.
        for sb in (self.card_w, self.paper_w, self.paper_h):
            v = sb.value()
            if v > 0:
                v = v * CM_PER_INCH if new == UNIT_CM else v / CM_PER_INCH
                sb.blockSignals(True)
                sb.setValue(v)
                sb.blockSignals(False)
        self._card_unit = new
        # _emit already fired from currentIndexChanged.

    def set_user_presets(self, names):
        """Rebuild the preset combo: Custom, built-ins, then saved user presets."""
        self._loading = True
        cur = self.preset.currentText()
        self.preset.clear()
        self.preset.addItem("Custom")
        for n in PRESETS:
            self.preset.addItem(n)
        if names:
            self.preset.insertSeparator(self.preset.count())
            for n in names:
                self.preset.addItem(n)
        idx = self.preset.findText(cur)
        self.preset.setCurrentIndex(idx if idx >= 0 else 0)
        self._loading = False

    # -- params <-> widgets --------------------------------------------------------
    def collect(self, base: Params) -> Params:
        return base.copy(
            doc_mode=self.doc_mode.currentData(),
            medium=self.medium.currentData(),
            wash_tiers=int(self.wash_tiers.value()),
            plan_colors=int(self.plan_colors.value()),
            min_passage_pct=float(self.min_passage.value()),
            paper_w=float(self.paper_w.value()),
            paper_h=float(self.paper_h.value()),
            show_plan_map=self.show_plan_map.isChecked(),
            show_plan_passes=self.show_plan_passes.isChecked(),
            show_plan_tips=self.show_plan_tips.isChecked(),
            tessera_layout=self.tessera_layout.currentData(),
            tiles_x=int(self.tiles_x.value()),
            mosaic_colors=int(self.mosaic_colors.value()),
            tessera_mm=float(self.tessera_mm.value()),
            grout_frac=self.grout_frac.value(),
            tessera_jitter=self.tessera_jitter.value(),
            show_mosaic_preview=self.show_mosaic_preview.isChecked(),
            show_mosaic_grid=self.show_mosaic_grid.isChecked(),
            show_mosaic_legend=self.show_mosaic_legend.isChecked(),
            extract_enabled=self.extract_enabled.isChecked(),
            extract_n=int(self.extract_n.value()),
            max_paints=int(self.max_paints.currentData()),
            ladder_steps=int(self.ladder_steps.value()),
            shade_mode=self.shade_mode.currentData(),
            harmony_scheme=self.harmony.currentData(),
            anchor_hex=self._anchor_hex,
            show_gamut=self.show_gamut.isChecked(),
            show_photo=self.show_photo.isChecked(),
            show_palette=self.show_palette.isChecked(),
            show_wheel=self.show_wheel.isChecked(),
            show_value_strip=self.show_value_strip.isChecked(),
            show_recipes=self.show_recipes.isChecked(),
            show_ladders=self.show_ladders.isChecked(),
            show_notes=self.show_notes.isChecked(),
            board_title=self.board_title.text(),
            card_w=float(self.card_w.value()),
            canvas_units=self.card_units.currentData(),
        )

    def apply(self, p: Params):
        self._loading = True
        self._set_combo_data(self.doc_mode, p.doc_mode)
        self._set_combo_data(self.medium, p.medium)
        self.wash_tiers.setValue(p.wash_tiers)
        self.plan_colors.setValue(p.plan_colors)
        self.min_passage.setValue(p.min_passage_pct)
        self.paper_w.setValue(p.paper_w)
        self.paper_h.setValue(p.paper_h)
        self.show_plan_map.setChecked(p.show_plan_map)
        self.show_plan_passes.setChecked(p.show_plan_passes)
        self.show_plan_tips.setChecked(p.show_plan_tips)
        self._set_combo_data(self.tessera_layout, p.tessera_layout)
        self.tiles_x.setValue(p.tiles_x)
        self.mosaic_colors.setValue(p.mosaic_colors)
        self.tessera_mm.setValue(p.tessera_mm)
        self.grout_frac.setValue(p.grout_frac)
        self.tessera_jitter.setValue(p.tessera_jitter)
        self.show_mosaic_preview.setChecked(p.show_mosaic_preview)
        self.show_mosaic_grid.setChecked(p.show_mosaic_grid)
        self.show_mosaic_legend.setChecked(p.show_mosaic_legend)
        self._sync_mode()
        self.extract_enabled.setChecked(p.extract_enabled)
        self.extract_n.setValue(p.extract_n)
        self._set_combo_data(self.max_paints, p.max_paints)
        self.ladder_steps.setValue(p.ladder_steps)
        self._set_combo_data(self.shade_mode, p.shade_mode)
        self._set_combo_data(self.harmony, p.harmony_scheme)
        self._anchor_hex = p.anchor_hex
        self._paint_anchor()
        self.show_gamut.setChecked(p.show_gamut)
        self.show_photo.setChecked(p.show_photo)
        self.show_palette.setChecked(p.show_palette)
        self.show_wheel.setChecked(p.show_wheel)
        self.show_value_strip.setChecked(p.show_value_strip)
        self.show_recipes.setChecked(p.show_recipes)
        self.show_ladders.setChecked(p.show_ladders)
        self.show_notes.setChecked(p.show_notes)
        self.board_title.setText(p.board_title)
        self.card_w.setValue(p.card_w)
        self._set_combo_data(self.card_units, p.canvas_units)
        self._card_unit = p.canvas_units
        self._loading = False
