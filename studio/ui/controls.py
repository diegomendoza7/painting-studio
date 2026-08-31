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
from ..params import (
    CM_PER_INCH,
    DOC_BOARD,
    DOC_PLAN,
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
        return self._to_value(self.slider.value() / 1000.0)

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
        self._set_combo_data(self.doc_mode, params.doc_mode)
        self.doc_mode.setToolTip(
            "Palette board: a reference sheet of your colours.\n"
            "Painting plan: numbered washes for the photo you loaded."
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
        if self.doc_mode.currentData() == DOC_PLAN:
            sp = SPECS["watercolor"]
            self.medium_rule.setText(
                "The painting plan is watercolor — the paper is your white, and "
                "everything lightens by dilution."
            )
        else:
            sp = SPECS[self.medium.currentData()]
            self.medium_rule.setText(f"{sp.lighten} {sp.darken}")

    def _sync_mode(self):
        """Show only the controls the current document actually uses."""
        plan = self.doc_mode.currentData() == DOC_PLAN
        self._g_plan.setVisible(plan)
        self.medium.setEnabled(not plan)      # plan mode is always watercolor
        for box in (self._g_targets, self._g_ladder, self._g_wheel, self._g_board):
            box.setVisible(not plan)
        self._update_medium_rule()

    # -- signal wiring ------------------------------------------------------------
    def _wire(self):
        for combo in (self.doc_mode, self.medium, self.max_paints, self.shade_mode,
                      self.harmony, self.card_units):
            combo.currentIndexChanged.connect(self._emit)
        self.medium.currentIndexChanged.connect(self._update_medium_rule)
        self.doc_mode.currentIndexChanged.connect(self._sync_mode)
        self.card_units.currentIndexChanged.connect(self._on_units_changed)
        for sb in (self.extract_n, self.ladder_steps, self.wash_tiers, self.plan_colors):
            sb.valueChanged.connect(self._emit)
        for sb in (self.card_w, self.min_passage, self.paper_w, self.paper_h):
            sb.valueChanged.connect(self._emit)
        for cb in (
            self.extract_enabled, self.show_gamut, self.show_photo, self.show_palette,
            self.show_wheel, self.show_value_strip, self.show_recipes, self.show_ladders,
            self.show_notes, self.show_plan_map, self.show_plan_passes, self.show_plan_tips,
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
