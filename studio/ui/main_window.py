"""The main application window: assembles panels, worker thread and exports."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..models import Target
from ..palettes import STARTERS
from ..params import (
    CM_PER_INCH,
    DOC_MOSAIC,
    DOC_PLAN,
    PRESETS,
    UNIT_CM,
    Params,
    apply_preset,
    apply_preset_values,
    preset_to_dict,
)
from ..pipeline.core import Result
from ..pipeline.loader import count_pdf_pages, is_pdf
from ..pipeline.mosaic_doc import build_grid_sheet
from ..session import decode_session, encode_session
from ..svg import scene_to_svg
from .controls import ControlsPanel
from .palette_panel import PalettePanel
from .pdf_print import write_scale_pdf as write_scale_pdf_vector, write_tiled_pdf
from .preview import PreviewWidget
from .render import ndarray_to_qimage, render_scene
from .upload import UploadPanel
from .worker import RenderRequest, Worker

DEBOUNCE_MS = 150
HISTORY_MAX = 100          # undo depth (Params snapshots are tiny + immutable)
HISTORY_COALESCE_S = 0.35  # collapse a rapid burst (slider drag) into one undo entry


def _in_rect(x: float, y: float, rect) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


class MainWindow(QMainWindow):
    request_load = Signal(str, int, int)   # path, page, dpi
    request_process = Signal(object)       # RenderRequest

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Painting Studio")
        self.resize(1320, 860)

        self.params = Params()
        self._path: Optional[str] = None
        self._busy = False
        self._pending = False
        self._result: Optional[Result] = None
        # Undo/redo of parameter edits. Each entry is one immutable Params;
        # _push_history records the pre-edit state at the start of each gesture.
        self._undo: list = []
        self._redo: list = []
        self._hist_t = 0.0
        self._user_presets: dict = {}      # name -> saved PRESET_FIELDS dict (QSettings)

        self._build_ui()
        self._load_user_presets()
        self._start_worker()
        self._restore_session()
        self._render_now()                 # the board exists even with no photo

    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        self.upload = UploadPanel()
        self.palette_panel = PalettePanel()
        self.palette_panel.set_palette(self.params.palette)
        self.preview = PreviewWidget()
        self.preview.clear("Building the palette board…")
        self.controls = ControlsPanel(self.params)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(self.controls)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(320)
        controls_scroll.setMaximumWidth(380)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(self.upload)
        left_lay.addWidget(self.palette_panel, 1)
        left.setMinimumWidth(250)
        left.setMaximumWidth(340)
        splitter.addWidget(left)
        splitter.addWidget(self.preview)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Toolbar
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.act_open = QAction("Open Photo", self)
        self.act_svg = QAction("Export SVG", self)
        self.act_png = QAction("Export PNG", self)
        self.act_pdf = QAction("Export PDF", self)
        self.act_pdf.setToolTip("To-scale PDF at the Card size — print and keep at the easel.")
        self.act_grid = QAction("Export Cartoon (to scale)", self)
        self.act_grid.setToolTip(
            "Print the numbered mosaic at its true tessera size, tiled across sheets "
            "you tape together, then trace or paint straight from it."
        )
        self.act_reset = QAction("Reset", self)
        for a in (self.act_open, self.act_svg, self.act_png, self.act_pdf,
                  self.act_grid, self.act_reset):
            tb.addAction(a)
        self.act_grid.setVisible(False)   # mosaic-only
        self.act_svg.setEnabled(False)
        self.act_png.setEnabled(False)
        self.act_pdf.setEnabled(False)

        tb.addSeparator()
        self.act_undo = QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_redo = QAction("Redo", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        for a in (self.act_undo, self.act_redo):
            tb.addAction(a)
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)

        tb.addSeparator()
        self.act_eyedrop = QAction("Eyedropper", self)
        self.act_eyedrop.setCheckable(True)
        self.act_eyedrop.setToolTip("Click the photo on the board to add that color as a target.")
        self.act_add_color = QAction("Add Color…", self)
        self.act_add_color.setToolTip("Pick a target color to mix toward.")
        tb.addAction(self.act_eyedrop)
        tb.addAction(self.act_add_color)

        # Zoom — pure view control on the preview (never re-runs the pipeline).
        tb.addSeparator()
        tb.addWidget(QLabel(" Zoom "))
        self.act_zoom_out = QAction("−", self)
        self.act_zoom_in = QAction("+", self)
        self.act_zoom_fit = QAction("Fit", self)
        for a in (self.act_zoom_out, self.act_zoom_in, self.act_zoom_fit):
            tb.addAction(a)
        self.zoom_label = QLabel(" 100% ")
        tb.addWidget(self.zoom_label)

        # Status bar + progress
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage(
            "Edit your palette on the left, then pick a document: a palette board, "
            "a watercolor painting plan, or a mosaic guide."
        )

        # Debounce timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._render_now)

        # Signals
        self.act_open.triggered.connect(self._choose_file)
        self.act_svg.triggered.connect(self._export_svg)
        self.act_png.triggered.connect(self._export_png)
        self.act_pdf.triggered.connect(self._export_pdf)
        self.act_grid.triggered.connect(self._export_grid)
        self.act_reset.triggered.connect(self._reset)
        self.act_undo.triggered.connect(self._do_undo)
        self.act_redo.triggered.connect(self._do_redo)
        self.act_eyedrop.toggled.connect(self._on_eyedrop_toggled)
        self.act_add_color.triggered.connect(self._on_add_color)
        self.upload.open_clicked.connect(self._choose_file)
        self.upload.file_chosen.connect(self._load_path)
        self.upload.page_changed.connect(self._reload_page)
        self.palette_panel.palette_changed.connect(self._on_palette_changed)
        self.palette_panel.starter_requested.connect(self._on_starter_set)
        self.controls.changed.connect(self._on_controls_changed)
        self.controls.preset_requested.connect(self._on_preset)
        self.controls.preset_save_requested.connect(self._on_preset_save)
        self.controls.preset_delete_requested.connect(self._on_preset_delete)
        self.controls.add_color_requested.connect(self._on_add_color)
        self.controls.clear_picked_requested.connect(self._on_clear_picked)
        self.preview.clicked.connect(self._on_preview_click)
        self.act_zoom_in.triggered.connect(self.preview.zoom_in)
        self.act_zoom_out.triggered.connect(self.preview.zoom_out)
        self.act_zoom_fit.triggered.connect(self.preview.zoom_fit)
        self.preview.zoom_changed.connect(self._on_zoom_changed)

    def _start_worker(self):
        self._thread = QThread(self)
        self._worker = Worker()
        self._worker.moveToThread(self._thread)
        self.request_load.connect(self._worker.load_source)
        self.request_process.connect(self._worker.process)
        self._worker.source_loaded.connect(self._on_source_loaded)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)
        self._thread.start()

    # -- session persistence (auto-restore) ---------------------------------
    def _restore_session(self):
        """Reload the last session (photo + params + window state) on launch."""
        state = decode_session(QSettings().value("session/json", "", str))
        if state is None:
            return

        self.params = state["params"]
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)

        geometry = QSettings().value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        path = state.get("path")
        if path and os.path.exists(path):
            self._path = path
            if is_pdf(path):
                try:
                    pages = count_pdf_pages(path)
                except Exception:  # noqa: BLE001 - a broken PDF shouldn't block launch
                    pages = 0
                self.upload.set_pdf_pages(pages)
                self.upload.set_pdf_state(int(state.get("page", 0)), int(state.get("dpi", 200)))
            else:
                self.upload.hide_pdf()
            self.statusBar().showMessage(f"Restoring {os.path.basename(path)}…")
            self.request_load.emit(path, self.upload.current_page(), self.upload.current_dpi())
        elif path:
            self.statusBar().showMessage(
                "Previous photo not found — settings restored. Open a photo to continue."
            )

    def _save_session(self):
        settings = QSettings()
        settings.setValue(
            "session/json",
            encode_session(
                path=self._path,
                page=self.upload.current_page(),
                dpi=self.upload.current_dpi(),
                params=self.params,
            ),
        )
        settings.setValue("window/geometry", self.saveGeometry())

    # -- file handling ------------------------------------------------------
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open reference photo", "",
            "Images & PDF (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf);;All files (*)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        self._path = path
        if is_pdf(path):
            try:
                pages = count_pdf_pages(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "PDF error", str(exc))
                return
            self.upload.set_pdf_pages(pages)
        else:
            self.upload.hide_pdf()
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        self.request_load.emit(path, self.upload.current_page(), self.upload.current_dpi())

    def _reload_page(self):
        if self._path and is_pdf(self._path):
            self.statusBar().showMessage("Loading page…")
            self.request_load.emit(
                self._path, self.upload.current_page(), self.upload.current_dpi()
            )

    def _on_source_loaded(self, loaded):
        qimg = ndarray_to_qimage(loaded.rgb)
        w, h = loaded.size
        self.upload.set_thumbnail(
            QPixmap.fromImage(qimg),
            f"{os.path.basename(loaded.source_path)}\n{w} × {h} px",
        )
        # Targets are colors, not geometry — they stay valid across photo swaps.
        self._render_now()

    # -- undo / redo of parameter edits --------------------------------------
    def _push_history(self):
        now = time.monotonic()
        if self._undo and (now - self._hist_t) < HISTORY_COALESCE_S:
            self._hist_t = now            # still mid-burst — keep the pre-gesture entry
            return
        self._hist_t = now
        if self._undo and self._undo[-1] == self.params:
            return
        self._undo.append(self.params)
        if len(self._undo) > HISTORY_MAX:
            self._undo.pop(0)
        self._redo.clear()
        self._update_history_actions()

    def _do_undo(self):
        if not self._undo:
            return
        self._redo.append(self.params)
        self._restore_snapshot(self._undo.pop())

    def _do_redo(self):
        if not self._redo:
            return
        self._undo.append(self.params)
        self._restore_snapshot(self._redo.pop())

    def _restore_snapshot(self, snap: Params):
        self.params = snap
        # apply()/set_palette() raise the widgets' guards, so syncing here can't
        # re-enter the changed handlers (no spurious history entries).
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)
        self.act_eyedrop.setChecked(self.params.eyedropper_mode)
        self._update_history_actions()
        self._render_now()

    def _update_history_actions(self):
        self.act_undo.setEnabled(bool(self._undo))
        self.act_redo.setEnabled(bool(self._redo))

    # -- edits -> render -------------------------------------------------------
    def _on_controls_changed(self):
        self._push_history()   # coalesced: one undo entry per edit gesture
        self.params = self.controls.collect(self.params)
        self._timer.start()

    def _on_palette_changed(self, palette):
        self._push_history()
        self.params = self.params.copy(palette=tuple(palette))
        self._timer.start()

    def _on_starter_set(self):
        self._push_history()
        starter = STARTERS.get(self.params.medium)
        if starter:
            self.params = self.params.copy(palette=starter)
            self.palette_panel.set_palette(starter)
            self.statusBar().showMessage("Loaded the starter palette for this medium.")
            self._render_now()

    def _on_preset(self, name: str):
        self._push_history()
        if name in self._user_presets:
            self.params = apply_preset_values(self.params, self._user_presets[name])
        else:
            self.params = apply_preset(self.params, name)
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)
        self._render_now()

    # -- user presets (saved looks) -----------------------------------------
    def _load_user_presets(self):
        raw = QSettings().value("presets/json", "", str)
        try:
            data = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            data = {}
        self._user_presets = data if isinstance(data, dict) else {}
        self.controls.set_user_presets(sorted(self._user_presets))

    def _persist_user_presets(self):
        QSettings().setValue("presets/json", json.dumps(self._user_presets))

    def _on_preset_save(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        if name == "Custom" or name in PRESETS:
            QMessageBox.warning(self, "Save preset", f"'{name}' is a reserved preset name.")
            return
        self._user_presets[name] = preset_to_dict(self.params)
        self._persist_user_presets()
        self.controls.set_user_presets(sorted(self._user_presets))
        self.statusBar().showMessage(f"Saved preset '{name}'.")

    def _on_preset_delete(self, name: str):
        if name not in self._user_presets:
            QMessageBox.information(
                self, "Delete preset", "Select one of your saved presets to delete.")
            return
        del self._user_presets[name]
        self._persist_user_presets()
        self.controls.set_user_presets(sorted(self._user_presets))
        self.statusBar().showMessage(f"Deleted preset '{name}'.")

    # -- targets -----------------------------------------------------------------
    def _on_add_color(self):
        color = QColorDialog.getColor(parent=self, title="Add target color")
        if not color.isValid():
            return
        self._push_history()
        targets = self.params.manual_targets + (Target(hex=color.name()),)
        self.params = self.params.copy(manual_targets=targets)
        self._render_now()

    def _on_clear_picked(self):
        if not self.params.manual_targets:
            return
        self._push_history()
        self.params = self.params.copy(manual_targets=(), selected_target=-1)
        self.statusBar().showMessage("Removed all picked target colors.")
        self._render_now()

    def _on_eyedrop_toggled(self, on: bool):
        # Transient mode, not undoable, in no cache key.
        self.params = self.params.copy(eyedropper_mode=bool(on))
        self.statusBar().showMessage(
            "Eyedropper on — click the photo on the board to add a target."
            if on else "Eyedropper off."
        )

    # -- preview click routing ------------------------------------------------------
    def _on_preview_click(self, x: float, y: float, label: int):
        if self._result is None:
            return
        layout = self._result.layout
        # Plan mode: clicking a wash highlights the shapes it covers on the map.
        for rect, idx in layout.pass_rects:
            if _in_rect(x, y, rect):
                sel = -1 if idx == self.params.selected_pass else idx
                self.params = self.params.copy(selected_pass=sel)
                self.statusBar().showMessage(
                    f"Wash {idx + 1} highlighted on the map — click again to clear."
                    if sel >= 0 else "Highlight cleared."
                )
                self._render_now()
                return
        # Mosaic mode: clicking a legend row highlights that colour's tesserae.
        for rect, idx in layout.color_rects:
            if _in_rect(x, y, rect):
                sel = -1 if idx == self.params.selected_color else idx
                self.params = self.params.copy(selected_color=sel)
                self.statusBar().showMessage(
                    f"Colour {idx + 1} highlighted — click again to clear."
                    if sel >= 0 else "Highlight cleared."
                )
                self._render_now()
                return
        # Eyedropper: sample the photo region.
        if (
            self.params.eyedropper_mode
            and layout.photo_rect is not None
            and self._result.working_rgb is not None
            and _in_rect(x, y, layout.photo_rect)
        ):
            rx, ry, rw, rh = layout.photo_rect
            iw, ih = layout.photo_img_size
            px = int((x - rx) / rw * iw)
            py = int((y - ry) / rh * ih)
            img = self._result.working_rgb
            x0, x1 = max(0, px - 1), min(iw, px + 2)
            y0, y1 = max(0, py - 1), min(ih, py + 2)
            mean = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            hex_ = "#%02x%02x%02x" % tuple(int(round(c)) for c in mean)
            self._push_history()
            targets = self.params.manual_targets + (Target(hex=hex_, source="eyedrop"),)
            self.params = self.params.copy(manual_targets=targets)
            self.statusBar().showMessage(f"Added target {hex_} from the photo.")
            self._render_now()
            return
        # Target rows: left-click selects (and anchors the harmony); right-click
        # removes a picked target (auto targets shrink via the extract count).
        for rect, idx in layout.target_rects:
            if _in_rect(x, y, rect):
                n_auto = len(self._result.auto_targets)
                if label == 0:
                    if idx >= n_auto:
                        self._push_history()
                        manual = list(self.params.manual_targets)
                        del manual[idx - n_auto]
                        self.params = self.params.copy(
                            manual_targets=tuple(manual), selected_target=-1)
                        self.statusBar().showMessage("Removed the picked target.")
                        self._render_now()
                    else:
                        self.statusBar().showMessage(
                            "Auto target — lower “Colors from photo” to remove it.")
                    return
                hex_ = self._result.recipes[idx].target_hex if idx < len(self._result.recipes) else ""
                self.params = self.params.copy(selected_target=idx, anchor_hex=hex_)
                self.controls.apply(self.params)
                self.statusBar().showMessage(f"Selected target {idx + 1} — harmony anchored on it.")
                self._render_now()
                return
        # Paint swatches: anchor the harmony on that paint.
        for rect, idx in layout.swatch_rects:
            if _in_rect(x, y, rect) and idx < len(self.params.palette):
                paint = self.params.palette[idx]
                self.params = self.params.copy(anchor_hex=paint.hex)
                self.controls.apply(self.params)
                self.statusBar().showMessage(f"Harmony anchored on {paint.name}.")
                self._render_now()
                return

    def _on_zoom_changed(self, zoom: float):
        self.zoom_label.setText(f" {int(round(zoom * 100))}% ")

    # -- render scheduling --------------------------------------------------
    def _render_now(self):
        self._pending = True
        if not self._busy:
            self._dispatch()

    def _dispatch(self):
        if not self._pending:
            return
        self._pending = False
        self._busy = True
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.request_process.emit(RenderRequest(params=self.params))

    # -- worker callbacks ---------------------------------------------------
    def _on_progress(self, label: str, frac: float):
        self.statusBar().showMessage(label)
        self.progress.setValue(int(frac * 100))

    def _on_finished(self, result: Result):
        self._busy = False
        self._result = result
        scene = result.scene
        image = render_scene(scene, 1.0)
        self.preview.set_image(image, scene.width, scene.height, scene.margin)
        self.act_svg.setEnabled(True)
        self.act_png.setEnabled(True)
        self.act_pdf.setEnabled(True)
        is_mosaic = self.params.doc_mode == DOC_MOSAIC
        self.act_grid.setVisible(is_mosaic)
        self.act_grid.setEnabled(is_mosaic and result.mosaic is not None)
        self.progress.setVisible(False)
        note = ", ".join(result.recomputed) if result.recomputed else "style only"
        if self.params.doc_mode == DOC_PLAN and not result.passes and not self._path:
            self.statusBar().showMessage("Open a reference photo to build a painting plan.")
        else:
            self.statusBar().showMessage(f"Ready — recomputed: {note}")
        if self._pending:
            self._dispatch()

    # -- naming -------------------------------------------------------------
    def _doc_stem(self) -> str:
        return {
            DOC_PLAN: "painting-plan",
            DOC_MOSAIC: "mosaic-guide",
        }.get(self.params.doc_mode, "palette-board")

    def _on_failed(self, message: str):
        self._busy = False
        self.progress.setVisible(False)
        self.statusBar().showMessage("Processing failed.")
        QMessageBox.warning(self, "Painting Studio", message)
        if self._pending:
            self._dispatch()

    # -- export & reset -----------------------------------------------------
    def _export_svg(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SVG", f"{self._doc_stem()}.svg", "SVG (*.svg)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(scene_to_svg(self._result.scene))
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_png(self):
        if not self._result:
            return
        scene = self._result.scene
        longest = max(scene.width, scene.height)
        target, ok = QInputDialog.getInt(
            self, "Export PNG", "Longest edge (px):", longest, 128, 8192, 64
        )
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", f"{self._doc_stem()}.png", "PNG (*.png)")
        if not path:
            return
        try:
            scale = target / float(longest)
            image = render_scene(scene, scale)
            if not image.save(path, "PNG"):
                raise RuntimeError("QImage.save returned False")
            self.statusBar().showMessage(
                f"Saved {os.path.basename(path)} ({image.width()}×{image.height()})")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_pdf(self):
        if not self._result:
            return
        scene = self._result.scene
        ipp = float(scene.inches_per_px)
        if ipp <= 0:
            QMessageBox.information(
                self, "Export PDF",
                "Set a Card width in the Card size panel first — the PDF prints "
                "at that physical size so the board reads true at the easel.")
            return
        w_in = scene.width * ipp
        h_in = scene.height * ipp
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", f"{self._doc_stem()}.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            write_scale_pdf_vector(scene, path)
            unit = scene.canvas_units
            wd, hd = (w_in, h_in) if unit != UNIT_CM else (w_in * CM_PER_INCH, h_in * CM_PER_INCH)
            self.statusBar().showMessage(
                f"Saved {os.path.basename(path)} ({wd:.1f}×{hd:.1f} {unit} to scale)")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_grid(self):
        """Write the numbered mosaic grid at true tessera size, tiled across pages."""
        if not self._result or self._result.mosaic is None:
            return
        grid = self._result.mosaic
        sheet = build_grid_sheet(self.params, grid)
        w_in = sheet.width * sheet.inches_per_px
        h_in = sheet.height * sheet.inches_per_px
        path, _ = QFileDialog.getSaveFileName(
            self, "Export cartoon to scale", "mosaic-cartoon.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            pages, rows, cols = write_tiled_pdf(sheet, path)
            self.statusBar().showMessage(
                f"Saved {os.path.basename(path)} — {grid.total} tesserae at "
                f"{self.params.tessera_mm:g} mm ({w_in:.1f}×{h_in:.1f} in) "
                f"across {pages} page{'s' if pages != 1 else ''} ({rows}×{cols})."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(exc))

    def _reset(self):
        self._push_history()   # a Reset is undoable
        self.params = Params()
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)
        self.act_eyedrop.setChecked(False)
        self.preview.zoom_fit()
        self._render_now()

    # -- shutdown -----------------------------------------------------------
    def closeEvent(self, event):
        self._save_session()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)
