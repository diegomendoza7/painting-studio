"""The main application window: assembles panels, worker thread and exports."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QMenu,
    QToolButton,
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
from ..library import Library, palette_or_first
from ..session import decode_session, encode_session, params_from_dict, params_to_dict
from ..svg import scene_to_svg
from . import projects
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
    request_load = Signal(str, int, int, int)
    request_adopt = Signal(object, int)
    request_process = Signal(object)       # RenderRequest
    request_clear = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Painting Studio")
        self.resize(1320, 860)

        self.params = Params()
        self._path: Optional[str] = None
        # The open project (None = an untitled workspace), and the state it was
        # last saved or opened in — what "has this changed?" compares against.
        from .easel_window import library_root
        self.lib = Library(library_root())
        self.project = None
        self._saved_key = None
        self._baseline_params = None
        self._easel = None
        self._busy = False
        self._pending = False
        self._result: Optional[Result] = None
        self._revision = 0
        self._inflight_state = None
        self._result_state = None
        self._load_token = 0
        self._source_token = 0
        self._loading_file = False
        self._mark_on_load = False
        self._committed_pdf = (0, 200, 0)
        self._requested_pdf = self._committed_pdf
        self._loading_path = None
        self._closing = False
        self._shutdown_done = False
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
        self.act_open = QAction("Reference Photo…", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_new_project = QAction("New Project", self)
        self.act_new_project.setShortcut(QKeySequence.New)
        self.act_open_project = QAction("Open…", self)
        self.act_open_project.setIconText("Open")
        self.act_open_project.setShortcut(QKeySequence.Open)
        self.act_open_project.setToolTip("Open a project (⌘O). The arrow lists recent ones.")
        self.act_save = QAction("Save", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.setToolTip("Save this project (⌘S) — photo, palette and every setting.")
        self.act_save_as = QAction("Save As…", self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.recent_menu = QMenu("Open Recent", self)
        # The Easel adds to the list too, so build it fresh each time it opens.
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
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
        self.act_easel = QAction("Easel", self)
        self.act_easel.setToolTip(
            "Open the easel: click your reference to see a colour big, with how to mix "
            "it from your palette. Swatches are saved per painting.")
        for a in (self.act_open_project, self.act_save, self.act_svg, self.act_png,
                  self.act_pdf, self.act_grid, self.act_reset, self.act_easel):
            tb.addAction(a)
        # Recent projects hang off the Open button's ▾, as in the drawing app. On
        # the button, not the action: an action with a menu turns into a
        # submenu in the File menu and stops opening the dialog.
        open_btn = tb.widgetForAction(self.act_open_project)
        if isinstance(open_btn, QToolButton):
            open_btn.setMenu(self.recent_menu)
            open_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.file_menu = file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.act_new_project)
        file_menu.addAction(self.act_open_project)
        file_menu.addMenu(self.recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        for a in (self.act_svg, self.act_png, self.act_pdf, self.act_grid):
            file_menu.addAction(a)
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
        self.act_new_project.triggered.connect(self._new_project)
        self.act_open_project.triggered.connect(self._open_project_dialog)
        self.act_save.triggered.connect(self._save)
        self.act_save_as.triggered.connect(self._save_as)
        self.act_svg.triggered.connect(self._export_svg)
        self.act_png.triggered.connect(self._export_png)
        self.act_pdf.triggered.connect(self._export_pdf)
        self.act_grid.triggered.connect(self._export_grid)
        self.act_easel.triggered.connect(self._open_easel)
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
        self.request_adopt.connect(self._worker.adopt_source)
        self.request_clear.connect(self._worker.clear_source)
        self.request_process.connect(self._worker.process)
        self._worker.source_loaded.connect(self._on_source_loaded)
        self._worker.load_failed.connect(self._on_load_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)
        self._thread.finished.connect(self._finish_close)
        self._thread.start()

    # -- session persistence (auto-restore) ---------------------------------
    def _own_palette(self):
        """The artist's palette from the library, if they've chosen one."""
        try:
            name = self.lib.default_palette_name()
            return self.lib.palette(name) if name else None
        except OSError:
            return None

    def _with_own_palette(self, params: Params) -> Params:
        pal = self._own_palette()
        return params.copy(palette=pal.paints, medium=pal.medium) if pal else params

    def _restore_session(self):
        """On launch: reopen the last project, or else the untitled workspace."""
        settings = QSettings()
        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._rebuild_recent_menu()
        last = projects.project_or_none(self.lib, settings.value(projects.LAST_KEY, "", str))
        if last is not None:
            self._open_project(last.id, ask=False)
            return
        state = decode_session(settings.value("session/json", "", str))
        if state is None:
            # First launch: start from the artist's own palette, not a starter kit.
            self._apply_document(self._with_own_palette(self.params), None, render=False)
        else:
            path = state.get("path")
            self._apply_document(state["params"], path, int(state.get("page", 0)),
                                 int(state.get("dpi", 200)), render=False)
            if path and not os.path.exists(path):
                self.statusBar().showMessage(
                    "Previous photo not found — settings restored. Open a photo to continue.")
        self._mark_saved()

    def _save_session(self):
        settings = QSettings()
        # The untitled workspace still restores itself exactly as before; a
        # project reopens from its own saved file.
        settings.setValue(
            "session/json",
            encode_session(
                path=self._path,
                page=self.upload.current_page(),
                dpi=self.upload.current_dpi(),
                params=self.params,
            ),
        )
        settings.setValue(projects.LAST_KEY, self.project.id if self.project else "")
        settings.setValue("window/geometry", self.saveGeometry())

    # -- projects -----------------------------------------------------------
    def _apply_document(self, params: Params, path: Optional[str], page: int = 0,
                        dpi: int = 200, render: bool = True):
        """Put a whole document on screen: settings, palette, photo. Undo starts over."""
        self.params = params
        self._undo.clear()
        self._redo.clear()
        self._update_history_actions()
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)
        self.act_eyedrop.setChecked(False)
        self._revision += 1
        self._load_token += 1
        self._source_token = self._load_token
        self._loading_file = False
        self._mark_on_load = False
        self._loading_path = None
        self._path = None
        self._result = None
        self._result_state = None
        self._committed_pdf = (0, dpi, 0)
        self._set_export_enabled(False)
        self.preview.clear("Building the document…")
        self.request_clear.emit(self._source_token)
        if path and os.path.exists(path):
            self._mark_on_load = True
            if is_pdf(path):
                try:
                    pages = count_pdf_pages(path)
                except Exception:  # noqa: BLE001 - a broken PDF shouldn't block opening
                    pages = 0
                self.upload.set_pdf_pages(pages)
                self.upload.set_pdf_state(page, dpi)
            else:
                self.upload.hide_pdf()
            self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
            self._request_load(path, self.upload.current_page(), self.upload.current_dpi())
        else:
            self.upload.clear()
            if render:
                self._render_now()
        self.preview.zoom_fit()

    def _doc_key(self):
        """Everything a save captures. Clicks that only highlight don't count as edits."""
        p = self.params.copy(eyedropper_mode=False, selected_target=-1,
                             selected_pass=-1, selected_color=-1)
        pdf = bool(self._path and is_pdf(self._path))
        return (p, self._path or "",
                self.upload.current_page() if pdf else 0,
                self.upload.current_dpi() if pdf else 0)

    def _is_dirty(self) -> bool:
        return self._saved_key is not None and self._doc_key() != self._saved_key

    def _needs_saving(self) -> bool:
        """Before replacing what's on screen: a changed project, or untitled work with a photo."""
        return self._is_dirty() or (self.project is None and self._path is not None)

    def _mark_saved(self):
        self._saved_key = self._doc_key()
        self._baseline_params = params_to_dict(self.params)
        self._update_title()

    def _update_title(self):
        name = self.project.name if self.project else "Untitled"
        self.setWindowTitle(f"{name}[*] — Painting Studio")
        self.setWindowModified(self._is_dirty())

    def _maybe_save(self) -> bool:
        """Ask before unsaved work is replaced. False means stop what you were doing."""
        if not self._needs_saving():
            return True
        answer = projects.ask_save_changes(self, self.project.name if self.project else "Untitled")
        if answer == projects.CANCEL:
            return False
        if answer == projects.SAVE:
            return self._save()
        return True

    def _document(self) -> dict:
        p = self.params.copy(eyedropper_mode=False, selected_target=-1,
                             selected_pass=-1, selected_color=-1)
        return {"params": params_to_dict(p), "page": self.upload.current_page(),
                "dpi": self.upload.current_dpi()}

    def _external_photo(self) -> Optional[str]:
        """The photo on screen, if it isn't already this project's own copy."""
        if not self._path:
            return None
        if self.project is not None:
            inside = os.path.abspath(self._path).startswith(os.path.abspath(self.project.path) + os.sep)
            if inside:
                return None
        return self._path

    def _new_project(self):
        if not self._maybe_save():
            return
        self.project = None
        self._apply_document(self._with_own_palette(Params()), None)
        self._mark_saved()
        self.statusBar().showMessage(
            "New project — open a reference photo, then Save (⌘S) to keep it.")

    def _open_project_dialog(self):
        dlg = projects.OpenProjectDialog(self, self.lib, self.project.id if self.project else "")
        if dlg.exec() == dlg.Accepted and dlg.chosen():
            self._open_project(dlg.chosen())

    def _open_project(self, pid: str, ask: bool = True):
        if self.project is not None and pid == self.project.id and not self._is_dirty():
            return
        if ask and not self._maybe_save():
            return
        proj = projects.project_or_none(self.lib, pid)
        if proj is None:
            QMessageBox.warning(self, "Open project", "That project can't be found any more.")
            projects.remove_recent(pid)
            self._rebuild_recent_menu()
            return
        doc = proj.document or {}
        if doc.get("params"):
            params = params_from_dict(doc["params"])
            pal = self.lib.palette(proj.palette)
            if pal is not None:
                params = params.copy(palette=pal.paints, medium=pal.medium)
            if proj.canvas_w > 0 and proj.canvas_h > 0:
                params = params.copy(paper_w=proj.canvas_w, paper_h=proj.canvas_h,
                                     canvas_units=proj.units)
        else:
            # Started in the Easel: open with its palette and a fresh board.
            pal = palette_or_first(self.lib, proj.palette)
            params = Params().copy(palette=pal.paints, medium=pal.medium)
        self.project = proj
        self._apply_document(params, self.lib.reference_path(proj),
                             int(doc.get("page", 0)), int(doc.get("dpi", 200)))
        projects.add_recent(pid)
        QSettings().setValue(projects.LAST_KEY, pid)
        self._rebuild_recent_menu()
        self._mark_saved()
        self.statusBar().showMessage(f"Opened “{proj.name}”.")

    def _save(self) -> bool:
        if self._loading_file:
            self.statusBar().showMessage("Wait for the reference photo to finish loading before saving.")
            return False
        if self.project is None:
            return self._save_as()
        try:
            self.project = self.lib.save_document(self.project.id, self._document(),
                                                  self._external_photo(),
                                                  {"params": self._baseline_params})
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save", f"Couldn't save: {exc}")
            return False
        self._path = self.lib.reference_path(self.project) or self._path
        saved = params_from_dict(self.project.document["params"])
        new = self.params.copy(palette=saved.palette, medium=saved.medium,
                               paper_w=saved.paper_w, paper_h=saved.paper_h,
                               canvas_units=saved.canvas_units)
        if new != self.params:
            self.params = new
            self.controls.apply(new)
            self.palette_panel.set_palette(new.palette)
            self._render_now()
        self._after_save(f"Saved “{self.project.name}”.")
        return True

    def _save_as(self) -> bool:
        if self._loading_file:
            self.statusBar().showMessage("Wait for the reference photo to finish loading before saving.")
            return False
        if self.project is not None:
            default = f"{self.project.name} copy"
        elif self._path:
            default = os.path.splitext(os.path.basename(self._path))[0]
        else:
            default = "Untitled"
        name, ok = QInputDialog.getText(self, "Save project", "Project name:", text=default)
        name = name.strip()
        if not ok or not name:
            return False
        photo = self._external_photo()          # decided before the project changes
        try:
            if self.project is not None:
                new = self.lib.duplicate_project(self.project.id, name)
            else:
                pal = self.lib.default_palette_name() or next(iter(self.lib.palettes()), "")
                new = self.lib.create_project(name, pal, None, self.params.paper_w,
                                              self.params.paper_h, self.params.canvas_units)
            self.project = self.lib.save_document(new.id, self._document(), photo)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save As", f"Couldn't save: {exc}")
            return False
        self._path = self.lib.reference_path(self.project) or self._path
        self._after_save(f"Saved as “{self.project.name}”.")
        return True

    def _after_save(self, message: str):
        projects.add_recent(self.project.id)
        QSettings().setValue(projects.LAST_KEY, self.project.id)
        self._rebuild_recent_menu()
        self._mark_saved()
        self.statusBar().showMessage(message)
        if self._easel is not None:
            self._easel.refresh_projects()       # a new project, or its photo changed

    def _rebuild_recent_menu(self):
        self.recent_menu.clear()
        recent = projects.recent_projects(self.lib)
        if not recent:
            none = self.recent_menu.addAction("No recent projects")
            none.setEnabled(False)
            return
        for proj in recent:
            act = self.recent_menu.addAction(proj.name)
            act.triggered.connect(lambda _=False, pid=proj.id: self._open_project(pid))
        self.recent_menu.addSeparator()
        clear = self.recent_menu.addAction("Clear Menu")
        clear.triggered.connect(self._clear_recent)

    def _clear_recent(self):
        QSettings().setValue(projects.RECENT_KEY, [])
        self._rebuild_recent_menu()

    # -- file handling ------------------------------------------------------
    def _request_load(self, path: str, page: int, dpi: int):
        if self._closing:
            return
        self._load_token += 1
        self._revision += 1
        self._loading_file = True
        self._loading_path = path
        self._requested_pdf = (page, dpi, self.upload.page_combo.count() if is_pdf(path) else 0)
        self._set_export_enabled(False)
        self.request_load.emit(path, page, dpi, self._load_token)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open reference photo", "",
            "Images & PDF (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf);;All files (*)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        if self.project is not None and self._external_photo() is None and self._path:
            fresh = projects.project_or_none(self.lib, self.project.id)
            n = len(fresh.swatches) if fresh else 0
            if n and QMessageBox.question(
                    self, "Replace reference photo",
                    f"“{self.project.name}” has {n} Easel swatch{'es' if n != 1 else ''}. "
                    "They keep their colours and recipes, but once you save, their "
                    "outlines will point at the new photo. Replace it?") != QMessageBox.Yes:
                return
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
        self._mark_on_load = False
        self._request_load(path, self.upload.current_page(), self.upload.current_dpi())

    def _reload_page(self):
        path = self._loading_path if self._loading_file else self._path
        if path and is_pdf(path):
            self.statusBar().showMessage("Loading page…")
            self._request_load(
                path, self.upload.current_page(), self.upload.current_dpi()
            )

    def _on_source_loaded(self, loaded, token=None):
        if self._closing or (token is not None and token != self._load_token):
            return
        self._loading_file = False
        self._loading_path = None
        self._path = loaded.source_path
        if self._mark_on_load:
            self._mark_on_load = False
            self._mark_saved()
        self._source_token = self._load_token if token is None else token
        self._committed_pdf = (loaded.page_index, self._requested_pdf[1], self._requested_pdf[2])
        self.upload.set_pdf_state(*self._committed_pdf[:2])
        self.request_adopt.emit(loaded, self._source_token)
        self._result = None
        self._result_state = None
        self.preview.clear("Building the document…")
        qimg = ndarray_to_qimage(loaded.rgb)
        w, h = loaded.size
        self.upload.set_thumbnail(
            QPixmap.fromImage(qimg),
            f"{os.path.basename(loaded.source_path)}\n{w} × {h} px",
        )
        # Targets are colors, not geometry — they stay valid across photo swaps.
        self._render_now()

    def _on_load_failed(self, message: str, token: int):
        if self._closing or token != self._load_token:
            return
        self._loading_file = False
        self._loading_path = None
        self._mark_on_load = False
        page, dpi, pages = self._committed_pdf
        self.upload.set_pdf_pages(pages)
        self.upload.set_pdf_state(page, dpi)
        QMessageBox.warning(self, "Open reference", message)
        self._render_now()

    # -- undo / redo of parameter edits --------------------------------------
    def _push_history(self):
        now = time.monotonic()
        self._redo.clear()
        self._update_history_actions()
        if self._undo and (now - self._hist_t) < HISTORY_COALESCE_S:
            self._hist_t = now            # still mid-burst — keep the pre-gesture entry
            return
        self._hist_t = now
        if self._undo and self._undo[-1] == self.params:
            return
        self._undo.append(self.params)
        if len(self._undo) > HISTORY_MAX:
            self._undo.pop(0)
        self._update_history_actions()

    def _do_undo(self):
        if not self._undo:
            return
        self._redo.append(self.params)
        self._hist_t = 0.0
        self._restore_snapshot(self._undo.pop())

    def _do_redo(self):
        if not self._redo:
            return
        self._undo.append(self.params)
        self._hist_t = 0.0
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
        self._set_export_enabled(False)
        self._timer.start()

    def _on_palette_changed(self, palette):
        self._push_history()
        self.params = self.params.copy(palette=tuple(palette))
        self._set_export_enabled(False)
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
        if self._closing:
            return
        self._revision += 1
        self._set_export_enabled(False)
        self._pending = True
        if not self._busy and not self._loading_file:
            self._dispatch()

    def _dispatch(self):
        if not self._pending or self._closing or self._loading_file:
            return
        self._pending = False
        self._busy = True
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._inflight_state = self.params
        self.request_process.emit(RenderRequest(params=self.params, revision=self._revision,
                                                source_token=self._source_token))

    # -- worker callbacks ---------------------------------------------------
    def _on_progress(self, label: str, frac: float):
        if self._closing or self._loading_file:
            return
        self.statusBar().showMessage(label)
        self.progress.setValue(int(frac * 100))

    def _on_finished(self, result: Result, revision=None):
        self._busy = False
        if self._closing:
            return
        if (result is None or self._loading_file or
                (revision is not None and (revision != self._revision or
                 self._inflight_state != self.params))):
            if self._pending:
                self._dispatch()
            return
        self._result = result
        self._result_state = self.params
        scene = result.scene
        image = render_scene(scene, 1.0)
        self.preview.set_image(image, scene.width, scene.height, scene.margin)
        self._set_export_enabled(True)
        is_mosaic = self.params.doc_mode == DOC_MOSAIC
        self.act_grid.setVisible(is_mosaic)
        self.act_grid.setEnabled(is_mosaic and result.mosaic is not None)
        self.progress.setVisible(False)
        self._update_title()
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

    def _on_failed(self, message: str, revision=None):
        if self._closing:
            return
        if revision is not None and (revision != self._revision or self._loading_file
                                     or self._inflight_state != self.params):
            self._busy = False
            if self._pending:
                self._dispatch()
            return
        self._busy = False
        self.progress.setVisible(False)
        self.statusBar().showMessage("Processing failed.")
        QMessageBox.warning(self, "Painting Studio", message)
        if self._pending:
            self._dispatch()

    # -- export & reset -----------------------------------------------------
    def _set_export_enabled(self, enabled: bool):
        for action in (self.act_svg, self.act_png, self.act_pdf, self.act_grid):
            action.setEnabled(enabled)

    def _export_ready(self) -> bool:
        return (self._result is not None and not self._closing and not self._busy
                and not self._pending and not self._loading_file
                and self._result_state == self.params)

    def _export_svg(self):
        if not self._export_ready():
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
        if not self._export_ready():
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
        if not self._export_ready():
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

    def _open_easel(self):
        from .easel_window import EaselWindow

        if self._easel is None:
            self._easel = EaselWindow(library=self.lib)
        if self.project is not None:
            self._easel.show_project(self.project.id)
        self._easel.show()
        self._easel.raise_()
        self._easel.activateWindow()

    def _export_grid(self):
        """Write the numbered mosaic grid at true tessera size, tiled across pages."""
        if not self._export_ready() or self._result.mosaic is None:
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
        # Reset the settings, not the paints you own: fall back to your palette.
        self.params = self._with_own_palette(Params())
        self.controls.apply(self.params)
        self.palette_panel.set_palette(self.params.palette)
        self.act_eyedrop.setChecked(False)
        self.preview.zoom_fit()
        self._render_now()

    # -- shutdown -----------------------------------------------------------
    def changeEvent(self, event):
        if (event.type() == QEvent.ActivationChange and self.isActiveWindow()
                and self.project is not None and not self._closing
                and not self._is_dirty() and not self._loading_file):
            fresh = projects.project_or_none(self.lib, self.project.id)
            if fresh is not None:
                pal = self.lib.palette(fresh.palette)
                changes = dict(paper_w=fresh.canvas_w, paper_h=fresh.canvas_h,
                               canvas_units=fresh.units)
                if pal is not None:
                    changes.update(palette=pal.paints, medium=pal.medium)
                new = self.params.copy(**changes)
                self.project = fresh
                if new != self.params:
                    self.params = new
                    self.controls.apply(new)
                    self.palette_panel.set_palette(new.palette)
                    self._mark_saved()
                    self._render_now()
        super().changeEvent(event)

    def closeEvent(self, event):
        if self._shutdown_done:
            super().closeEvent(event)
            return
        if self._closing:
            event.ignore()
            return
        # A project with unsaved changes asks first. The untitled workspace
        # doesn't: it restores itself next launch, so nothing can be lost.
        if self.project is not None and self._is_dirty():
            answer = projects.ask_save_changes(self, self.project.name)
            if answer == projects.CANCEL or (answer == projects.SAVE and not self._save()):
                event.ignore()
                return
        self._save_session()
        self._closing = True
        self._timer.stop()
        self._pending = False
        self.setEnabled(False)
        self.statusBar().showMessage("Finishing current processing before closing…")
        self._thread.requestInterruption()
        self._thread.quit()
        event.ignore()
        self._finish_close()

    def _finish_close(self):
        if self._closing and not self._shutdown_done and not self._thread.isRunning():
            self._shutdown_done = True
            QTimer.singleShot(0, self.close)
