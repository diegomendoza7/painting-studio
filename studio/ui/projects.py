"""Projects in the UI: the recent list, the Open dialog, and "save changes?".

A project is one painting — its reference photo, the document window's
board / plan / mosaic settings, and the Easel's swatches — stored as a folder
by :class:`studio.library.Library`. Both windows open the same projects, so the
recent list lives here and is shared between them.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)

from ..library import Library, Project

RECENT_KEY = "projects/recent"
LAST_KEY = "projects/last"
MAX_RECENT = 8
THUMB = QSize(72, 72)


# --- recent projects -------------------------------------------------------------
def recent_ids() -> List[str]:
    raw = QSettings().value(RECENT_KEY, [])
    if isinstance(raw, str):          # QSettings flattens a one-item list to a string
        raw = [raw] if raw else []
    return [str(x) for x in (raw or []) if x]


def add_recent(pid: str) -> None:
    ids = [pid] + [x for x in recent_ids() if x != pid]
    QSettings().setValue(RECENT_KEY, ids[:MAX_RECENT])


def remove_recent(pid: str) -> None:
    QSettings().setValue(RECENT_KEY, [x for x in recent_ids() if x != pid])


def recent_projects(lib: Library) -> List[Project]:
    """The recent list as projects, skipping any that have since been deleted."""
    out: List[Project] = []
    for pid in recent_ids():
        try:
            out.append(lib.load_project(pid))
        except (OSError, ValueError, KeyError):
            continue
    return out


# --- thumbnails ------------------------------------------------------------------
def project_icon(lib: Library, proj: Project) -> QIcon:
    """The reference photo, decoded straight to thumbnail size (fast on big photos)."""
    path = lib.reference_path(proj)
    pix = QPixmap(THUMB)
    pix.fill(Qt.darkGray)
    if path:
        reader = QImageReader(path)
        reader.setAutoTransform(True)                 # phone photos: honour EXIF rotation
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            reader.setScaledSize(size.scaled(THUMB, Qt.KeepAspectRatio))
        img = reader.read()
        if not img.isNull():
            pix = QPixmap.fromImage(img)
    return QIcon(pix)


def describe(proj: Project) -> str:
    kind = ((proj.document or {}).get("params") or {}).get("doc_mode", "")
    doc = {"board": "palette board", "plan": "painting plan", "mosaic": "mosaic"}.get(kind, "")
    parts = [p for p in (doc, f"{len(proj.swatches)} swatch{'es' if len(proj.swatches) != 1 else ''}"
                         if proj.swatches else "") if p]
    when = (proj.updated or proj.created)[:16].replace("T", " ")
    return f"{proj.name}\n{' · '.join(parts) or 'empty'}  —  saved {when}"


# --- the Open dialog ---------------------------------------------------------------
class OpenProjectDialog(QDialog):
    """Every project, newest first, with its reference photo."""

    def __init__(self, parent, lib: Library, current: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Open project")
        self.resize(520, 560)
        self._chosen = ""
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setIconSize(THUMB)
        self.list.setSpacing(2)
        for proj in lib.projects():
            item = QListWidgetItem(project_icon(lib, proj), describe(proj))
            item.setData(Qt.UserRole, proj.id)
            if proj.id == current:
                item.setText(item.text().replace(proj.name, f"{proj.name}  (open now)", 1))
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
            lay.addWidget(self.list, 1)
        else:
            lay.addWidget(QLabel("No projects yet — Save one first (⌘S)."))
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Open).setEnabled(self.list.count() > 0)
        lay.addWidget(buttons)

    def chosen(self) -> str:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else ""


# --- save changes? -----------------------------------------------------------------
SAVE, DISCARD, CANCEL = "save", "discard", "cancel"


def ask_save_changes(parent, name: str) -> str:
    """The standard three-way question before unsaved work is closed or replaced."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setText(f"Save changes to “{name}”?")
    box.setInformativeText("Your changes will be lost if you don't save them.")
    box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Save)
    answer = box.exec()
    if answer == QMessageBox.Save:
        return SAVE
    if answer == QMessageBox.Discard:
        return DISCARD
    return CANCEL


def project_or_none(lib: Library, pid: Optional[str]) -> Optional[Project]:
    if not pid:
        return None
    try:
        return lib.load_project(pid)
    except (OSError, ValueError, KeyError):
        return None
