"""Left panel — reference photo upload (drag-drop / open), thumbnail, PDF page picker."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.loader import is_supported


class UploadPanel(QWidget):
    file_chosen = Signal(str)     # path
    page_changed = Signal()       # page or dpi changed
    open_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.open_btn = QPushButton("Open reference photo…")
        self.open_btn.clicked.connect(self.open_clicked)
        lay.addWidget(self.open_btn)

        self.drop = QLabel("Drop a reference photo here")
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setWordWrap(True)
        self.drop.setFrameShape(QFrame.StyledPanel)
        self.drop.setMinimumHeight(160)
        self.drop.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; color: #888; }"
        )
        lay.addWidget(self.drop)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #bbb;")
        lay.addWidget(self.info)

        # PDF controls (hidden for raster images)
        self.pdf_row = QWidget()
        pdf_lay = QVBoxLayout(self.pdf_row)
        pdf_lay.setContentsMargins(0, 0, 0, 0)
        self.page_combo = QComboBox()
        self.page_combo.currentIndexChanged.connect(self._on_pdf_change)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 600)
        self.dpi.setValue(200)
        self.dpi.setSuffix(" dpi")
        self.dpi.valueChanged.connect(self._on_pdf_change)
        pdf_lay.addWidget(QLabel("PDF page"))
        pdf_lay.addWidget(self.page_combo)
        pdf_lay.addWidget(self.dpi)
        self.pdf_row.setVisible(False)
        lay.addWidget(self.pdf_row)

        lay.addStretch(1)
        self._suppress = False

    # -- public API ---------------------------------------------------------
    def set_thumbnail(self, pixmap: QPixmap, caption: str):
        scaled = pixmap.scaled(
            self.drop.width() - 8, self.drop.height() - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.drop.setPixmap(scaled)
        self.info.setText(caption)

    def clear(self):
        """Back to the empty drop zone — a project with no photo yet."""
        self.drop.clear()
        self.drop.setText("Drop a reference photo here")
        self.info.setText("")
        self.hide_pdf()

    def set_pdf_pages(self, count: int):
        self._suppress = True
        self.page_combo.clear()
        for i in range(count):
            self.page_combo.addItem(f"Page {i + 1}", i)
        self.pdf_row.setVisible(count > 0)
        self._suppress = False

    def hide_pdf(self):
        self.pdf_row.setVisible(False)

    def set_pdf_state(self, page_index: int, dpi: int):
        """Restore the saved page + DPI selection without re-emitting ``page_changed``.

        Call *after* :meth:`set_pdf_pages` has populated the combo. The ``_suppress``
        guard keeps a session restore from triggering a spurious page reload.
        """
        self._suppress = True
        idx = self.page_combo.findData(int(page_index))
        if idx >= 0:
            self.page_combo.setCurrentIndex(idx)
        self.dpi.setValue(int(dpi))
        self._suppress = False

    def current_page(self) -> int:
        data = self.page_combo.currentData()
        return int(data) if data is not None else 0

    def current_dpi(self) -> int:
        return int(self.dpi.value())

    # -- internals ----------------------------------------------------------
    def _on_pdf_change(self):
        if not self._suppress:
            self.page_changed.emit()

    def _accept_path(self, path: str) -> Optional[str]:
        return path if path and is_supported(path) else None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if self._accept_path(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = self._accept_path(url.toLocalFile())
            if path:
                self.file_chosen.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()
