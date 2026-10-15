"""Background worker: solving and composing run here so the UI never freezes.

The worker owns the single ``Pipeline`` (and therefore its stage caches),
lives on a dedicated QThread, and communicates only via signals. Unlike
bargue there is no "has source" gate — the board is a valid document with no
photo loaded, so every request renders.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..errors import StudioError
from ..params import Params
from ..pipeline import Pipeline
from ..pipeline.loader import load


@dataclass
class RenderRequest:
    """What MainWindow hands the worker each render: a Params snapshot."""

    params: Params
    revision: int = 0
    source_token: int = 0


class Worker(QObject):
    source_loaded = Signal(object, int)
    load_failed = Signal(str, int)
    finished = Signal(object, int)
    failed = Signal(str, int)
    progress = Signal(str, float)          # label, fraction 0..1

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline()
        self.source_token = 0

    @property
    def has_source(self) -> bool:
        return self.pipeline.has_source

    @Slot(str, int, int, int)
    def load_source(self, path: str, page_index: int, dpi: int, token: int = 0) -> None:
        if QThread.currentThread().isInterruptionRequested():
            return
        try:
            loaded = load(path, page_index, dpi)
            self.source_loaded.emit(loaded, token)
        except StudioError as exc:
            self.load_failed.emit(str(exc), token)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            self.load_failed.emit(f"Could not load file: {exc}", token)

    @Slot(object, int)
    def adopt_source(self, loaded, token: int) -> None:
        self.pipeline.set_source(loaded)
        self.source_token = token

    @Slot(int)
    def clear_source(self, token: int = 0) -> None:
        """Forget the photo — a new, empty project. Queued like every other
        request, so it can never land in the middle of a render."""
        self.pipeline.clear_source()
        self.source_token = token

    @Slot(object)
    def process(self, req: RenderRequest) -> None:
        if (req.source_token != self.source_token
                or QThread.currentThread().isInterruptionRequested()):
            self.finished.emit(None, req.revision)
            return
        def report(label: str, frac: float) -> None:
            self.progress.emit(label, frac)

        try:
            self.finished.emit(self.pipeline.run(req.params, progress=report), req.revision)
        except StudioError as exc:
            self.failed.emit(str(exc), req.revision)
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            self.failed.emit(f"Unexpected processing error: {exc}", req.revision)
