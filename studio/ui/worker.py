"""Background worker: solving and composing run here so the UI never freezes.

The worker owns the single ``Pipeline`` (and therefore its stage caches),
lives on a dedicated QThread, and communicates only via signals. Unlike
bargue there is no "has source" gate — the board is a valid document with no
photo loaded, so every request renders.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, Slot

from ..errors import StudioError
from ..params import Params
from ..pipeline import Pipeline
from ..pipeline.loader import load


@dataclass
class RenderRequest:
    """What MainWindow hands the worker each render: a Params snapshot."""

    params: Params


class Worker(QObject):
    source_loaded = Signal(object)         # LoadedImage
    finished = Signal(object)              # Result
    failed = Signal(str)                   # message
    progress = Signal(str, float)          # label, fraction 0..1

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline()

    @property
    def has_source(self) -> bool:
        return self.pipeline.has_source

    @Slot(str, int, int)
    def load_source(self, path: str, page_index: int, dpi: int) -> None:
        try:
            loaded = load(path, page_index, dpi)
            self.pipeline.set_source(loaded)
            self.source_loaded.emit(loaded)
        except StudioError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            self.failed.emit(f"Could not load file: {exc}")

    @Slot(object)
    def process(self, req: RenderRequest) -> None:
        def report(label: str, frac: float) -> None:
            self.progress.emit(label, frac)

        try:
            self.finished.emit(self.pipeline.run(req.params, progress=report))
        except StudioError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            self.failed.emit(f"Unexpected processing error: {exc}")
