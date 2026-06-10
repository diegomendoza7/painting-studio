"""Stage 0 — load any supported file to a normalized RGB working image.

Supports JPG/PNG/WEBP/BMP/TIFF via Pillow and multi-page PDF via PyMuPDF
(rasterized at a chosen DPI). Alpha is flattened — the reference photo is only
sampled for color here, never matted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..errors import LoadError

RASTER_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


@dataclass
class LoadedImage:
    rgb: np.ndarray                 # HxWx3 uint8
    source_path: str
    page_index: int = 0

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_pdf(path: str) -> bool:
    return _ext(path) in PDF_EXTS


def is_supported(path: str) -> bool:
    return _ext(path) in RASTER_EXTS | PDF_EXTS


def count_pdf_pages(path: str) -> int:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise LoadError(
            "PDF support needs PyMuPDF. Install it with: pip install pymupdf"
        ) from exc
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception as exc:
        raise LoadError(f"Could not open PDF: {exc}") from exc


def _rasterize_pdf_page(path: str, page_index: int, dpi: int) -> np.ndarray:
    import fitz

    try:
        with fitz.open(path) as doc:
            if page_index < 0 or page_index >= doc.page_count:
                raise LoadError(
                    f"PDF page {page_index + 1} is out of range "
                    f"(document has {doc.page_count})."
                )
            page = doc.load_page(page_index)
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA -> RGB
                arr = arr[:, :, :3]
            elif pix.n == 1:  # grayscale -> RGB
                arr = np.repeat(arr, 3, axis=2)
            return np.ascontiguousarray(arr[:, :, :3])
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Failed to rasterize PDF: {exc}") from exc


def _load_raster(path: str) -> np.ndarray:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise LoadError(f"Could not read image '{os.path.basename(path)}': {exc}") from exc

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Flatten transparency onto white — the board samples color only.
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.ascontiguousarray(np.asarray(img.convert("RGB")))


def load(path: str, page_index: int = 0, dpi: int = 200) -> LoadedImage:
    """Load a file to a ``LoadedImage`` (not yet normalized)."""
    if not os.path.exists(path):
        raise LoadError(f"File not found: {path}")
    ext = _ext(path)
    if ext in PDF_EXTS:
        rgb = _rasterize_pdf_page(path, page_index, dpi)
        return LoadedImage(rgb=rgb, source_path=path, page_index=page_index)
    if ext in RASTER_EXTS:
        return LoadedImage(rgb=_load_raster(path), source_path=path, page_index=0)
    raise LoadError(
        f"Unsupported file type '{ext}'. Supported: "
        + ", ".join(sorted(RASTER_EXTS | PDF_EXTS))
    )


def normalize(loaded: LoadedImage, max_edge: int) -> LoadedImage:
    """Downscale so the longest edge equals ``max_edge`` (never upscales)."""
    from PIL import Image

    h, w = loaded.rgb.shape[:2]
    if h == 0 or w == 0:
        raise LoadError("Image has zero size.")
    longest = max(h, w)
    if longest <= max_edge:
        return loaded
    scale = max_edge / float(longest)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = Image.fromarray(loaded.rgb).resize((new_w, new_h), Image.LANCZOS)
    return LoadedImage(
        rgb=np.ascontiguousarray(np.asarray(img)),
        source_path=loaded.source_path,
        page_index=loaded.page_index,
    )
