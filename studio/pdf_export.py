"""To-scale PDF export: place a rendered palette board on a page sized in inches.

Reuses the already-present PyMuPDF (``fitz``) — no new dependency. The page is
exactly the physical canvas size so a print transfers / compares 1:1; the caller
renders the raster at print DPI and hands it in as PNG bytes.
"""
from __future__ import annotations

PT_PER_INCH = 72.0
MAX_RASTER_EDGE = 8192  # cap the intermediate render so a huge canvas can't blow memory


def raster_scale(scene_w: int, scene_h: int, margin: int, ipp: float, dpi: float = 300.0) -> float:
    """The render scale for a to-scale PDF: ``dpi`` dots per physical inch, capped so
    the intermediate raster's longest edge never exceeds ``MAX_RASTER_EDGE`` px (a big
    canvas then simply prints at a proportionally lower DPI instead of exhausting RAM)."""
    longest = float(max(scene_w, scene_h) + 2 * max(0, int(margin)))
    scale = float(ipp) * float(dpi)
    if longest * scale > MAX_RASTER_EDGE:
        scale = MAX_RASTER_EDGE / longest
    return scale


def write_scale_pdf(png_bytes: bytes, w_in: float, h_in: float, path: str) -> None:
    """Write a single-page PDF of ``w_in × h_in`` inches with the image full-bleed."""
    import fitz  # PyMuPDF

    w_pt, h_pt = float(w_in) * PT_PER_INCH, float(h_in) * PT_PER_INCH
    if w_pt <= 0 or h_pt <= 0:
        raise ValueError("Canvas size must be positive to export a to-scale PDF.")
    doc = fitz.open()
    try:
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(fitz.Rect(0, 0, w_pt, h_pt), stream=png_bytes)
        doc.save(path)
    finally:
        doc.close()
