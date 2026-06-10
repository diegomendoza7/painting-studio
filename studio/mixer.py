"""pymixbox wrapper: pigment-plausible mixing in Mixbox's latent space.

All Mixbox calls live here so the engine is swappable (its CC BY-NC 4.0
license is fine for personal use but not for distribution in a commercial
product). The import is lazy and guarded so a missing dependency surfaces as
a clear, user-facing :class:`SolverError`.

Multi-paint mixing follows Mixbox's documented method: convert each paint to
its 7-float latent, average the latents by the mix weights, convert back.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .color import RGB, hex_to_rgb
from .errors import SolverError

_mixbox = None
_latents: Dict[RGB, tuple] = {}


def _mb():
    global _mixbox
    if _mixbox is None:
        try:
            import mixbox
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SolverError(
                "Pigment mixing needs pymixbox. Install it with: pip install pymixbox"
            ) from exc
        _mixbox = mixbox
    return _mixbox


def latent(rgb: RGB) -> tuple:
    z = _latents.get(rgb)
    if z is None:
        z = tuple(_mb().rgb_to_latent(rgb))
        _latents[rgb] = z
    return z


def latent_hex(s: str) -> tuple:
    return latent(hex_to_rgb(s))


def mix(latents: Sequence[tuple], weights: Sequence[float]) -> RGB:
    """Weighted pigment mix -> sRGB. ``weights`` should sum to ~1."""
    mb = _mb()
    n = mb.LATENT_SIZE
    z = [0.0] * n
    for zi, w in zip(latents, weights):
        for k in range(n):
            z[k] += w * zi[k]
    return tuple(mb.latent_to_rgb(z))  # type: ignore[return-value]


def lerp(rgb1: RGB, rgb2: RGB, t: float) -> RGB:
    """Pigment-mix interpolation: t=0 -> rgb1, t=1 -> rgb2."""
    return tuple(_mb().lerp(rgb1, rgb2, t))  # type: ignore[return-value]
