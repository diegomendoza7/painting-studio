"""Color math: sRGB <-> LAB/LCh (D65), CIEDE2000, Beer-Lambert dilution.

Two parallel implementations on purpose: vectorized numpy versions for image
work (palette extraction, swatch fills) and plain-``math`` scalar versions for
the mix solver's hot loop, where per-call numpy overhead would dominate the
actual arithmetic (a few thousand evaluations per target).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

RGB = Tuple[int, int, int]

# sRGB (D65) <-> XYZ matrices and white point.
_M = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_INV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS = (6.0 / 29.0) ** 3
_KAPPA_DEN = 3.0 * (6.0 / 29.0) ** 2  # f(t) linear-segment divisor


# --- hex helpers ------------------------------------------------------------
def hex_to_rgb(s: str) -> RGB:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# --- vectorized (numpy) -----------------------------------------------------
def srgb_to_linear(c01: np.ndarray) -> np.ndarray:
    c = np.asarray(c01, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB uint8 (...,3) -> LAB float64 (...,3)."""
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    xyz = (lin @ _M.T) / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), xyz / _KAPPA_DEN + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab) -> np.ndarray:
    """LAB (...,3) -> sRGB uint8 (...,3), clipped into gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    cube = f**3
    xyz = np.where(cube > _EPS, cube, _KAPPA_DEN * (f - 4.0 / 29.0)) * _WHITE
    lin = xyz @ _M_INV.T
    return np.clip(np.round(linear_to_srgb(lin) * 255.0), 0, 255).astype(np.uint8)


# --- scalar (pure math; solver hot path) ------------------------------------
def _lin1(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb1(v: float) -> float:
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def linear3(rgb: RGB) -> Tuple[float, float, float]:
    return (_lin1(rgb[0]), _lin1(rgb[1]), _lin1(rgb[2]))


def _f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > _EPS else t / _KAPPA_DEN + 4.0 / 29.0


def lab_from_linear(lin: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = lin
    fx = _f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047)
    fy = _f(0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
    fz = _f((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def lab(rgb: RGB) -> Tuple[float, float, float]:
    return lab_from_linear(linear3(rgb))


def lstar(rgb: RGB) -> float:
    return lab(rgb)[0]


def lch(rgb: RGB) -> Tuple[float, float, float]:
    """(L*, chroma, hue degrees 0..360)."""
    L, a, b = lab(rgb)
    return (L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0)


def dilute(rgb: RGB, d: float) -> RGB:
    """Beer-Lambert wash: paint at optical thickness ``d`` over white paper.

    d=1 is masstone, d->0 approaches paper white. Transmittance through a
    thinner layer is the masstone transmittance raised to ``d`` (in linear RGB).
    """
    lin = linear3(rgb)
    return tuple(int(round(_srgb1(c**d) * 255.0)) for c in lin)  # type: ignore[return-value]


def temperature(rgb: RGB) -> str:
    """Painterly warm/cool: "W", "C", or "" for near-neutrals."""
    L, c, h = lch(rgb)
    if c < 12.0:
        return ""
    return "W" if (h < 112.0 or h >= 320.0) else "C"


def delta_e_2000(lab1, lab2) -> float:
    """CIEDE2000 (Sharma et al. formulation) between two LAB tuples."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    c7 = Cbar**7
    G = 0.5 * (1.0 - math.sqrt(c7 / (c7 + 25.0**7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0 if (a2p or b2) else 0.0

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    if C1p * C2p == 0.0:
        hbp = h1p + h2p
    else:
        s = h1p + h2p
        if abs(h1p - h2p) > 180.0:
            hbp = (s + 360.0) / 2.0 if s < 360.0 else (s - 360.0) / 2.0
        else:
            hbp = s / 2.0
    T = (
        1.0
        - 0.17 * math.cos(math.radians(hbp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hbp))
        + 0.32 * math.cos(math.radians(3.0 * hbp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hbp - 63.0))
    )
    dtheta = 30.0 * math.exp(-(((hbp - 275.0) / 25.0) ** 2))
    cbp7 = Cbp**7
    Rc = 2.0 * math.sqrt(cbp7 / (cbp7 + 25.0**7))
    Sl = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / math.sqrt(20.0 + (Lbp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc
    return math.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def delta_e_rgb(rgb1: RGB, rgb2: RGB) -> float:
    return delta_e_2000(lab(rgb1), lab(rgb2))
