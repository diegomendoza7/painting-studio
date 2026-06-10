"""Stage 1 — extract the photo's dominant colors as auto targets.

A small seeded k-means (k-means++ init) over the working image's pixels in
LAB space — perceptual distances, so clusters land on the colors an artist
would actually pick. Deterministic by construction: fixed RNG seed, fixed
subsample stride, clusters ordered by population then lightness.
"""
from __future__ import annotations

import numpy as np

from ..color import lab_to_rgb, rgb_to_hex, rgb_to_lab
from ..models import Target

_SEED = 42
_MAX_SAMPLES = 20000
_MAX_ITER = 25


def _kmeans(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded k-means++ over (N,3) float samples -> (centers (k,3), labels (N,))."""
    rng = np.random.default_rng(_SEED)
    n = samples.shape[0]
    # k-means++ init.
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = samples[rng.integers(n)]
    d2 = np.sum((samples - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        centers[i] = samples[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((samples - centers[i]) ** 2, axis=1))
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(_MAX_ITER):
        dists = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for i in range(k):
            pts = samples[labels == i]
            if len(pts):
                centers[i] = pts.mean(axis=0)
    return centers, labels


def extract_targets(rgb: np.ndarray, n: int) -> tuple[Target, ...]:
    """Dominant colors of an HxWx3 uint8 image as auto ``Target``s.

    Ordered by cluster population (the colors that cover the most canvas
    first), ties broken dark-to-light for stability.
    """
    n = max(1, int(n))
    pixels = rgb.reshape(-1, 3)
    stride = max(1, len(pixels) // _MAX_SAMPLES)
    samples = rgb_to_lab(pixels[::stride].astype(np.float64))
    k = min(n, len(samples))
    centers, labels = _kmeans(samples, k)
    counts = np.bincount(labels, minlength=k)
    order = sorted(range(k), key=lambda i: (-counts[i], centers[i][0]))
    out = []
    for rank, i in enumerate(order):
        srgb = lab_to_rgb(centers[i])
        out.append(
            Target(hex=rgb_to_hex(tuple(int(c) for c in srgb)), label=f"Photo {rank + 1}", source="auto")
        )
    return tuple(out)
