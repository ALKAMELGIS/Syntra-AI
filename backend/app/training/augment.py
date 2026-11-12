"""
Geometric augmentation for square tiles: 90/180/270 rotations and flips.
Images are HxWxC arrays; points are normalized (x, y) in [0, 1] with y down.

Ported verbatim (pure NumPy, no GIS dependency) from the GeoSyntra AI QGIS
plugin's core/augment.py.
"""
from __future__ import annotations

import numpy as np

# name -> (image op, point op)
_OPS = {
    "orig":   (lambda im: im,                    lambda p: p),
    "rot90":  (lambda im: np.rot90(im, k=-1),    lambda p: np.c_[1.0 - p[:, 1], p[:, 0]]),   # clockwise
    "rot180": (lambda im: np.rot90(im, k=2),     lambda p: np.c_[1.0 - p[:, 0], 1.0 - p[:, 1]]),
    "rot270": (lambda im: np.rot90(im, k=1),     lambda p: np.c_[p[:, 1], 1.0 - p[:, 0]]),    # counter-clockwise
    "fliplr": (lambda im: im[:, ::-1],           lambda p: np.c_[1.0 - p[:, 0], p[:, 1]]),
    "flipud": (lambda im: im[::-1, :],           lambda p: np.c_[p[:, 0], 1.0 - p[:, 1]]),
}

DEFAULT_VARIANTS = ["orig", "rot90", "rot180", "rot270", "fliplr"]
ALL_VARIANTS = list(_OPS.keys())


def augment_image(img: np.ndarray, op: str) -> np.ndarray:
    return np.ascontiguousarray(_OPS[op][0](img))


def augment_points(pts, op: str) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return _OPS[op][1](pts)
