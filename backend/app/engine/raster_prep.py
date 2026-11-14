"""
Getting a real satellite product into the shape a detector expects.

Detectors are trained on 8-bit RGB. Satellite GeoTIFFs often aren't:

* **Bit depth.** Pan and multispectral products are usually 11-16 bit
  (uint16). Feeding raw DN values to a model trained on 0-255 gives garbage.
  We compute one global percentile stretch per band from a decimated read of
  the whole scene, and apply that same stretch to every tile, so
  neighbouring tiles look alike and cross-tile merging stays consistent.
* **Band order.** "The first three bands" is RGB only by accident. GDAL
  colour interpretation is used when the file declares it; otherwise a
  sensor-specific or band-count default; and the caller can always override.
* **No-data.** Scenes clipped to a footprint can be mostly empty (a
  SuperView Neo-3 coastal scene tested here was 72% zeros). Tiles with no
  valid pixels are skipped instead of being sent to the model.

rasterio is imported lazily, as elsewhere in the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import math

import numpy as np

# Known band layouts, 1-based (R, G, B). Keyed by (sensor prefix, band count).
# SVN3 8-band order was established from the product itself: per-band
# radiance over open sea falls b1~b5 > b2 > b6 > b3 > b7 > b4 > b8 (water
# absorbs more at longer wavelengths), giving coastal/blue, green, yellow,
# red=b3, red-edge=b7, NIR=b4/b8; true colour b3/b2/b1 was then checked
# visually (turquoise shallows, beige desert).
SENSOR_RGB: dict[tuple[str, int], tuple[int, int, int]] = {
    ("SVN3", 8): (3, 2, 1),
    ("WV02", 8): (5, 3, 2),     # coastal, blue, green, yellow, red, red-edge, nir1, nir2
    ("WV03", 8): (5, 3, 2),
    ("PHR", 4): (1, 2, 3),      # Pleiades primary product: R, G, B, NIR
    ("PNEO", 4): (1, 2, 3),
}

# Fallback by band count when nothing else is known.
DEFAULT_RGB: dict[int, tuple[int, int, int]] = {
    1: (1, 1, 1),               # panchromatic -> grey
    2: (1, 1, 1),
    3: (1, 2, 3),
    4: (3, 2, 1),               # B, G, R, NIR is the most common 4-band layout
    8: (5, 3, 2),               # WorldView-style 8-band
}


@dataclass
class RgbPlan:
    bands: tuple[int, int, int]         # 1-based band indexes for R, G, B
    source: str                         # "override" | "colorinterp" | "description" | "sensor" | "default"
    lo: list[float] = field(default_factory=list)   # per output channel
    hi: list[float] = field(default_factory=list)
    nodata: float | None = None
    dtype: str = "uint8"
    lut: np.ndarray | None = None       # (256, 3) colour table for palette images

    def to_dict(self) -> dict:
        return {"bands": list(self.bands), "source": self.source, "lo": self.lo, "hi": self.hi,
                "nodata": json_safe(self.nodata), "dtype": self.dtype, "palette": self.lut is not None}


def json_safe(v):
    """NaN / inf are valid no-data values but not valid JSON."""
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def valid_pixels(data, nodata: float | None) -> np.ndarray:
    """(C, H, W) array or masked array -> (H, W) bool of pixels with data.

    A pixel is empty when GDAL masks it (declared no-data, alpha band or an
    internal mask, via masked reads), when any band is NaN, or, for files
    that declare no no-data value, when every band is 0 (how clipped scene
    footprints are usually filled).
    """
    mask = np.ma.getmaskarray(data)
    arr = np.ma.getdata(data)
    valid = ~mask.any(axis=0) if mask.ndim == 3 else np.ones(arr.shape[1:], bool)
    if np.issubdtype(arr.dtype, np.floating):
        valid &= ~np.isnan(arr).any(axis=0)
    if nodata is None:
        valid &= ~(arr == 0).all(axis=0)
    elif not (isinstance(nodata, float) and math.isnan(nodata)):
        valid &= ~(arr == nodata).all(axis=0)
    return valid


def choose_rgb_bands(count: int, colorinterp: list[str] | None = None, descriptions: list[str | None] | None = None,
                     sensor: str | None = None, override: list[int] | None = None) -> tuple[tuple[int, int, int], str]:
    if override:
        if len(override) not in (1, 3) or any(b < 1 or b > count for b in override):
            raise ValueError(f"rgb_bands must be 1 or 3 band numbers between 1 and {count}.")
        b = tuple(override) * 3 if len(override) == 1 else tuple(override)
        return b[:3], "override"

    ci = [c.lower() for c in (colorinterp or [])]
    if all(x in ci for x in ("red", "green", "blue")):
        return (ci.index("red") + 1, ci.index("green") + 1, ci.index("blue") + 1), "colorinterp"

    desc = [(d or "").lower() for d in (descriptions or [])]
    def find(word):
        hits = [i + 1 for i, d in enumerate(desc) if d.strip() == word or d.startswith(word + " ") or d.endswith(" " + word)]
        return hits[0] if hits else None
    r, g, b = find("red"), find("green"), find("blue")
    if r and g and b:
        return (r, g, b), "description"

    if sensor:
        for (prefix, n), bands in SENSOR_RGB.items():
            if n == count and sensor.upper().startswith(prefix):
                return bands, "sensor"

    if count in DEFAULT_RGB:
        return DEFAULT_RGB[count], "default"
    return ((1, 2, 3) if count >= 3 else (1, 1, 1)), "default"


def detect_sensor(path: str | Path) -> str | None:
    """Best-effort sensor id from the file name or a sidecar XML (SatelliteID)."""
    p = Path(path)
    name = p.name.upper()
    for prefix in {k[0] for k in SENSOR_RGB}:
        if name.startswith(prefix) or f"_{prefix}" in name:
            return prefix
    for xml in (p.with_suffix(".xml"), p.with_suffix(".XML")):
        try:
            text = xml.read_text(errors="ignore")
        except OSError:
            continue
        import re
        m = re.search(r"<SatelliteID>\s*([^<\s]+)\s*</SatelliteID>", text)
        if m:
            return m.group(1).upper()
    return None


def compute_stretch(sample: np.ndarray, nodata: float | None, low_pct: float = 2.0, high_pct: float = 98.0,
                    valid: np.ndarray | None = None) -> tuple[list[float], list[float]]:
    """Per-channel percentile stretch from a (C, H, W) sample, ignoring empty pixels."""
    if valid is None:
        valid = valid_pixels(sample, nodata)
    sample = np.ma.getdata(sample)
    lo, hi = [], []
    for ch in sample:
        v = ch[valid]
        if v.size == 0:
            lo.append(0.0); hi.append(255.0)
            continue
        a, b = np.percentile(v, [low_pct, high_pct])
        if b <= a:
            b = a + 1
        lo.append(float(a)); hi.append(float(b))
    return lo, hi


def to_uint8(tile_chw: np.ndarray, lo: list[float], hi: list[float]) -> np.ndarray:
    """(C, H, W) any dtype -> (H, W, 3) uint8 using a fixed global stretch."""
    out = np.empty((tile_chw.shape[1], tile_chw.shape[2], 3), dtype=np.uint8)
    tile_chw = np.ma.getdata(tile_chw)
    for i in range(3):
        ch = tile_chw[min(i, tile_chw.shape[0] - 1)].astype(np.float32)
        v = np.clip((ch - lo[i]) / (hi[i] - lo[i]) * 255.0, 0, 255)
        out[..., i] = np.nan_to_num(v, nan=0.0).astype(np.uint8)
    return out


def tile_has_data(tile_chw: np.ndarray, nodata: float | None, min_fraction: float = 0.005,
                  valid: np.ndarray | None = None) -> bool:
    if valid is None:
        valid = valid_pixels(tile_chw, nodata)
    return float(np.mean(valid)) >= min_fraction


def read_rgb(ds, plan: "RgbPlan", window=None, out_shape: tuple[int, int] | None = None):
    """The one way pixels are read for models and previews.

    -> (H, W, 3) uint8 RGB with empty pixels set to 0, and the (H, W) valid mask.
    Honours GDAL masks (no-data, alpha, internal mask), NaN and palettes.
    """
    from rasterio.enums import Resampling

    uniq = sorted(set(plan.bands))
    kw = {"indexes": uniq, "window": window, "masked": True}
    if out_shape is not None:
        kw.update(out_shape=(len(uniq), *out_shape), resampling=Resampling.nearest)
    raw = ds.read(**kw)
    ordered = np.ma.stack([raw[uniq.index(b)] for b in plan.bands])
    valid = valid_pixels(ordered, plan.nodata)
    if plan.lut is not None:
        idx = np.clip(np.ma.getdata(ordered[0]).astype(np.int64), 0, 255)
        rgb = plan.lut[idx]
    else:
        rgb = to_uint8(ordered, plan.lo, plan.hi)
    rgb[~valid] = 0
    return rgb, valid


def plan_rgb(path: str | Path, override: list[int] | None = None, sample_dim: int = 2048) -> RgbPlan:
    """Choose bands and a global stretch for a raster (one decimated read)."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        colorinterp = [c.name for c in ds.colorinterp]
        nodata = ds.nodata
        dtype = str(ds.dtypes[0])
        lut = None
        if ds.count >= 1 and colorinterp and colorinterp[0] == "palette" and override is None:
            # classified / indexed image: show the colours, not the class numbers
            try:
                cmap = ds.colormap(1)
                lut = np.zeros((256, 3), dtype=np.uint8)
                for k, rgba in cmap.items():
                    if 0 <= k < 256:
                        lut[k] = rgba[:3]
            except ValueError:
                lut = None
        if lut is not None:
            return RgbPlan(bands=(1, 1, 1), source="palette", lo=[0.0] * 3, hi=[255.0] * 3,
                           nodata=nodata, dtype=dtype, lut=lut)
        bands, source = choose_rgb_bands(ds.count, colorinterp, list(ds.descriptions), detect_sensor(path), override)
        uniq = sorted(set(bands))
        scale = min(1.0, sample_dim / max(ds.width, ds.height))
        shape = (len(uniq), max(1, round(ds.height * scale)), max(1, round(ds.width * scale)))
        # nearest keeps no-data as no-data (bilinear would smear it into the edges)
        sample = ds.read(indexes=uniq, out_shape=shape, resampling=Resampling.nearest, masked=True)

    ordered = np.ma.stack([sample[uniq.index(b)] for b in bands])
    if dtype == "uint8" and source != "override":
        lo, hi = [0.0, 0.0, 0.0], [255.0, 255.0, 255.0]   # already display-ready: leave as is
    else:
        lo, hi = compute_stretch(ordered, nodata, valid=valid_pixels(ordered, nodata))
    return RgbPlan(bands=bands, source=source, lo=lo, hi=hi, nodata=nodata, dtype=dtype)
