"""
Vegetation and tree analysis from spectral indices (no trained model).

* **NDVI** = (NIR - Red) / (NIR + Red) when the image has a near-infrared
  band (most satellite multispectral products). DN values are first converted
  to radiance with the product's calibration gains when its metadata provides
  them (SuperView-style `<AbsCalibrationGain>`), because bands with different
  gains make a raw-DN NDVI biased.
* **VARI** = (G - R) / (G + R - B) for RGB-only images (drones). Weaker than
  NDVI and sensitive to lighting; the manifest says which index was used.

Output: vegetation polygons (exterior + holes) with ground area, mean index
value and a density class, plus a per-polygon count of *tree crown
candidates* (local maxima of the smoothed index at least ~3 m apart). The
count is an estimate and is labelled as one; at pixel sizes above 2.5 m
crowns aren't resolved and no count is given.

Large windows are read decimated (pixels averaged) to stay within memory;
the manifest records the effective resolution actually analysed.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

# 1-based (red, nir) by (sensor prefix, band count). SVN3 order established
# from the product (see raster_prep.SENSOR_RGB); NIR1 = band 4.
SENSOR_RED_NIR: dict[tuple[str, int], tuple[int, int]] = {
    ("SVN3", 8): (3, 4),
    ("WV02", 8): (5, 7),
    ("WV03", 8): (5, 7),
    ("PHR", 4): (1, 4),
    ("PNEO", 4): (1, 4),
}
DEFAULT_RED_NIR: dict[int, tuple[int, int]] = {4: (3, 4), 5: (3, 4), 8: (5, 7)}

# Band-averaged solar irradiance (W m-2 um-1) for the red and NIR bands.
# Radiance NDVI is biased low because the sun delivers less energy in the NIR
# than in the red; dividing each band's radiance by its ESUN gives a ratio of
# top-of-atmosphere reflectances, and sun elevation and Earth-Sun distance
# cancel out of the NDVI. SVN3 has no published ESUN table we could find, so
# the WorldView-3 values for the same spectral bands (red 630-690 nm, NIR1
# 770-895 nm) are used as an approximation. On the SVN3 scene this moves bare
# sand from NDVI -0.16 (radiance) to about 0.01 (physically expected) and
# vegetation peaks from ~0.35 to ~0.54.
SENSOR_ESUN: dict[str, dict[str, float]] = {
    "SVN3": {"red": 1555.0, "nir": 1069.0},
    "WV02": {"red": 1559.0, "nir": 1069.0},
    "WV03": {"red": 1555.0, "nir": 1069.0},
}

DEFAULT_THRESHOLD = {"NDVI": 0.20, "VARI": 0.10}
TREE_MIN_SPACING_M = 3.0
# Crowns of 3-8 m are only resolvable at about 1 m or finer. Tested on a
# 2 m SuperView MUX scene: a 117 ha continuous intertidal patch produced
# ~10,000 "maxima" that were just the spacing grid of a flat plateau, so
# counts are withheld above 1 m and each maximum must also stand out.
TREE_MAX_PIXEL_M = 1.0
TREE_MIN_PROMINENCE = 0.05


class VegetationError(ValueError):
    pass


def choose_index_bands(count: int, sensor: str | None, colorinterp: list[str] | None = None,
                       override: list[int] | None = None) -> tuple[str, dict[str, int]]:
    """Return (index name, band roles) e.g. ("NDVI", {"red": 3, "nir": 4})."""
    if override:
        if len(override) == 2 and all(1 <= b <= count for b in override):
            return "NDVI", {"red": override[0], "nir": override[1]}
        if len(override) == 3 and all(1 <= b <= count for b in override):
            return "VARI", {"red": override[0], "green": override[1], "blue": override[2]}
        raise VegetationError(f"index_bands must be [red, nir] or [red, green, blue] band numbers between 1 and {count}.")
    if sensor:
        for (prefix, n), (r, nir) in SENSOR_RED_NIR.items():
            if n == count and sensor.upper().startswith(prefix):
                return "NDVI", {"red": r, "nir": nir}
    ci = [c.lower() for c in (colorinterp or [])]
    if count >= 4 and count in DEFAULT_RED_NIR:
        r, nir = DEFAULT_RED_NIR[count]
        return "NDVI", {"red": r, "nir": nir}
    if count >= 3:
        if all(x in ci for x in ("red", "green", "blue")):
            return "VARI", {"red": ci.index("red") + 1, "green": ci.index("green") + 1, "blue": ci.index("blue") + 1}
        return "VARI", {"red": 1, "green": 2, "blue": 3}
    raise VegetationError("Vegetation analysis needs a near-infrared band (NDVI) or at least red, green and blue (VARI).")


def read_calibration_gains(path: str | Path) -> tuple[list[float], list[float]] | None:
    """Per-band gain/bias from a sidecar XML, if the product ships one."""
    p = Path(path)
    for xml in (p.with_suffix(".xml"), p.with_suffix(".XML")):
        try:
            text = xml.read_text(errors="ignore")
        except OSError:
            continue
        g = re.search(r"<AbsCalibrationGain>([^<]+)</AbsCalibrationGain>", text)
        if not g:
            return None
        gains = [float(v) for v in g.group(1).split(",")]
        b = re.search(r"<AbsCalibrationBias>([^<]+)</AbsCalibrationBias>", text)
        bias = [float(v) for v in b.group(1).split(",")] if b else [0.0] * len(gains)
        return gains, bias
    return None


def compute_index(name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "NDVI":
            r, n = bands["red"], bands["nir"]
            idx = (n - r) / (n + r)
        else:
            r, g, b = bands["red"], bands["green"], bands["blue"]
            idx = (g - r) / (g + r - b)
    idx = np.where(np.isfinite(idx), idx, np.nan)
    return np.clip(idx, -1, 1)


def tree_candidates(index: np.ndarray, veg: np.ndarray, pixel_m: float, min_value: float) -> np.ndarray:
    """Boolean mask of local maxima of the smoothed index (crown centres)."""
    import cv2

    if pixel_m > TREE_MAX_PIXEL_M:
        return np.zeros_like(veg, dtype=bool)
    sigma = max(0.6, 1.0 / pixel_m)                       # ~1 m smoothing
    sm = cv2.GaussianBlur(np.nan_to_num(index, nan=-1).astype(np.float32), (0, 0), sigma)
    k = max(3, int(round(TREE_MIN_SPACING_M / pixel_m)) | 1)
    dil = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # prominence: the crown must rise above the lowest value in a ring ~2 crowns wide
    low = cv2.erode(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    peaks = (sm >= dil) & veg & (sm >= min_value) & (sm - low >= TREE_MIN_PROMINENCE)
    # A real crown blurs to a compact peak of a few pixels; a flat patch makes
    # long ridges of equal "maxima" along its edge. Keep compact components
    # only, one point each.
    n, lab, stats, cent = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    out = np.zeros_like(peaks, dtype=bool)
    max_px = max(4, (k * k) // 4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_px:
            cx, cy = cent[i]
            out[int(round(cy)), int(round(cx))] = True
    return out


def run_vegetation(
    path: str,
    out_dir: str | Path,
    model_id: str = "vegetation_index",
    *,
    aoi=None,                                   # app.engine.aoi.AoiInRaster
    threshold: float | None = None,
    index_bands: list[int] | None = None,
    min_area_m2: float = 20.0,
    max_pixels: float = 36e6,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    import cv2
    import rasterio
    from pyproj import Geod, Transformer
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask, shapes
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform

    from app.engine.raster_prep import detect_sensor

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0, 3)

    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise VegetationError("This image has no CRS; vegetation areas can't be measured or mapped.")
        index_name, roles = choose_index_bands(ds.count, detect_sensor(path), [c.name for c in ds.colorinterp], index_bands)
        thr = DEFAULT_THRESHOLD[index_name] if threshold is None else float(threshold)

        c0, r0, w, h = aoi.window if aoi else (0, 0, ds.width, ds.height)
        f = max(1, math.ceil(math.sqrt((w * h) / max_pixels)))
        out_h, out_w = max(1, h // f), max(1, w // f)
        win = Window(c0, r0, w, h)
        order = sorted(set(roles.values()))
        raw = ds.read(indexes=order, window=win, out_shape=(len(order), out_h, out_w),
                      resampling=Resampling.average if f > 1 else Resampling.nearest).astype(np.float32)
        wt = window_transform(win, ds.transform)
        sx, sy = w / out_w, h / out_h
        tr = rasterio.Affine(wt.a * sx, wt.b * sy, wt.c, wt.d * sx, wt.e * sy, wt.f)   # scale pixels, keep origin
        crs = ds.crs
        units_m = float(crs.linear_units_factor[1]) if crs.is_projected else None
        nodata = ds.nodata

    gains = read_calibration_gains(path)
    sensor = detect_sensor(path)
    esun = SENSOR_ESUN.get((sensor or "")[:4]) if gains and index_name == "NDVI" else None
    bands = {}
    for role, b in roles.items():
        arr = raw[order.index(b)]
        if gains and b <= len(gains[0]):
            arr = arr * gains[0][b - 1] + gains[1][b - 1]
            if esun:
                arr = arr / esun[role]
        bands[role] = arr
    radiometry = "toa_reflectance_ratio" if esun else ("radiance" if gains else "dn")
    nd = 0 if nodata is None else nodata
    valid = ~np.all(raw == nd, axis=0)
    if aoi is not None:
        valid &= ~geometry_mask([mapping(aoi.geom)], out_shape=(out_h, out_w), transform=tr)

    index = compute_index(index_name, bands)
    index[~valid] = np.nan
    veg = (index >= thr) & valid
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    if progress_cb:
        progress_cb(1, 3)

    if crs.is_projected:
        pixel_m = abs(tr.a) * units_m
        pixel_area_m2 = abs(tr.a * tr.e) * units_m ** 2
    else:
        lat = tr.d * 0 + tr.e * (out_h / 2) + tr.f
        pixel_m = math.sqrt(abs(tr.a) * 111_320 * math.cos(math.radians(lat)) * abs(tr.e) * 111_320)
        pixel_area_m2 = pixel_m ** 2

    trees = tree_candidates(index, veg, pixel_m, thr + 0.1)
    n_labels, labels = cv2.connectedComponents(veg.astype(np.uint8), connectivity=8)
    flat = labels.ravel()
    ndvi_sum = np.bincount(flat, weights=np.nan_to_num(index).ravel(), minlength=n_labels)
    px_count = np.bincount(flat, minlength=n_labels)
    tree_count = np.bincount(flat, weights=trees.ravel().astype(np.float64), minlength=n_labels)

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")
    features, raw_count = [], 0
    resolve_trees = pixel_m <= TREE_MAX_PIXEL_M
    for geom, value in shapes(labels.astype(np.int32), mask=veg, transform=tr, connectivity=8):
        lab = int(value)
        if lab == 0:
            continue
        raw_count += 1
        poly = shape(geom).simplify(abs(tr.a) / 2, preserve_topology=True)   # half a pixel
        poly_wgs = shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), poly)
        area = abs(geod.geometry_area_perimeter(poly_wgs)[0])
        if area < min_area_m2:
            continue
        mean_idx = float(ndvi_sum[lab] / max(px_count[lab], 1))
        props = {
            "class": "vegetation",
            "density": "dense" if mean_idx >= thr + 0.2 else "sparse",
            f"mean_{index_name.lower()}": round(mean_idx, 3),
            "area_m2": round(area, 1),
            "area_ha": round(area / 10_000, 4),
        }
        if resolve_trees:
            props["tree_candidates"] = int(tree_count[lab])
        features.append({"type": "Feature", "geometry": mapping(poly_wgs), "properties": props})
    if progress_cb:
        progress_cb(2, 3)

    total_area = sum(f_["properties"]["area_m2"] for f_ in features)
    valid_area = float(valid.sum()) * pixel_area_m2
    summary = {
        "index": index_name,
        "threshold": thr,
        "vegetation_area_ha": round(total_area / 10_000, 3),
        "analysed_area_ha": round(valid_area / 10_000, 3),
        "cover_pct": round(100 * total_area / valid_area, 2) if valid_area else 0.0,
        "tree_candidates": int(sum(f_["properties"].get("tree_candidates", 0) for f_ in features)) if resolve_trees else None,
        "effective_gsd_m": round(pixel_m, 3),
    }

    geojson_path = out_dir / f"{model_id}.geojson"
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    manifest = {
        "model_id": model_id,
        "count": len(features),
        "crs": "EPSG:4326",
        "source_crs": crs.to_string(),
        "bands": roles,
        "calibration_applied": bool(gains),
        "radiometry": radiometry,
        "decimation_factor": f,
        "min_area_m2": min_area_m2,
        "note": (
            "Positive NDVI means chlorophyll, not necessarily trees: on coasts, intertidal algal mats and "
            "seagrass in very shallow water also score as vegetation."
        ),
        "tree_count_note": (
            "Local maxima of the smoothed index at least 3 m apart that stand out from their surroundings: "
            "an estimate of crown centres, not a validated count."
            if resolve_trees else f"Not computed: pixels are {pixel_m:.1f} m, too coarse to separate crowns."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **summary,
    }
    manifest_path = out_dir / f"{model_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if progress_cb:
        progress_cb(3, 3)

    return {
        "model_id": model_id,
        "device_used": "cpu",
        "device_reason": "index computation",
        "tiles_processed": 1,
        "raw_detection_count": raw_count,
        "final_detection_count": len(features),
        "duration_seconds": round(time.time() - t0, 3),
        "geojson_path": str(geojson_path),
        "manifest_path": str(manifest_path),
        "summary": summary,
    }
