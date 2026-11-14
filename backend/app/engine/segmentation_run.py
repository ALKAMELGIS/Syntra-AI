"""
Run an installed segmentation pack (e.g. building footprints) over a raster
or an AOI and export polygons in WGS84 with their ground area.

Tiles are read at the scale the model was trained at, segmented, written
into one mask for the analysed window, then polygonised once, so a building
cut by a tile edge comes out as one polygon.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

MAX_MASK_PIXELS = 150_000_000      # ~12,000 x 12,000; larger needs an AOI


class SegmentationError(ValueError):
    pass


def ground_gsd_m(path: str) -> float | None:
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        if ds.crs.is_projected:
            return abs(ds.res[0]) * float(ds.crs.linear_units_factor[1])
        cx = (ds.bounds.left + ds.bounds.right) / 2
        cy = (ds.bounds.bottom + ds.bounds.top) / 2
        lat = warp_transform(ds.crs, "EPSG:4326", [cx], [cy])[1][0]
        return math.sqrt(abs(ds.res[0]) * 111_320 * math.cos(math.radians(lat)) * abs(ds.res[1]) * 111_320)


def run_segmentation(
    path: str,
    out_dir: str | Path,
    model_id: str,
    weights_path: str,
    *,
    class_names: dict[int, str],
    aoi=None,
    rgb_bands: list[int] | None = None,
    min_area_m2: float = 8.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    import rasterio
    from pyproj import Geod, Transformer
    from rasterio.features import geometry_mask, shapes
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform

    from app.engine.raster_io import count_tiles, iter_tiles, prefetch
    from app.engine.raster_prep import plan_rgb
    from app.models.geodeep_onnx import load_session, segment_tile, tile_px_for_gsd

    t0 = time.time()
    sess, cfg = load_session(weights_path)
    gsd = ground_gsd_m(path)
    tile = tile_px_for_gsd(cfg, gsd)

    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise SegmentationError("This image has no CRS; footprints can't be placed or measured.")
        crs = ds.crs
        c0, r0, w, h = aoi.window if aoi else (0, 0, ds.width, ds.height)
        win_tf = window_transform(Window(c0, r0, w, h), ds.transform)
    if w * h > MAX_MASK_PIXELS:
        raise SegmentationError(
            f"This area is {w * h / 1e6:.0f} megapixels; draw an area of interest under {MAX_MASK_PIXELS / 1e6:.0f}.")

    mask = np.zeros((h, w), dtype=np.uint8)
    plan = plan_rgb(path, override=rgb_bands)
    total = count_tiles(w, h, tile, 0.1)
    done = 0

    def skip():
        nonlocal done
        done += 1
        if progress_cb:
            progress_cb(done, total)

    for t in prefetch(iter_tiles(path, tile_size=tile, overlap=0.1, rgb_plan=plan, on_skip=skip,
                                 window=aoi.window if aoi else None,
                                 tile_filter=aoi.tile_intersects if aoi else None)):
        m = segment_tile(t.data, sess, cfg, nominal_tile=tile)
        r, c = t.row_off - r0, t.col_off - c0
        region = mask[r:r + t.height, c:c + t.width]
        np.maximum(region, m[: region.shape[0], : region.shape[1]], out=region)
        done += 1
        if progress_cb:
            progress_cb(done, total)

    if aoi is not None:
        mask[geometry_mask([mapping(aoi.geom)], out_shape=mask.shape, transform=win_tf)] = 0

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")
    features, raw = [], 0
    for geom, val in shapes(mask, mask=mask > 0, transform=win_tf, connectivity=8):
        raw += 1
        poly = shape(geom).simplify(abs(win_tf.a) * 0.75, preserve_topology=True)
        poly_wgs = shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), poly)
        area = abs(geod.geometry_area_perimeter(poly_wgs)[0])
        if area < min_area_m2:
            continue
        cls = int(val)
        features.append({"type": "Feature", "geometry": mapping(poly_wgs), "properties": {
            "class": class_names.get(cls, class_names.get(0, f"class_{cls}")),
            "area_m2": round(area, 1)}})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gj = out_dir / f"{model_id}.geojson"
    from app.engine.measure import measure_features
    gj.write_text(json.dumps(measure_features({"type": "FeatureCollection", "features": features})))
    total_area = sum(f["properties"]["area_m2"] for f in features)
    summary = {"count": len(features), "total_area_ha": round(total_area / 10_000, 3),
               "image_gsd_m": round(gsd, 3) if gsd else None, "model_gsd_m": cfg["resolution"] / 100,
               "tile_px": tile}
    manifest = {"model_id": model_id, "count": len(features), "crs": "EPSG:4326", "source_crs": crs.to_string(),
                "weights": Path(weights_path).name, "rgb_bands": list(plan.bands), "min_area_m2": min_area_m2,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}
    mf = out_dir / f"{model_id}.manifest.json"
    mf.write_text(json.dumps(manifest, indent=2))
    return {"model_id": model_id, "device_used": "cpu", "device_reason": "onnxruntime",
            "tiles_processed": done, "raw_detection_count": raw, "final_detection_count": len(features),
            "duration_seconds": round(time.time() - t0, 3), "geojson_path": str(gj), "manifest_path": str(mf),
            "summary": summary}
