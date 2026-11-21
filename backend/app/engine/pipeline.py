"""
DetectionEngine — the orchestrator that ties together:

    raster_io (windowed tiling) -> model.predict (per tile)
      -> model.georeference (tile-local pixel -> mosaic geo coords)
      -> postprocess.merge_and_dedupe (cross-tile NMS)
      -> model.export_results (GeoJSON + manifest)

Works against anything implementing DetectionModel, so it is identical for
a real ultralytics-backed detector, a segmentation model, or (in tests) the
offline stub.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from app.core.compute import DeviceOption, resolve_device
from app.engine.geo import GeoTransform, reproject_detections_to_wgs84
from app.engine.postprocess import merge_and_dedupe
from app.models.base import DetectionModel


@dataclass
class RunResult:
    model_id: str
    device_used: str
    device_reason: str
    tiles_processed: int
    raw_detection_count: int
    final_detection_count: int
    duration_seconds: float
    geojson_path: str
    manifest_path: str


class DetectionEngine:
    def __init__(self, model: DetectionModel):
        self.model = model

    def run(
        self,
        tiles: Iterable,                           # yields engine.raster_io.Tile (or any object with .data/.transform)
        out_dir: str | Path,
        device_option: DeviceOption = DeviceOption.AUTO,
        confidence_threshold: float | None = None,
        nms_iou: float = 0.45,
        progress_cb: Callable[[int, int | None], None] | None = None,
        allowed_gpu_indices: list[int] | None = None,
        source_crs: str | None = None,             # raster CRS; output is reprojected to WGS84
        tiles_total: int | None = None,            # for progress reporting only
        keep_detection: Callable[[dict], bool] | None = None,   # e.g. centre inside an AOI (raster CRS)
    ) -> RunResult:
        t0 = time.time()

        profile = resolve_device(device_option, allowed_gpu_indices=allowed_gpu_indices)
        self.model.load_model(profile.resolved)
        ok, msg = self.model.validate_model()
        if not ok:
            raise RuntimeError(f"Model failed validation before run: {msg}")

        class_names = self.model.metadata.class_names
        all_global_detections: list[dict] = []
        tiles_processed = 0

        for tile in tiles:
            raw = self.model.predict(tile.data, confidence_threshold=confidence_threshold)
            raw = self.model.postprocess(raw)
            geo_dets = self.model.georeference(raw, tile.transform, class_names)
            all_global_detections.extend(geo_dets)
            tiles_processed += 1
            if progress_cb:
                progress_cb(tiles_processed, tiles_total)

        raw_count = len(all_global_detections)
        merged = merge_and_dedupe(all_global_detections, iou_threshold=nms_iou)
        if keep_detection is not None:
            merged = [d for d in merged if keep_detection(d)]
        merged, output_crs = reproject_detections_to_wgs84(merged, source_crs)

        export = self.model.export_results(
            str(out_dir),
            merged,
            extra_meta={
                "device": profile.resolved,
                "device_reason": profile.reason,
                "tiles_processed": tiles_processed,
                "raw_detection_count": raw_count,
                "source_crs": source_crs,
                "crs": output_crs,
            },
        )

        return RunResult(
            model_id=self.model.metadata.model_id,
            device_used=profile.resolved,
            device_reason=profile.reason,
            tiles_processed=tiles_processed,
            raw_detection_count=raw_count,
            final_detection_count=len(merged),
            duration_seconds=round(time.time() - t0, 3),
            geojson_path=export["geojson_path"],
            manifest_path=export["manifest_path"],
        )
