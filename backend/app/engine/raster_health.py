"""
Raster health check: what the analyst needs to know about a GeoTIFF before
trusting a detection run on it.

Every finding has a level (ok / info / warning / error), a short title and
a plain explanation, plus a fix where there is one. The checks come from
real failure modes seen on satellite products:

* No CRS: results can't be placed on Earth at all.
* Geographic CRS (degrees): fine for display, but pixel size is in degrees,
  so the ground resolution has to be converted before comparing it with a
  model's supported range.
* Ground resolution outside what the models were trained on.
* 16-bit data: must be stretched to 8-bit for the models (done
  automatically; reported so the analyst knows).
* Stripped layout and no overviews: every tile read pulls whole rows of a
  multi-GB file, and previews must scan the full image. A Cloud-Optimized
  GeoTIFF fixes both.
* RPC metadata present: the product is likely a system-corrected level
  (e.g. L2A), not terrain-orthorectified, so absolute position can be off
  by several metres in relief.
* Mostly empty footprint.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Bump when the findings change, so saved checks in the image library are redone.
HEALTH_VERSION = 3
MODEL_GSD_RANGE_M = (0.01, 2.00)   # widest range any catalog detector supports (drone 1-15 cm, satellite up to 2 m)


@dataclass
class Finding:
    level: str          # ok | info | warning | error
    code: str
    title: str
    detail: str
    fix: str | None = None


@dataclass
class RasterHealth:
    path: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None
    crs_name: str | None
    epsg: int | None
    projected: bool | None
    units: str | None
    gsd_m: float | None
    nodata: float | None
    valid_fraction: float | None
    tiled: bool
    block_shape: list[int]
    overviews: list[int]
    compression: str | None
    has_rpc: bool
    file_size_bytes: int
    has_calibration: bool = False
    rgb_bands: list[int] | None = None
    rgb_source: str | None = None
    sensor: str | None = None
    distinct_values: int | None = None     # single-band files: how many values the sample holds
    findings: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        levels = {f.level for f in self.findings}
        return "error" if "error" in levels else "warning" if "warning" in levels else "ok"

    def to_dict(self) -> dict:
        from app.engine.raster_prep import json_safe

        d = asdict(self)
        d["nodata"] = json_safe(d["nodata"])
        d["status"] = self.status
        d["version"] = HEALTH_VERSION
        return d


def _ground_gsd_m(res_x: float, res_y: float, projected: bool | None, lat_deg: float | None,
                  units_to_m: float = 1.0) -> float | None:
    if projected:
        return round(abs(res_x) * units_to_m, 4)
    if projected is False and lat_deg is not None:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
        return round(math.sqrt(abs(res_x) * m_per_deg_lon * abs(res_y) * m_per_deg_lat), 4)
    return None


def check_raster(path: str | Path, sample_dim: int = 1024) -> RasterHealth:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    from app.engine.raster_prep import choose_rgb_bands, detect_sensor

    path = Path(path)
    with rasterio.open(path) as ds:
        crs = ds.crs
        epsg = crs.to_epsg() if crs else None
        projected = bool(crs.is_projected) if crs else None
        units = None
        units_to_m = 1.0
        if crs is not None:
            if projected:
                try:
                    units, factor = crs.linear_units_factor
                    units_to_m = float(factor)
                except Exception:
                    units = crs.linear_units
            else:
                units = "degree"
        lat = None
        if crs is not None:
            cx = (ds.bounds.left + ds.bounds.right) / 2
            cy = (ds.bounds.bottom + ds.bounds.top) / 2
            try:
                _, ys = warp_transform(crs, "EPSG:4326", [cx], [cy])
                lat = ys[0]
            except Exception:
                lat = None
        gsd = _ground_gsd_m(ds.res[0], ds.res[1], projected, lat, units_to_m)

        scale = min(1.0, sample_dim / max(ds.width, ds.height))
        sample = ds.read(out_shape=(ds.count, max(1, round(ds.height * scale)), max(1, round(ds.width * scale))),
                         resampling=Resampling.nearest, masked=True)
        from app.engine.raster_prep import valid_pixels
        valid_fraction = round(float(np.mean(valid_pixels(sample, ds.nodata))), 4)
        distinct = int(len(np.unique(np.ma.getdata(sample)[0]))) if ds.count == 1 else None

        rgb, rgb_source = choose_rgb_bands(ds.count, [c.name for c in ds.colorinterp], list(ds.descriptions), detect_sensor(path))
        h = RasterHealth(
            path=str(path), width=ds.width, height=ds.height, bands=ds.count, dtype=str(ds.dtypes[0]),
            crs=crs.to_string() if crs else None,
            crs_name=(crs.to_wkt().split('"')[1] if crs else None),
            epsg=epsg, projected=projected, units=units, gsd_m=gsd, nodata=ds.nodata,
            valid_fraction=valid_fraction,
            tiled=bool(ds.profile.get("tiled", False)),
            block_shape=list(ds.block_shapes[0]),
            overviews=ds.overviews(1),
            compression=getattr(ds.compression, "value", ds.compression) if ds.compression else None,  # a str for YCbCr JPEG on some GDAL builds
            has_rpc=ds.rpcs is not None,
            file_size_bytes=path.stat().st_size,
            rgb_bands=list(rgb), rgb_source=rgb_source, sensor=detect_sensor(path),
            distinct_values=distinct,
        )
        from app.engine.vegetation import read_calibration_gains
        h.has_calibration = read_calibration_gains(path) is not None

    h.findings = _findings(h)
    return h


_MASK_NAME = re.compile(r"(cloud|cld|mask|udm|_qa|qa_|snow|shadow)", re.I)


def looks_like_mask(h: RasterHealth) -> bool:
    """A cloud / quality mask: named like one, or one band with at most 3 values."""
    if h.bands != 1:
        return False
    named = bool(_MASK_NAME.search(Path(h.path).stem))
    few = h.distinct_values is not None and h.distinct_values <= 3
    return named and (few or h.valid_fraction == 0) or (few and h.dtype == "uint8" and h.width * h.height > 1_000_000)


def _findings(h: RasterHealth) -> list[Finding]:
    f: list[Finding] = []

    if h.crs is None:
        f.append(Finding("error", "no_crs", "No coordinate reference system",
                         "The file has no CRS, so detections can't be placed on the map or exported to GIS.",
                         "Assign the correct CRS in QGIS (Raster > Projections > Assign Projection) or with "
                         "gdal_edit.py -a_srs EPSG:xxxx, then upload again."))
    elif h.projected:
        f.append(Finding("ok", "crs", f"CRS {h.crs}", f"{h.crs_name}, units: {h.units}. Results are exported in WGS84 (EPSG:4326)."))
    else:
        f.append(Finding("info", "geographic_crs", f"Geographic CRS {h.crs}",
                         "Pixel size is in degrees. Ground resolution below is converted to metres at the scene's latitude. "
                         "Areas and distances are computed after reprojection, so results are still correct.",
                         "For measurement-heavy work, reproject to the local UTM zone first."))

    if h.gsd_m is not None:
        lo, hi = MODEL_GSD_RANGE_M
        if h.gsd_m < lo or h.gsd_m > hi:
            f.append(Finding("warning", "gsd_range", f"Ground resolution {h.gsd_m:g} m",
                             f"The detection models work best between {lo:g} and {hi:g} m per pixel. "
                             "Objects may be too small or too large for reliable detection at this resolution.",
                             "Drone models need 1-15 cm, satellite models 0.3-0.5 m: resample, or pansharpen a coarser image."))
        else:
            f.append(Finding("ok", "gsd", f"Ground resolution {h.gsd_m:g} m", "Within the range the detection models support."))

    if h.dtype != "uint8":
        f.append(Finding("info", "bit_depth", f"{h.dtype} data",
                         "Converted to 8-bit for the models with one scene-wide 2–98% stretch, so all tiles are treated the same."))

    if h.bands == 1 and not looks_like_mask(h):
        f.append(Finding("info", "single_band", "Single band (panchromatic)",
                         "Detection runs on the grey image. Colour helps some classes; pansharpening with the "
                         "multispectral image usually improves results."))
    elif h.bands > 3:
        f.append(Finding("info", "band_choice", f"{h.bands} bands; using {h.rgb_bands} as R, G, B",
                         f"Chosen from {h.rgb_source}. If colours look wrong in the preview, override the bands for the run."))

    big = h.file_size_bytes > 200 * 1024 * 1024
    if not h.tiled and big:
        f.append(Finding("warning", "striped", "Stripped layout",
                         f"The file is stored in rows ({h.block_shape[0]}×{h.block_shape[1]}), so every tile read loads whole "
                         "image rows. Large scenes will process slowly.",
                         "gdal_translate -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER input.tif output_cog.tif"))
    if not h.overviews and big:
        f.append(Finding("warning", "no_overviews", "No overviews",
                         "Previews and zoomed-out views have to read the full-resolution image.",
                         "gdaladdo -r average input.tif 2 4 8 16 32, or convert to COG (which builds them)."))

    if h.has_rpc:
        f.append(Finding("info", "rpc", "RPC sensor model present",
                         "This is typically a system-corrected product (for example L2A), not orthorectified with a terrain "
                         "model. Positions can be off by a few metres where there is relief; on flat coastal terrain the effect is small.",
                         "For survey-grade positions, orthorectify with a DEM (gdalwarp -rpc -to RPC_DEM=dem.tif)."))

    if h.bands >= 4 and h.dtype != "uint8" and not h.has_calibration:
        f.append(Finding("info", "no_calibration", "No calibration metadata",
                         "Vegetation (NDVI) will be computed from raw values, which reads low on most sensors.",
                         "Upload the product's .xml together with the image."))
    elif h.has_calibration:
        f.append(Finding("ok", "calibration", "Calibration metadata found",
                         "NDVI uses calibrated reflectance."))

    mask = looks_like_mask(h)
    if h.valid_fraction == 0:
        if mask:
            f.append(Finding("error", "no_data", "No image data: this is a mask",
                             f"{Path(h.path).name} is a mask that ships with the scene (cloud or quality), not the image. "
                             "Every pixel is 0, so there is nothing to show on the map or to detect.",
                             "Upload the image file from the same folder, the one without Cloud or Mask in its name "
                             "(for SuperView Neo: the PAN, MUX or PMS .tiff)."))
        else:
            f.append(Finding("error", "no_data", "No image data",
                             "Every pixel is 0 or no-data, so there is nothing to show on the map or to detect.",
                             "Check the export: open the file in QGIS and confirm it shows the scene."))
        return f
    if mask:
        f.append(Finding("warning", "mask", "Looks like a mask, not imagery",
                         f"One band with {h.distinct_values} value{'s' if h.distinct_values != 1 else ''}"
                         f"{' and a mask-like name' if _MASK_NAME.search(Path(h.path).stem) else ''}. "
                         "Detection models need the image itself.",
                         "Upload the scene's image file (for SuperView Neo: PAN, MUX or PMS)."))
    if h.valid_fraction is not None and h.valid_fraction < 0.5:
        f.append(Finding("info", "sparse", f"{h.valid_fraction:.0%} of the frame has data",
                         "Empty tiles are skipped automatically, so this costs no processing time."))
    return f
