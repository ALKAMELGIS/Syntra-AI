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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODEL_GSD_RANGE_M = (0.10, 2.00)   # widest range any catalog detector supports


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
    rgb_bands: list[int] | None = None
    rgb_source: str | None = None
    sensor: str | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        levels = {f.level for f in self.findings}
        return "error" if "error" in levels else "warning" if "warning" in levels else "ok"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
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
                         resampling=Resampling.nearest)
        nd = 0 if ds.nodata is None else ds.nodata
        valid_fraction = round(float(np.mean(~np.all(sample == nd, axis=0))), 4)

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
            compression=ds.compression.value if ds.compression else None,
            has_rpc=ds.rpcs is not None,
            file_size_bytes=path.stat().st_size,
            rgb_bands=list(rgb), rgb_source=rgb_source, sensor=detect_sensor(path),
        )

    h.findings = _findings(h)
    return h


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
                             "Resample to about 0.3–0.5 m, or pansharpen a coarser multispectral image."))
        else:
            f.append(Finding("ok", "gsd", f"Ground resolution {h.gsd_m:g} m", "Within the range the detection models support."))

    if h.dtype != "uint8":
        f.append(Finding("info", "bit_depth", f"{h.dtype} data",
                         "Converted to 8-bit for the models with one scene-wide 2–98% stretch, so all tiles are treated the same."))

    if h.bands == 1:
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

    if h.valid_fraction is not None and h.valid_fraction < 0.5:
        f.append(Finding("info", "sparse", f"{h.valid_fraction:.0%} of the frame has data",
                         "Empty tiles are skipped automatically, so this costs no processing time."))
    return f
