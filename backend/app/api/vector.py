"""
Read a vector file for the map's Layers panel (Add > From a file).

Shapefile (as a .zip with its .shp/.shx/.dbf/.prj), GeoPackage, KML,
GeoJSON, FlatGeobuf: read with geopandas, reprojected to WGS84 lon/lat, and
returned as GeoJSON for display only. Nothing is kept on the server: the
upload goes to a private temporary folder that is removed after the read.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import safe_filename

router = APIRouter(prefix="/api/vector", tags=["vector"])

MAX_MB = 100
MAX_FEATURES = 50_000
DIRECT = {".geojson", ".json", ".gpkg", ".kml", ".fgb"}
SHAPE_PARTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"}


def _extract(zpath: Path, dest: Path) -> Path:
    """Unpack a zip safely and return the one vector file inside it."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        total = 0
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())) or name.startswith("/") or ".." in Path(name).parts:
                raise HTTPException(status_code=400, detail="The zip has a path that points outside it.")
            total += info.file_size
            if total > MAX_MB * 4 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="The zip unpacks to more than 400 MB.")
        z.extractall(dest)
    found = sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix.lower() in (DIRECT | {".shp"}))
    shp = [p for p in found if p.suffix.lower() == ".shp"]
    pick = shp or found
    if not pick:
        raise HTTPException(status_code=400, detail="No .shp, .gpkg, .kml or .geojson inside the zip.")
    return pick[0]


def read_vector(path: Path) -> dict:
    """GeoJSON FeatureCollection in EPSG:4326, plus what was done to get there."""
    try:
        import geopandas as gpd
    except ImportError:
        raise HTTPException(status_code=503, detail="geopandas is not installed on this server.") from None
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - the reader's own message is the useful part
        raise HTTPException(status_code=400, detail=f"Could not read {path.name}: {exc}") from None
    if len(gdf) > MAX_FEATURES:
        raise HTTPException(status_code=413, detail=f"{len(gdf):,} features: the map shows up to {MAX_FEATURES:,}.")
    note = None
    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds if len(gdf) else (0, 0, 0, 0)
        if -180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90:
            gdf = gdf.set_crs(4326)
            note = "No coordinate system in the file; its numbers look like longitude and latitude, so they were read as WGS84."
        else:
            raise HTTPException(status_code=400, detail="The file has no coordinate system (.prj) and its coordinates are not longitude / latitude.")
    source_crs = gdf.crs.to_string()
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    for col in gdf.columns:
        if col != gdf.geometry.name and str(gdf[col].dtype).startswith(("datetime", "timedelta")):
            gdf[col] = gdf[col].astype(str)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    fc = json.loads(gdf.to_json(na="null", drop_id=True))
    return {"geojson": fc, "count": len(fc["features"]), "source_crs": source_crs, "note": note}


@router.post("/read")
async def read_upload(file: UploadFile = File(...)):
    name = safe_filename(file.filename, default="layer")
    ext = Path(name).suffix.lower()
    if ext not in DIRECT | {".zip"}:
        raise HTTPException(status_code=400, detail="Send a .zip of a Shapefile, or a .gpkg, .kml, .geojson or .fgb file.")
    settings.ensure_dirs()
    tmp_root = settings.DATA_DIR / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vector-", dir=tmp_root))
    try:
        dest = work / name
        size = 0
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"Files up to {MAX_MB} MB.")
                out.write(chunk)
        target = await run_in_threadpool(_extract, dest, work / "unzipped") if ext == ".zip" else dest
        result = await run_in_threadpool(read_vector, target)
        return {"name": Path(name).stem, **result}
    finally:
        shutil.rmtree(work, ignore_errors=True)
