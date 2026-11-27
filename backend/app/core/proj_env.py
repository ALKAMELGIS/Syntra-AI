"""
Make rasterio / pyproj use the PROJ and GDAL data that belong to *this*
Python environment.

On Windows, installers for PostgreSQL/PostGIS, QGIS or ArcGIS often set
PROJ_LIB / PROJ_DATA / GDAL_DATA system-wide, pointing at their own copy of
proj.db. A different PROJ version then refuses it ("DATABASE.LAYOUT.VERSION.MINOR
= 2 whereas a number >= 6 is expected") and every CRS lookup fails, so no
GeoTIFF can be read.

Rule: a value that points inside this environment is kept. A foreign one is
replaced by this environment's data folder when there is one (conda), or
removed so the wheels fall back to the data they bundle (pip).
Runs before rasterio or pyproj are imported (app/__init__.py).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _inside(path: str, roots: list[Path]) -> bool:
    try:
        p = Path(path).resolve()
    except OSError:
        return False
    return any(p == r or r in p.parents for r in roots)


def _package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return None


def ensure_own_geodata() -> dict[str, str | None]:
    prefix = Path(sys.prefix).resolve()
    roots = [prefix, Path(sys.base_prefix).resolve()]
    proj_own, gdal_own = [], []
    for base in (prefix / "Library" / "share", prefix / "share"):          # conda (Windows, then Unix)
        proj_own.append(base / "proj")
        gdal_own.append(base / "gdal")
    for pkg, proj_sub, gdal_sub in (("rasterio", "proj_data", "gdal_data"), ("pyproj", "proj_dir/share/proj", None)):
        d = _package_dir(pkg)
        if d:
            roots.append(d.resolve())
            proj_own.append(d / proj_sub)
            if gdal_sub:
                gdal_own.append(d / gdal_sub)

    def pick(cands: list[Path], marker: str) -> str | None:
        for c in cands:
            if (c / marker).exists():
                return str(c)
        return None

    changed: dict[str, str | None] = {}
    proj_dir = pick(proj_own[:2], "proj.db")           # conda: set explicitly
    gdal_dir = pick(gdal_own[:2], "gdalvrt.xsd") or pick(gdal_own[:2], "header.dxf")
    for keys, own in ((("PROJ_DATA", "PROJ_LIB"), proj_dir), (("GDAL_DATA",), gdal_dir)):
        for k in keys:
            cur = os.environ.get(k)
            if cur and _inside(cur, roots):
                continue
            if own:
                if cur != own:
                    os.environ[k] = own
                    changed[k] = own
            elif cur:
                del os.environ[k]                          # pip wheels: use their bundled data
                changed[k] = None
    return changed
