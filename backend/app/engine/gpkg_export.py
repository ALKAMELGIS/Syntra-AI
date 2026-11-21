"""
GeoPackage and CSV export, with no GDAL/geopandas needed.

A GeoPackage is a SQLite file with a few required tables (OGC 12-128r18):
gpkg_spatial_ref_sys, gpkg_contents, gpkg_geometry_columns, and the feature
table, whose geometry is "GeoPackage binary": a small header (magic GP,
version, flags, srs id, envelope) followed by standard WKB. QGIS and ArcGIS
open it directly. Results are WGS84 (EPSG:4326).
"""
from __future__ import annotations

import csv
import io
import re
import sqlite3
import struct
from pathlib import Path

WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
             'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')


def _gpkg_geom(geom) -> bytes:
    minx, miny, maxx, maxy = geom.bounds
    flags = 0b00000011            # little endian, envelope [minx, maxx, miny, maxy]
    header = b"GP" + bytes([0, flags]) + struct.pack("<i", 4326) + struct.pack("<4d", minx, maxx, miny, maxy)
    return header + geom.wkb


def _column(name: str, used: set[str]) -> str:
    c = re.sub(r"[^A-Za-z0-9_]", "_", name) or "field"
    base, i = c, 1
    while c.lower() in used or c.lower() in ("fid", "geom"):
        c = f"{base}_{i}"; i += 1
    used.add(c.lower())
    return c


def _fields(features: list[dict]) -> list[tuple[str, str, str]]:
    """(property key, column name, sqlite type) for every key any feature has."""
    keys: list[str] = []
    for f in features:
        keys += [k for k in (f.get("properties") or {}) if k not in keys]
    used: set[str] = set()
    out = []
    for k in keys:
        vals = [(f.get("properties") or {}).get(k) for f in features]
        real = [v for v in vals if v is not None]
        if real and all(isinstance(v, bool) for v in real):
            t = "BOOLEAN"
        elif real and all(isinstance(v, int) and not isinstance(v, bool) for v in real):
            t = "INTEGER"
        elif real and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in real):
            t = "REAL"
        else:
            t = "TEXT"
        out.append((k, _column(k, used), t))
    return out


def write_gpkg(path: str | Path, geojson: dict, table: str) -> Path:
    from shapely.geometry import shape

    path = Path(path)
    path.unlink(missing_ok=True)
    table = re.sub(r"[^A-Za-z0-9_]", "_", table) or "results"
    feats = [f for f in geojson.get("features", []) if f.get("geometry")]
    geoms = [shape(f["geometry"]) for f in feats]
    gtype = "GEOMETRY"
    kinds = {g.geom_type for g in geoms}
    if len(kinds) == 1:
        gtype = kinds.pop().upper()
    fields = _fields(feats)

    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA application_id = 1196444487")      # 'GPKG'
        cur.execute("PRAGMA user_version = 10400")
        cur.execute("""CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
            organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)""")
        cur.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, WGS84_WKT, "longitude/latitude WGS84"),
        ])
        cur.execute("""CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
            identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER)""")
        cur.execute("""CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name))""")
        cols = ", ".join(f'"{c}" {t}' for _, c, t in fields)
        cur.execute(f'CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom {gtype}{", " + cols if cols else ""})')
        if geoms:
            from shapely.ops import unary_union
            minx, miny, maxx, maxy = unary_union(geoms).bounds
        else:
            minx = miny = maxx = maxy = None
        cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) VALUES (?,?,?,?,?,?,?,?)",
                    (table, "features", table, minx, miny, maxx, maxy, 4326))
        cur.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)", (table, "geom", gtype, 4326, 0, 0))
        names = ["geom"] + [f'"{c}"' for _, c, _ in fields]
        sql = f'INSERT INTO "{table}" ({", ".join(names)}) VALUES ({", ".join("?" * len(names))})'
        for f, g in zip(feats, geoms):
            props = f.get("properties") or {}
            row = [_gpkg_geom(g)]
            for k, _, t in fields:
                v = props.get(k)
                if isinstance(v, (dict, list)):
                    v = str(v)
                row.append(v)
            cur.execute(sql, row)
        con.commit()
    finally:
        con.close()
    return path


def to_csv(geojson: dict) -> str:
    """One row per object: its properties, then centroid lon/lat and the shape as WKT."""
    from shapely.geometry import shape

    feats = [f for f in geojson.get("features", []) if f.get("geometry")]
    keys: list[str] = []
    for f in feats:
        keys += [k for k in (f.get("properties") or {}) if k not in keys and k not in ("lon", "lat")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(keys + ["lon", "lat", "wkt"])
    for f in feats:
        g = shape(f["geometry"])
        p = f.get("properties") or {}
        c = g.centroid
        w.writerow([p.get(k, "") for k in keys] + [p.get("lon", round(c.x, 7)), p.get("lat", round(c.y, 7)), g.wkt])
    return buf.getvalue()
