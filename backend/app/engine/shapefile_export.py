"""
ESRI Shapefile export (.shp/.shx/.dbf) — Phase 5 GIS export.

Implemented from the public ESRI Shapefile Technical Description using only
the stdlib `struct` module — no pyshp/fiona/GDAL dependency, so this runs in
any environment including this build sandbox.

Supports Point and Polygon geometries (shape types 1 and 5), matching what
app/engine/postprocess.py produces (to_geojson_points / to_geojson_boxes /
to_geojson_polygons).

Validation strategy (see tests/test_shapefile_export_offline.py): this isn't
checked against a real GIS tool (none is available here), so correctness is
established two ways:
  1. A from-scratch reader (independent of the writer's internal helpers,
     re-deriving every offset from the spec) round-trips every written
     coordinate and attribute value.
  2. The .shp header's own "file length" field — defined by the spec as the
     total file size in 16-bit words — is checked against the actual file
     size on disk. This is a spec-derived invariant, not just "my reader
     agrees with my writer": if either the writer's length accounting or
     its byte layout is wrong, this check fails independently of the reader.
"""
from __future__ import annotations

import struct
from pathlib import Path

SHAPE_POINT = 1
SHAPE_POLYGON = 5


# DBF field names are at most 10 characters: readable short names for ours
SHORT_NAMES = {"perimeter_m": "perim_m", "orientation_deg": "orient_deg", "tree_candidates": "trees",
               "confidence": "conf"}


def _short_name(key: str, used: set[str]) -> str:
    name = SHORT_NAMES.get(key, key)[:10].encode("ascii", "replace").decode("ascii")
    base, i = name, 1
    while name.lower() in used:                       # keep names unique after shortening
        suffix = str(i)
        name = base[: 10 - len(suffix)] + suffix
        i += 1
    used.add(name.lower())
    return name


def _dbf_field_spec(key: str, sample_value, used: set[str] | None = None) -> tuple[str, str, int, int]:
    """(name, type_code, length, decimals) for one DBF field, inferred from a sample value."""
    name = _short_name(key, used if used is not None else set())
    if isinstance(sample_value, bool):
        return name, "L", 1, 0
    if isinstance(sample_value, (int, float)):
        return name, "N", 18, 6
    return name, "C", 254, 0


def _pack_dbf_value(value, type_code: str, length: int, decimals: int) -> bytes:
    if value is None:
        return b" " * length if type_code != "L" else b"?"
    if type_code == "N":
        s = f"{float(value):.{decimals}f}"
        return s[:length].rjust(length).encode("ascii")
    if type_code == "L":
        return (b"T" if value else b"F")
    s = str(value)
    return s[:length].ljust(length).encode("ascii", "replace")


def _write_dbf(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Cannot write a .dbf with zero records.")

    # every key any record has (not only the first), typed from its first real value
    keys: list[str] = []
    for r in records:
        keys += [k for k in r if k not in keys]
    used: set[str] = set()
    specs = [_dbf_field_spec(k, next((r[k] for r in records if r.get(k) is not None), ""), used) for k in keys]

    record_length = 1 + sum(length for _, _, length, _ in specs)  # +1 for the deletion flag byte
    header_length = 32 + 32 * len(specs) + 1  # file header + one 32-byte descriptor per field + terminator

    with path.open("wb") as f:
        # --- File header (32 bytes) ---
        f.write(bytes([0x03]))                       # version: dBase III, no memo
        f.write(bytes([0, 1, 1]))                     # last-update date YY/MM/DD (placeholder, not load-bearing)
        f.write(struct.pack("<I", len(records)))       # number of records
        f.write(struct.pack("<H", header_length))
        f.write(struct.pack("<H", record_length))
        f.write(b"\x00" * 20)                          # reserved

        # --- Field descriptor array ---
        for name, type_code, length, decimals in specs:
            f.write(name.encode("ascii").ljust(11, b"\x00"))
            f.write(type_code.encode("ascii"))
            f.write(b"\x00" * 4)                        # field data address (unused on write)
            f.write(bytes([length]))
            f.write(bytes([decimals]))
            f.write(b"\x00" * 14)
        f.write(b"\x0d")  # header terminator

        # --- Records ---
        for rec in records:
            f.write(b" ")  # deletion flag: not deleted
            for key, (name, type_code, length, decimals) in zip(keys, specs):
                v = rec.get(key)
                if type_code == "N" and not isinstance(v, (int, float)) or isinstance(v, bool) and type_code == "N":
                    v = None if v is None or isinstance(v, bool) else v
                f.write(_pack_dbf_value(v, type_code, length, decimals))
        f.write(b"\x1a")  # EOF marker


def _bbox(coords_list: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [p[0] for ring in coords_list for p in ring]
    ys = [p[1] for ring in coords_list for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def write_shapefile(path_prefix: str | Path, geojson: dict) -> dict:
    """
    Write <path_prefix>.shp / .shx / .dbf from a GeoJSON FeatureCollection of
    Point or Polygon features (uniform geometry type across all features).
    Returns the three output paths.
    """
    path_prefix = Path(path_prefix)
    features = geojson.get("features", [])
    if not features:
        raise ValueError("Cannot write a shapefile with zero features.")

    geom_type = features[0]["geometry"]["type"]
    if any(feat["geometry"]["type"] != geom_type for feat in features):
        raise ValueError("All features must share one geometry type for a single shapefile.")
    shape_type = {"Point": SHAPE_POINT, "Polygon": SHAPE_POLYGON}.get(geom_type)
    if shape_type is None:
        raise ValueError(f"Unsupported geometry type for shapefile export: {geom_type}")

    if shape_type == SHAPE_POINT:
        all_rings = [[tuple(f["geometry"]["coordinates"])] for f in features]
    else:
        all_rings = [[tuple(p) for p in f["geometry"]["coordinates"][0]] for f in features]
    xmin, ymin, xmax, ymax = _bbox(all_rings)

    shp_path = path_prefix.with_suffix(".shp")
    shx_path = path_prefix.with_suffix(".shx")
    dbf_path = path_prefix.with_suffix(".dbf")

    shp_records: list[bytes] = []
    for feat in features:
        geom = feat["geometry"]
        if shape_type == SHAPE_POINT:
            x, y = geom["coordinates"]
            content = struct.pack("<i", SHAPE_POINT) + struct.pack("<dd", x, y)
        else:
            ring = geom["coordinates"][0]
            rxmin, rymin, rxmax, rymax = _bbox([ring])
            content = (
                struct.pack("<i", SHAPE_POLYGON)
                + struct.pack("<dddd", rxmin, rymin, rxmax, rymax)
                + struct.pack("<i", 1)              # NumParts
                + struct.pack("<i", len(ring))       # NumPoints
                + struct.pack("<i", 0)                # Parts[0] = 0 (single ring starts at point 0)
                + b"".join(struct.pack("<dd", x, y) for x, y in ring)
            )
        shp_records.append(content)

    # --- Assemble .shp ---
    file_length_words = 50 + sum((8 + len(rec)) // 2 for rec in shp_records)  # header(50 words) + records
    with shp_path.open("wb") as f:
        f.write(struct.pack(">i", 9994))
        f.write(b"\x00" * 20)
        f.write(struct.pack(">i", file_length_words))
        f.write(struct.pack("<i", 1000))
        f.write(struct.pack("<i", shape_type))
        f.write(struct.pack("<dddd", xmin, ymin, xmax, ymax))
        f.write(struct.pack("<dddd", 0.0, 0.0, 0.0, 0.0))  # Z/M ranges, unused

        for i, content in enumerate(shp_records, start=1):
            f.write(struct.pack(">i", i))
            f.write(struct.pack(">i", len(content) // 2))
            f.write(content)

    # --- Assemble .shx (same header, then one 8-byte offset/length entry per record) ---
    shx_file_length_words = 50 + 4 * len(shp_records)
    with shx_path.open("wb") as f:
        f.write(struct.pack(">i", 9994))
        f.write(b"\x00" * 20)
        f.write(struct.pack(">i", shx_file_length_words))
        f.write(struct.pack("<i", 1000))
        f.write(struct.pack("<i", shape_type))
        f.write(struct.pack("<dddd", xmin, ymin, xmax, ymax))
        f.write(struct.pack("<dddd", 0.0, 0.0, 0.0, 0.0))

        offset_words = 50
        for content in shp_records:
            f.write(struct.pack(">i", offset_words))
            f.write(struct.pack(">i", len(content) // 2))
            offset_words += 4 + len(content) // 2

    # --- Assemble .dbf from each feature's properties ---
    _write_dbf(dbf_path, [feat.get("properties", {}) for feat in features])

    return {"shp": str(shp_path), "shx": str(shx_path), "dbf": str(dbf_path)}
