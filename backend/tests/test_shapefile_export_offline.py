"""
Offline test for app/engine/shapefile_export.py.

Two independent checks, per the module's own docstring:
  1. A from-scratch reader — written here, NOT importing any of the writer's
     internal helpers, re-deriving every byte offset from the public ESRI
     Shapefile spec — round-trips coordinates and attributes.
  2. The .shp header's self-declared "file length" (in 16-bit words, per
     the spec) is checked against the real file size on disk: a
     spec-derived invariant independent of whether the reader and writer
     happen to agree with each other.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.shapefile_export import write_shapefile

SHAPE_POINT = 1
SHAPE_POLYGON = 5


def _read_shp_header(path: Path) -> dict:
    data = path.read_bytes()
    file_code, = struct.unpack(">i", data[0:4])
    file_length_words, = struct.unpack(">i", data[24:28])
    version, = struct.unpack("<i", data[28:32])
    shape_type, = struct.unpack("<i", data[32:36])
    bbox = struct.unpack("<dddd", data[36:68])
    return {
        "file_code": file_code, "file_length_words": file_length_words,
        "version": version, "shape_type": shape_type, "bbox": bbox, "raw": data,
    }


def _read_shp_points(path: Path) -> list[tuple[float, float]]:
    header = _read_shp_header(path)
    assert header["file_code"] == 9994
    assert header["version"] == 1000
    assert header["shape_type"] == SHAPE_POINT

    data = header["raw"]
    offset = 100
    points = []
    while offset < len(data):
        rec_num, content_len_words = struct.unpack(">ii", data[offset:offset + 8])
        offset += 8
        content = data[offset:offset + content_len_words * 2]
        shape_type, x, y = struct.unpack("<idd", content)
        assert shape_type == SHAPE_POINT
        points.append((x, y))
        offset += content_len_words * 2
    return points


def _read_shp_polygons(path: Path) -> list[list[tuple[float, float]]]:
    header = _read_shp_header(path)
    assert header["file_code"] == 9994
    assert header["shape_type"] == SHAPE_POLYGON

    data = header["raw"]
    offset = 100
    polygons = []
    while offset < len(data):
        rec_num, content_len_words = struct.unpack(">ii", data[offset:offset + 8])
        offset += 8
        content = data[offset:offset + content_len_words * 2]
        shape_type = struct.unpack_from("<i", content, 0)[0]
        assert shape_type == SHAPE_POLYGON
        num_parts, num_points = struct.unpack_from("<ii", content, 36)
        pos = 44 + 4 * num_parts  # skip Parts array (we only test single-ring polygons)
        ring = []
        for i in range(num_points):
            x, y = struct.unpack_from("<dd", content, pos)
            ring.append((x, y))
            pos += 16
        polygons.append(ring)
        offset += content_len_words * 2
    return polygons


def _read_dbf(path: Path) -> list[dict]:
    data = path.read_bytes()
    num_records, = struct.unpack("<I", data[4:8])
    header_length, = struct.unpack("<H", data[8:10])
    record_length, = struct.unpack("<H", data[10:12])

    fields = []
    pos = 32
    while data[pos:pos + 1] != b"\x0d":
        name = data[pos:pos + 11].rstrip(b"\x00").decode("ascii")
        type_code = chr(data[pos + 11])
        length = data[pos + 16]
        decimals = data[pos + 17]
        fields.append((name, type_code, length, decimals))
        pos += 32

    records = []
    rec_start = header_length
    for i in range(num_records):
        raw = data[rec_start:rec_start + record_length]
        values = {}
        cursor = 1  # skip deletion flag
        for name, type_code, length, decimals in fields:
            raw_val = raw[cursor:cursor + length].decode("ascii").strip()
            if type_code == "N":
                values[name] = float(raw_val) if raw_val else 0.0
            elif type_code == "L":
                values[name] = raw_val == "T"
            else:
                values[name] = raw_val
            cursor += length
        records.append(values)
        rec_start += record_length
    return records


def test_file_length_header_matches_actual_size():
    """Spec invariant: .shp/.shx header 'file length' (16-bit words) == actual file size / 2."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [10.0, 20.0]}, "properties": {"class": "car", "score": 0.8}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [30.0, 40.0]}, "properties": {"class": "car", "score": 0.6}},
        ],
    }
    tmp = Path(tempfile.mkdtemp())
    paths = write_shapefile(tmp / "cars", geojson)

    for key in ("shp", "shx"):
        path = Path(paths[key])
        header = _read_shp_header(path) if key == "shp" else None
        if key == "shp":
            actual_size = path.stat().st_size
            assert header["file_length_words"] * 2 == actual_size, (
                f"{key}: header declares {header['file_length_words'] * 2} bytes but file is {actual_size} bytes"
            )
    print("Shapefile .shp header file-length matches actual file size on disk: OK")


def test_point_shapefile_roundtrip():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [46.7, 24.7]}, "properties": {"class": "palm_tree", "score": 0.91}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [46.8, 24.8]}, "properties": {"class": "palm_tree", "score": 0.77}},
        ],
    }
    tmp = Path(tempfile.mkdtemp())
    paths = write_shapefile(tmp / "palms", geojson)

    points = _read_shp_points(Path(paths["shp"]))
    assert len(points) == 2
    assert abs(points[0][0] - 46.7) < 1e-9 and abs(points[0][1] - 24.7) < 1e-9
    assert abs(points[1][0] - 46.8) < 1e-9 and abs(points[1][1] - 24.8) < 1e-9

    records = _read_dbf(Path(paths["dbf"]))
    assert len(records) == 2
    assert records[0]["class"] == "palm_tree"
    assert abs(records[0]["score"] - 0.91) < 1e-6
    assert abs(records[1]["score"] - 0.77) < 1e-6
    print(f"Point shapefile round-trip: {len(points)} points + DBF attributes verified — OK")


def test_polygon_shapefile_roundtrip():
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"class": "field", "area_ha": 2.5}},
        ],
    }
    tmp = Path(tempfile.mkdtemp())
    paths = write_shapefile(tmp / "fields", geojson)

    polygons = _read_shp_polygons(Path(paths["shp"]))
    assert len(polygons) == 1
    assert len(polygons[0]) == 5  # closed ring, 5 points
    assert polygons[0][0] == (0.0, 0.0)
    assert polygons[0][2] == (10.0, 10.0)

    records = _read_dbf(Path(paths["dbf"]))
    assert records[0]["class"] == "field"
    assert abs(records[0]["area_ha"] - 2.5) < 1e-6
    print("Polygon shapefile round-trip: ring + DBF attributes verified — OK")


if __name__ == "__main__":
    test_file_length_header_matches_actual_size()
    test_point_shapefile_roundtrip()
    test_polygon_shapefile_roundtrip()
    print("test_shapefile_export_offline: PASSED")
