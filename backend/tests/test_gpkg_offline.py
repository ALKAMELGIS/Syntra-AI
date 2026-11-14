"""GeoPackage and CSV exports: readable by GDAL as real layers with every attribute."""
import csv
import io

import pytest

pytest.importorskip("shapely")

SQ = [[55.0, 25.0], [55.001, 25.0], [55.001, 25.001], [55.0, 25.001], [55.0, 25.0]]
GJ = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [SQ]},
     "properties": {"class": "ship", "score": 0.91, "area_m2": 112.3, "orientation_deg": 30.0}},
    {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[x + 0.01, y] for x, y in SQ]]},
     "properties": {"class": "ship", "score": 0.55, "area_m2": 98.1}},
]}


def test_gpkg_opens_in_gdal_with_attributes_and_crs(tmp_path):
    from app.engine.gpkg_export import write_gpkg

    p = write_gpkg(tmp_path / "r.gpkg", GJ, "ship_detection")
    import sqlite3
    con = sqlite3.connect(p)
    assert con.execute("SELECT data_type, srs_id FROM gpkg_contents").fetchone() == ("features", 4326)
    rows = con.execute('SELECT class, score, area_m2, orientation_deg FROM "ship_detection" ORDER BY fid').fetchall()
    assert rows == [("ship", 0.91, 112.3, 30.0), ("ship", 0.55, 98.1, None)]
    blob = con.execute('SELECT geom FROM "ship_detection" LIMIT 1').fetchone()[0]
    assert blob[:2] == b"GP"
    con.close()


def test_gpkg_read_by_gdal(tmp_path):
    """GDAL itself (through pyogrio, when installed) reads the layer, its features and its CRS."""
    pyogrio = pytest.importorskip("pyogrio")
    from app.engine.gpkg_export import write_gpkg

    p = write_gpkg(tmp_path / "r.gpkg", GJ, "ship_detection")
    info = pyogrio.read_info(p)
    assert info["features"] == 2 and "EPSG:4326" in str(info["crs"])


def test_csv_has_every_column_and_wkt():
    from app.engine.gpkg_export import to_csv

    rows = list(csv.reader(io.StringIO(to_csv(GJ))))
    assert rows[0] == ["class", "score", "area_m2", "orientation_deg", "lon", "lat", "wkt"]
    assert rows[2][3] == "" and rows[1][-1].startswith("POLYGON")
