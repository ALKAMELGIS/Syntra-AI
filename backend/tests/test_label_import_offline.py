"""Labels from a Shapefile / GeoJSON: the field standard, clipping, roles, duplicates."""
import io
import json
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import MultiPolygon, Point, box  # noqa: E402

X0, Y0, GSD = 300000, 2800000, 0.5          # image: 600 x 400 px, EPSG:32640


def px(c, r):
    return X0 + c * GSD, Y0 - r * GSD


def pbox(c0, r0, c1, r1):
    (x0, y0), (x1, y1) = px(c0, r1), px(c1, r0)
    return box(x0, y0, x1, y1)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    settings.ensure_dirs()
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "port.tif"
    with rasterio.open(p, "w", driver="GTiff", width=600, height=400, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(X0, Y0, GSD, GSD)) as f:
        f.write(np.full((3, 400, 600), 90, np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("port.tif", fh, "image/tiff")}).json()
    return c, img, tmp_path


def shp_zip(tmp_path, gdf) -> bytes:
    d = tmp_path / "shp"
    d.mkdir(exist_ok=True)
    gdf.to_file(d / "labels.shp")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in d.iterdir():
            z.write(f, f.name)
    return buf.getvalue()


def test_shapefile_with_type_field_imports_by_the_standard(env):
    c, img, tmp = env
    gdf = gpd.GeoDataFrame({
        "Type": ["Ship", " ship ", "Building", "Ship", "Ship", "Ship", ""],
        "role": ["", "object", "", "", "accuracy", "", ""],
        "geometry": [pbox(100, 100, 160, 140), pbox(200, 100, 240, 130), pbox(300, 200, 380, 260),
                     pbox(580, 380, 640, 420),                       # crosses the image edge: clipped
                     pbox(0, 0, 450, 250),                           # an accuracy area
                     pbox(2000, 2000, 2050, 2050),                   # outside the image
                     pbox(10, 10, 40, 40)],                          # no class
    }, crs="EPSG:32640")
    data = shp_zip(tmp, gdf)
    pre = c.post(f"/api/images/{img['image_id']}/annotations/import/preview",
                 files={"file": ("labels.zip", data, "application/zip")}).json()
    assert pre["class_field"] == "Type" and pre["role_field"] == "role"
    assert pre["polygons"] == 7 and pre["on_image"] == 6 and "32640" in pre["source_crs"]

    r = c.post(f"/api/images/{img['image_id']}/annotations/import",
               files={"file": ("labels.zip", data, "application/zip")},
               data={"class_field": "Type", "role_field": "role"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["by_class"] == {"ship": 3, "building": 1, "__check_area__": 1}
    assert rep["skipped"]["outside"] == 1 and rep["skipped"]["no_class"] == 1 and rep["skipped"]["clipped"] == 1
    ship = next(a for a in rep["annotations"] if a["class_name"] == "ship" and min(p[0] for p in a["points_px"]) > 90)
    xs = [p[0] for p in ship["points_px"]]
    assert min(xs) == pytest.approx(100, abs=1) and max(xs) == pytest.approx(160, abs=1)   # reprojected onto the pixels
    assert ship["tool"] == "import" and ship["source"] == "labels"

    again = c.post(f"/api/images/{img['image_id']}/annotations/import",
                   files={"file": ("labels.zip", data, "application/zip")},
                   data={"class_field": "Type", "role_field": "role"}).json()
    assert again["added"] == 0 and again["skipped"]["duplicates"] == 5


def test_one_class_for_the_file_and_a_class_map(env):
    c, img, tmp = env
    from app.annotations import store

    def ll(c_, r_):
        return list(store.pixel_to_lonlat(c_, r_, img["bounds_wgs84"], img["width"], img["height"]))

    ring = lambda a, b, cc, d: [ll(a, b), ll(cc, b), ll(cc, d), ll(a, d), ll(a, b)]  # noqa: E731
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"kind": "T1"}, "geometry": {"type": "Polygon", "coordinates": [ring(10, 10, 50, 50)]}},
        {"type": "Feature", "properties": {"kind": "T2"}, "geometry": {"type": "Polygon", "coordinates": [ring(60, 10, 90, 50)]}},
        {"type": "Feature", "properties": {"kind": "T1"}, "geometry": {"type": "Point", "coordinates": ll(5, 5)}},
        {"type": "Feature", "properties": {"kind": "T1"}, "geometry": {"type": "MultiPolygon", "coordinates": [
            [ring(100, 10, 120, 30)], [ring(130, 10, 150, 30)]]}},
    ]}
    body = json.dumps(fc).encode()
    one = c.post(f"/api/images/{img['image_id']}/annotations/import",
                 files={"file": ("t.geojson", body, "application/geo+json")}, data={"fixed_class": "Oil Tank"}).json()
    assert one["by_class"] == {"oil tank": 4} and one["skipped"]["not_polygon"] == 1   # a MultiPolygon's parts are 2 labels
    c.delete(f"/api/images/{img['image_id']}/annotations")
    mapped = c.post(f"/api/images/{img['image_id']}/annotations/import",
                    files={"file": ("t.geojson", body, "application/geo+json")},
                    data={"class_field": "kind", "class_map": json.dumps({"T1": "tank", "T2": None})}).json()
    assert mapped["by_class"] == {"tank": 3} and mapped["skipped"]["left_out"] == 1
    bad = c.post(f"/api/images/{img['image_id']}/annotations/import",
                 files={"file": ("t.geojson", body, "application/geo+json")})
    assert bad.status_code == 400 and "class" in bad.json()["detail"]
