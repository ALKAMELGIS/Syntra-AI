"""Measurements on result objects: true metres for area, sides and orientation."""
import math

import pytest

pytest.importorskip("pyproj")
pytest.importorskip("shapely")


def _rect_lonlat(lon0, lat0, length_m, width_m, bearing_deg):
    """A rectangle of known size and bearing, built on the ground and turned into lon/lat."""
    from pyproj import Transformer

    t = Transformer.from_crs(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", "EPSG:4326", always_xy=True)
    b = math.radians(bearing_deg)
    ux, uy = math.sin(b), math.cos(b)            # along the long side
    vx, vy = math.cos(b), -math.sin(b)            # across
    pts = []
    for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
        x = sl * length_m / 2 * ux + sw * width_m / 2 * vx
        y = sl * length_m / 2 * uy + sw * width_m / 2 * vy
        pts.append(list(t.transform(x, y)))
    return pts


@pytest.mark.parametrize("bearing", [0, 30, 95, 160])
def test_ship_sized_box_is_measured_in_true_metres(bearing):
    from app.engine.measure import measure_features

    fc = {"features": [{"geometry": {"type": "Polygon", "coordinates": [_rect_lonlat(55.27, 25.07, 40, 8, bearing)]},
                        "properties": {"class": "ship", "score": 0.9}}]}
    p = measure_features(fc)["features"][0]["properties"]
    assert p["class"] == "ship" and p["score"] == 0.9
    assert p["length_m"] == pytest.approx(40, abs=0.05) and p["width_m"] == pytest.approx(8, abs=0.05)
    assert p["area_m2"] == pytest.approx(320, rel=0.002)
    assert p["perimeter_m"] == pytest.approx(96, abs=0.1)
    assert min(abs(p["orientation_deg"] - bearing % 180), 180 - abs(p["orientation_deg"] - bearing % 180)) < 0.5
    assert p["lon"] == pytest.approx(55.27, abs=1e-6) and p["lat"] == pytest.approx(25.07, abs=1e-6)


def test_existing_values_are_kept_and_points_get_coordinates():
    from app.engine.measure import measure_features

    fc = {"features": [
        {"geometry": {"type": "Polygon", "coordinates": [_rect_lonlat(55.0, 25.0, 10, 10, 0)]}, "properties": {"area_m2": 99.0}},
        {"geometry": {"type": "Point", "coordinates": [55.1, 25.2]}, "properties": {}},
    ]}
    out = measure_features(fc)["features"]
    assert out[0]["properties"]["area_m2"] == 99.0
    assert out[1]["properties"] == {"lon": 55.1, "lat": 25.2}


def test_shapefile_takes_every_field_and_short_names(tmp_path):
    """Records with different keys and empty values export; long names become readable short ones."""
    from app.engine.shapefile_export import write_shapefile

    sq = [[55.0, 25.0], [55.001, 25.0], [55.001, 25.001], [55.0, 25.001], [55.0, 25.0]]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [sq]},
         "properties": {"class": "ship", "score": 0.9}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [sq]},
         "properties": {"class": "ship", "score": 0.8, "perimeter_m": 12.5, "orientation_deg": None}},
    ]}
    paths = write_shapefile(tmp_path / "r", gj)
    dbf = open(paths["dbf"], "rb").read()
    header = dbf[32:dbf.index(b"\x0d")]
    names = [header[i:i + 11].rstrip(b"\x00").decode() for i in range(0, len(header), 32)]
    assert names == ["class", "score", "perim_m", "orient_deg"]
