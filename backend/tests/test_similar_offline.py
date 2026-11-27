"""Similar objects: template matching finds the copies of a labelled object in the
view, skips what is already labelled, and outlines each from the example (no SAM)
or with the click engine."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("cv2")
pytest.importorskip("shapely")
from rasterio.transform import from_origin  # noqa: E402

# five copies of an L-shaped "building" (bright with a dark wing) on grey ground,
# one of them turned a quarter, and a plain bright square that is not one of them
L_SHAPE = np.zeros((40, 60), bool)
L_SHAPE[:, :25] = True
L_SHAPE[25:, :] = True
SPOTS = [(100, 100, 0), (100, 400, 0), (400, 100, 0), (700, 600, 0), (400, 500, 1)]


@pytest.fixture()
def app_img(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.engine import click_segment

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    click_segment.set_engine(None)
    monkeypatch.setattr(click_segment, "status", lambda: {"ready": False})
    from app.main import app

    rng = np.random.default_rng(3)
    img = (110 + rng.integers(0, 12, (3, 900, 1000))).astype(np.uint8)
    for y, x, turn in SPOTS:
        m = np.rot90(L_SHAPE, turn)
        img[:, y:y + m.shape[0], x:x + m.shape[1]][:, m] = np.array([235, 220, 200])[:, None]
    img[:, 700:750, 150:200] = 235                                  # a plain square: not an L
    p = tmp_path / "site.tif"
    with rasterio.open(p, "w", driver="GTiff", width=1000, height=900, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(300000, 2800000, 0.3, 0.3)) as f:
        f.write(img)
    c = TestClient(app)
    with open(p, "rb") as fh:
        info = c.post("/api/images", files={"file": ("site.tif", fh, "image/tiff")}).json()
    return c, info


def ll(img, x, y):
    from app.annotations import store
    return list(store.pixel_to_lonlat(x, y, img["bounds_wgs84"], img["width"], img["height"]))


def label(c, img, ring_px, cls="building"):
    pts = [ll(img, x, y) for x, y in ring_px]
    return c.post(f"/api/images/{img['image_id']}/annotations", json={"class_name": cls, "points_lonlat": pts}).json()


def L_ring(x, y):
    return [(x, y), (x + 25, y), (x + 25, y + 25), (x + 60, y + 25), (x + 60, y + 40), (x, y + 40)]


def view(img):
    b = img["bounds_wgs84"]
    return [b["west"], b["south"], b["east"], b["north"]] if isinstance(b, dict) else b


def centres(objs, img):
    from app.annotations import store
    out = []
    for o in objs:
        px = [store.lonlat_to_pixel(lon, lat, img["bounds_wgs84"], img["width"], img["height"]) for lon, lat in o["points_lonlat"]]
        out.append((np.mean([p[0] for p in px]), np.mean([p[1] for p in px])))
    return out


def test_finds_each_copy_once_including_a_turned_one(app_img):
    c, img = app_img
    ex = label(c, img, L_ring(100, 100))
    r = c.post(f"/api/images/{img['image_id']}/similar", json={"view": view(img), "example_ids": [ex["id"]], "shape": "polygon"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outlined_by"] == "example" and body["class_name"] == "building"
    got = centres(body["objects"], img)
    assert len(got) == 4, got
    for wx, wy in [(430, 120), (130, 420), (630, 720), (520, 430)]:
        assert any(abs(gx - wx) < 25 and abs(gy - wy) < 25 for gx, gy in got), (wx, wy, got)
    assert not any(abs(gx - 175) < 20 and abs(gy - 725) < 20 for gx, gy in got)   # the plain square


def test_skips_objects_that_are_already_labelled(app_img):
    c, img = app_img
    ex = label(c, img, L_ring(100, 100))
    label(c, img, L_ring(400, 100))
    r = c.post(f"/api/images/{img['image_id']}/similar", json={"view": view(img), "example_ids": [ex["id"]]}).json()
    assert len(r["objects"]) == 3


def test_needs_an_example(app_img):
    c, img = app_img
    r = c.post(f"/api/images/{img['image_id']}/similar", json={"view": view(img), "example_ids": ["nope"]})
    assert r.status_code == 400


def test_outlines_with_the_click_engine_when_it_is_there(app_img, monkeypatch):
    from app.engine import click_segment

    class BoxEngine:
        def encode(self, rgb):
            return {"size": rgb.shape[:2]}

        def decode(self, emb, points, labels):
            h, w = emb["size"]
            m = np.zeros((h, w), bool)
            (x0, y0), (x1, y1) = points[:2]
            m[int(y0):int(y1), int(x0):int(x1)] = True
            return m, 0.95

    click_segment.set_engine(BoxEngine())
    monkeypatch.setattr(click_segment, "status", lambda: {"ready": True})
    try:
        c, img = app_img
        ex = label(c, img, L_ring(100, 100))
        r = c.post(f"/api/images/{img['image_id']}/similar", json={"view": view(img), "example_ids": [ex["id"]], "shape": "box"}).json()
        assert r["outlined_by"] == "sam" and len(r["objects"]) == 4
    finally:
        click_segment.set_engine(None)
