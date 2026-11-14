"""Accept / reject results, then turn them into training labels and hard negatives."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


@pytest.fixture()
def run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.core.jobs import get_store
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "harbour.tif"
    with rasterio.open(p, "w", driver="GTiff", width=1000, height=1000, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(55.0, 25.1, 1e-5, 1e-5)) as f:
        f.write(np.full((3, 1000, 1000), 90, np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("harbour.tif", fh, "image/tiff")}).json()

    def box(x, y, s=20):            # a ship of s px at pixel (x, y), as lon/lat
        lon, lat = 55.0 + x * 1e-5, 25.1 - y * 1e-5
        d = s * 1e-5
        return {"type": "Feature", "properties": {"class": "ship", "score": 0.8},
                "geometry": {"type": "Polygon", "coordinates": [[[lon, lat], [lon + d, lat], [lon + d, lat - d], [lon, lat - d], [lon, lat]]]}}

    job = get_store().create("predict", model_id="ship_detection", image_id=img["image_id"])
    job_id = job["job_id"] if isinstance(job, dict) else job
    out = settings.RESULTS_DIR / job_id
    out.mkdir(parents=True)
    (out / "ship_detection.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [box(100, 100), box(400, 400), box(700, 700)]}))
    return c, job_id, img


def test_review_is_saved_and_counted(run):
    c, job_id, _ = run
    url = f"/api/results/{job_id}/ship_detection/review"
    assert c.get(url).json()["open"] == 3
    r = c.put(url, json={"changes": {"0": "accept", "1": "reject"}}).json()
    assert (r["accepted"], r["rejected"], r["open"]) == (1, 1, 1)
    r = c.put(url, json={"changes": {"1": None}}).json()                  # undo
    assert (r["accepted"], r["rejected"], r["open"]) == (1, 0, 2)
    assert c.put(url, json={"changes": {"9": "accept"}}).status_code == 400
    assert c.put(url, json={"changes": {"0": "maybe"}}).status_code == 400
    assert c.get(url).json()["decisions"] == {"0": "accept"}              # survives a reload


def test_apply_makes_labels_and_hard_negatives_once(run):
    c, job_id, img = run
    url = f"/api/results/{job_id}/ship_detection/review"
    c.put(url, json={"changes": {"0": "accept", "1": "reject"}})
    r = c.post(url + "/apply").json()
    assert (r["accepted"], r["rejected"]) == (1, 1)
    anns = c.get(f"/api/images/{img['image_id']}/annotations").json()["annotations"]
    ships = [a for a in anns if a["class_name"] == "ship"]
    areas = [a for a in anns if a["class_name"] == "__train_area__"]
    assert len(ships) == 1 and len(areas) == 2
    xs = [p[0] for p in ships[0]["points_px"]]
    assert min(xs) == pytest.approx(100, abs=1) and max(xs) == pytest.approx(120, abs=1)
    a0 = [a for a in areas if a["source"]["index"] == 0][0]
    ax = [p[0] for p in a0["points_px"]]
    assert min(ax) == pytest.approx(92, abs=1) and max(ax) == pytest.approx(128, abs=1)   # 8 px pad (min)
    # apply again: same result, no duplicates; a changed decision replaces the old one
    c.put(url, json={"changes": {"1": "accept"}})
    r = c.post(url + "/apply").json()
    anns = c.get(f"/api/images/{img['image_id']}/annotations").json()["annotations"]
    assert r["accepted"] == 2 and len([a for a in anns if a["class_name"] == "ship"]) == 2
    assert len([a for a in anns if a["class_name"] == "__train_area__"]) == 2


def test_hand_drawn_labels_are_kept(run):
    c, job_id, img = run
    c.post(f"/api/images/{img['image_id']}/annotations",
           json={"class_name": "ship", "points_lonlat": [[55.001, 25.099], [55.002, 25.099], [55.002, 25.098]]})
    url = f"/api/results/{job_id}/ship_detection/review"
    c.put(url, json={"changes": {"2": "reject"}})
    c.post(url + "/apply")
    anns = c.get(f"/api/images/{img['image_id']}/annotations").json()["annotations"]
    assert any(a["class_name"] == "ship" and "source" not in a for a in anns)


def test_rejected_object_is_background_in_the_training_tile(run):
    """Masking: only the small area around the rejected ship is kept in its tile."""
    from app.api.training import area_filter

    c, job_id, img = run
    url = f"/api/results/{job_id}/ship_detection/review"
    c.put(url, json={"changes": {"1": "reject"}})
    c.post(url + "/apply")
    anns = c.get(f"/api/images/{img['image_id']}/annotations").json()["annotations"]
    areas = [a["points_px"] for a in anns if a["class_name"] == "__train_area__"]
    tile_filter, _ = area_filter(areas, [], 512, 1000, 1000)
    m = tile_filter(0, 0)
    assert m is not False and m[410, 410] and not m[100, 100]          # ship 1 kept, unreviewed ship 0 hidden
