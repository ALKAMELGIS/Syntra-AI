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


def test_review_reaches_the_dataset(run):
    """End to end: accepted ships become labelled tiles, the rejected one a background tile.

    Before the fix every tile was dropped: review areas cover under 1% of a
    640 px tile, under the 35% area rule and over the 80% black rule."""
    from pathlib import Path

    c, job_id, img = run
    url = f"/api/results/{job_id}/ship_detection/review"
    c.put(url, json={"changes": {"0": "accept", "2": "accept", "1": "reject"}})
    c.post(url + "/apply")
    r = c.post("/api/training/export-dataset", json={"image_id": img["image_id"], "task": "obb", "tile": 640,
                                                     "overlap": 64, "augment": False})
    assert r.status_code == 200, r.text
    st = c.get(f"/api/training/export-dataset/{r.json()['job_id']}").json()
    assert st["status"] == "completed", st
    stats = st["info"]["stats"]
    assert stats["class_counts"]["ship"] == 2
    assert stats.get("reviewed_negatives", 0) >= 1
    labels = list((Path(st["data_yaml"]).parent / "labels").rglob("*.txt"))
    neg = [p for p in labels if "_neg_" in p.name]
    assert neg and all(p.read_text().strip() == "" for p in neg)          # "not a ship": no label rows
    pos = [p for p in labels if "_neg_" not in p.name and p.read_text().strip()]
    assert pos and all(len(row.split()) == 9 for p in pos for row in p.read_text().splitlines())   # OBB rows


def test_fix_relabels_to_another_class(run):
    c, job_id, img = run
    url = f"/api/results/{job_id}/ship_detection/review"
    r = c.put(url, json={"changes": {"0": "fix: tug ", "2": "accept"}}).json()
    assert (r["fixed"], r["accepted"], r["open"]) == (1, 1, 1)
    assert c.get(url).json()["decisions"]["0"] == "fix:tug"
    assert c.put(url, json={"changes": {"1": "fix:../x"}}).status_code == 400
    assert c.put(url, json={"changes": {"1": "fix:"}}).status_code == 400
    r = c.post(url + "/apply").json()
    assert (r["fixed"], r["accepted"]) == (1, 1)
    anns = c.get(f"/api/images/{img['image_id']}/annotations").json()["annotations"]
    assert [a["class_name"] for a in anns if a["class_name"] in ("ship", "tug")].count("tug") == 1
