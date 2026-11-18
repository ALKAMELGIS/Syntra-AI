"""Refine outlines with SAM (stand-in engine) and score a layer on accuracy areas."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("cv2")
pytest.importorskip("shapely")
from rasterio.transform import from_origin  # noqa: E402


class RectEngine:
    """Decodes the box prompt as a rectangle shrunk by 3 px on every side (a tighter outline)."""

    def __init__(self, score=0.95):
        self.score = score

    def encode(self, rgb):
        return {"size": rgb.shape[:2]}

    def decode(self, emb, points, labels):
        assert list(labels) == [2, 3]
        h, w = emb["size"]
        (x0, y0), (x1, y1) = points
        m = np.zeros((h, w), bool)
        m[int(y0) + 3:int(y1) - 3, int(x0) + 3:int(x1) - 3] = True
        return m, self.score


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "port.tif"
    with rasterio.open(p, "w", driver="GTiff", width=600, height=400, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(300000, 2800000, 0.5, 0.5)) as f:
        f.write(np.full((3, 400, 600), 90, np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("port.tif", fh, "image/tiff")}).json()
    yield c, img, settings
    from app.engine import click_segment
    click_segment.set_engine(None, None)


def ll(img, x, y):
    from app.annotations import store
    return list(store.pixel_to_lonlat(x, y, img["bounds_wgs84"], img["width"], img["height"]))


def rect(img, x0, y0, x1, y1, **props):
    ring = [ll(img, x0, y0), ll(img, x1, y0), ll(img, x1, y1), ll(img, x0, y1)]
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}


def run_with_result(c, img, settings, feats):
    from app.core.jobs import get_store

    jid = get_store().create("predict", model_id="m1", image_id=img["image_id"], status="completed")
    d = settings.RESULTS_DIR / jid
    d.mkdir(parents=True)
    (d / "m1.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return jid


def test_refine_tightens_outlines_and_keeps_unsure_ones(env):
    from app.engine import click_segment

    c, img, settings = env
    click_segment.set_engine(RectEngine())
    jid = run_with_result(c, img, settings, [rect(img, 100, 100, 160, 140, **{"class": "ship", "area_m2": 1})])
    r = c.post(f"/api/results/{jid}/m1/refine")
    assert r.status_code == 200, r.text
    job = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job.get("error")
    assert job["stats"]["refined"] == 1
    out = c.get(f"/api/results/{jid}/m1__refined").json()
    f = out["features"][0]
    assert f["properties"]["outline"] == "sam" and f["properties"]["sam_score"] == 0.95
    # 54 x 34 px at 0.5 m, re-measured on the new outline (the old area_m2 is gone)
    assert f["properties"]["area_m2"] == pytest.approx(54 * 34 * 0.25, rel=0.1)

    click_segment.set_engine(RectEngine(score=0.3))         # SAM unsure: the detector outline stays
    jid2 = run_with_result(c, img, settings, [rect(img, 100, 100, 160, 140, **{"class": "ship"})])
    rid = c.post(f"/api/results/{jid2}/m1/refine").json()["job_id"]
    assert c.get(f"/api/jobs/{rid}").json()["stats"]["kept"] == 1
    f2 = json.loads((settings.RESULTS_DIR / jid2 / "m1__refined.geojson").read_text())["features"][0]
    assert f2["properties"]["outline"] == "detector" and f2["properties"]["refine_note"] == "SAM not sure"


def test_refine_needs_sam_and_refuses_paths_outside(env):
    from app.engine import click_segment

    c, img, settings = env
    click_segment.set_engine(None, None)
    jid = run_with_result(c, img, settings, [rect(img, 10, 10, 50, 50)])
    r = c.post(f"/api/results/{jid}/m1/refine")
    assert r.status_code == 409 and "SAM" in r.json()["detail"]
    assert c.post("/api/results/..%2F..%2Fx/m1/refine").status_code in (400, 403, 404)
    assert c.post(f"/api/results/{jid}/..%2F..%2Fsecret/score").status_code in (400, 403, 404)


def test_score_matches_inside_accuracy_areas(env):
    from app.annotations import store

    c, img, settings = env
    path = next(settings.UPLOAD_DIR.rglob("port.tif"))
    jid = run_with_result(c, img, settings, [
        rect(img, 100, 100, 160, 140, **{"class": "ship"}),        # matches label 1
        rect(img, 300, 100, 340, 120, **{"class": "ship"}),        # false positive
        rect(img, 500, 300, 560, 340, **{"class": "ship"}),        # outside the area: not counted
    ])
    r = c.post(f"/api/results/{jid}/m1/score", json={})
    assert r.status_code == 409 and "accuracy area" in r.json()["detail"]

    def add(cls, x0, y0, x1, y1):
        store.add_annotation(path, cls, [ll(img, x0, y0), ll(img, x1, y0), ll(img, x1, y1), ll(img, x0, y1)],
                             img["bounds_wgs84"], img["width"], img["height"])
    add(store.CHECK_AREA, 0, 0, 450, 250)
    add("ship", 102, 102, 160, 140)
    add("ship", 200, 180, 240, 220)                                # missed
    res = c.post(f"/api/results/{jid}/m1/score", json={"class_name": "ship"}).json()
    assert (res["results"], res["labels"], res["matched"]) == (2, 2, 1)
    assert res["precision"] == 0.5 and res["recall"] == 0.5 and res["mean_iou"] > 0.9
