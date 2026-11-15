"""The per-class data folder and the one-time move into it."""
import json

import pytest


def test_old_uploads_move_into_image_folders_and_labels_into_the_class_library(tmp_path, monkeypatch):
    from app.annotations import store
    from app.core.config import settings
    from app.core.image_library import list_library
    from app.core.migrate import migrate_layout

    data = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "UPLOAD_DIR", data / "images")
    old = data / "uploads"
    old.mkdir(parents=True)
    iid = "7f3a91c2-1111-2222-3333-444455556666"
    tif = old / f"{iid}_dubai harbour.tif"
    tif.write_bytes(b"TIFF")
    (old / f"{iid}_dubai harbour.tif.meta.json").write_text(json.dumps({"image_id": iid, "filename": "dubai harbour.tif", "path": str(tif)}))
    (old / f"{iid}_dubai harbour.xml").write_text("<x/>")
    ring = [[55.0, 25.1], [55.001, 25.1], [55.001, 25.099]]
    (old / f"{iid}_dubai harbour.tif.annotations.json").write_text(json.dumps([
        {"id": "a", "class_name": "Ship", "points_lonlat": ring, "points_px": [[0, 0], [1, 0], [1, 1]], "created_at": "x"},
        {"id": "b", "class_name": "__check_area__", "points_lonlat": ring, "points_px": [[0, 0], [1, 0], [1, 1]], "created_at": "x"},
    ]))

    rep = migrate_layout(data)
    assert rep["moved_images"] == 1 and rep["labels"] == 2 and not rep["errors"]
    new = data / "images" / "7f3a91c2" / "dubai harbour.tif"
    assert new.read_bytes() == b"TIFF" and (new.parent / "dubai harbour.xml").exists()
    assert not old.exists()                                                  # emptied and removed
    lib = list_library(settings.UPLOAD_DIR)
    assert lib[0]["path"] == str(new) and lib[0]["image_id"] == iid
    ship = json.loads((data / "classes" / "ship" / "labels" / "7f3a91c2.geojson").read_text())
    assert ship["features"][0]["properties"]["class_name"] == "Ship"
    areas = json.loads((data / "classes" / "_areas" / "7f3a91c2.geojson").read_text())
    assert areas["features"][0]["properties"]["role"] == "accuracy"
    assert [a["id"] for a in store.load_annotations(new)] == ["a", "b"]
    assert (new.parent / "dubai harbour.tif.annotations.json.moved").exists()   # the old file is kept
    assert json.loads((data / "migration.json").read_text())["moves"]           # every move is recorded
    assert migrate_layout(data)["moved_images"] == 0                         # second start: nothing to do


def test_versions_and_approved_pointer(tmp_path, monkeypatch):
    from app.core import layout
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    assert layout.group_slug(["Ship", "building"]) == "building+ship"
    v1, p1 = layout.next_version(layout.datasets_dir() / "ship")
    v2, _ = layout.next_version(layout.datasets_dir() / "ship")
    assert (v1, v2) == ("v001", "v002") and p1.is_dir()
    assert [p.name for p in layout.versions(layout.datasets_dir() / "ship")] == ["v001", "v002"]
    layout.set_approved("ship", "v002")
    assert layout.approved("ship") == "v002"
    layout.set_approved("ship", None)
    assert layout.approved("ship") is None
    assert layout.class_slug("../../etc") == "etc"                           # never a path outside classes/


def test_dataset_goes_to_its_class_folder_with_a_manifest(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from fastapi.testclient import TestClient
    from rasterio.transform import from_origin

    from app.core.config import settings
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "site.tif"
    with rasterio.open(p, "w", driver="GTiff", width=3000, height=2000, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(55.0, 25.1, 1e-5, 1e-5)) as f:
        f.write(np.full((3, 2000, 3000), 100, np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("site.tif", fh, "image/tiff")}).json()

    def ring(x, y, s=20):
        return [[55.0 + a * 1e-5, 25.1 - b * 1e-5] for a, b in ((x, y), (x + s, y), (x + s, y + s), (x, y + s))]

    for x in range(100, 2900, 250):
        for y in range(100, 1300, 250):
            assert c.post(f"/api/images/{img['image_id']}/annotations", json={"class_name": "Ship", "points_lonlat": ring(x, y)}).status_code == 200
    c.post(f"/api/images/{img['image_id']}/annotations", json={"class_name": "__check_area__", "points_lonlat": ring(1500, 1500, 450)})
    c.post(f"/api/images/{img['image_id']}/annotations", json={"class_name": "Ship", "points_lonlat": ring(1600, 1600)})
    r = c.post("/api/training/export-dataset", json={"image_id": img["image_id"], "task": "detect", "augment": False})
    st = c.get(f"/api/training/export-dataset/{r.json()['job_id']}").json()
    assert st["status"] == "completed", st
    out = settings.DATA_DIR / "datasets" / "ship" / "v001"
    assert st["data_yaml"] == str(out / "data.yaml")
    m = json.loads((out / "manifest.json").read_text())
    assert m["dataset"] == "ship" and m["version"] == "v001" and m["split"]["method"] == "spatial blocks"
    assert m["counts"]["test"]["objects"] >= 1 and m["counts"]["val"]["images"] >= 1
    assert m["labels"][0]["file"].startswith("ship/labels/") and len(m["labels"][0]["sha256"]) == 64
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir() and (out / "labels" / split).is_dir()


def test_library_builds_a_class_from_every_image_and_approves_only_trained_versions(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from fastapi.testclient import TestClient
    from rasterio.transform import from_origin

    from app.core import layout
    from app.core.config import settings
    from app.main import app

    c = TestClient(app)
    ids = []
    for k in range(2):
        p = tmp_path / f"site{k}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=2400, height=1600, count=3, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(55.0 + k, 25.1, 1e-5, 1e-5)) as f:
            f.write(np.full((3, 1600, 2400), 100, np.uint8))
        with open(p, "rb") as fh:
            img = c.post("/api/images", files={"file": ("image.tif", fh, "image/tiff")}).json()   # same file name twice
        ids.append(img["image_id"])
        for x in range(100, 2300, 300):
            for y in range(100, 1500, 300):
                ring = [[55.0 + k + a * 1e-5, 25.1 - b * 1e-5] for a, b in ((x, y), (x + 20, y), (x + 20, y + 20), (x, y + 20))]
                c.post(f"/api/images/{img['image_id']}/annotations", json={"class_name": "Palm", "points_lonlat": ring})
    lib = c.get("/api/library").json()
    palm = next(x for x in lib["classes"] if x["slug"] == "palm")
    assert palm["samples"] == 2 * 8 * 5 and palm["images"] == 2 and palm["datasets"] == []

    r = c.post("/api/library/palm/build", json={"augment": False})
    assert r.status_code == 200 and r.json()["images"] == 2
    st = c.get(f"/api/training/export-dataset/{r.json()['job_id']}").json()
    assert st["status"] == "completed", st
    out = settings.DATA_DIR / "datasets" / "palm" / "v001"
    names = [f.name for f in (out / "images" / "train").iterdir()]
    assert len({n.split("_")[0] for n in names}) == 2                     # tiles of both images, no clash
    palm = next(x for x in c.get("/api/library").json()["classes"] if x["slug"] == "palm")
    ds = palm["datasets"][0]
    assert ds["version"] == "v001" and ds["images"] == 2 and ds["export_job_id"] == r.json()["job_id"]
    assert palm["new_since_last"] == 0

    assert c.post("/api/library/palm/approve", json={"version": "v001"}).status_code == 404   # no model yet
    _, mdir = layout.next_version(layout.models_dir() / "palm")
    assert c.post("/api/library/palm/approve", json={"version": "v001"}).status_code == 409   # no weights
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.onnx").write_bytes(b"x")
    assert c.post("/api/library/palm/approve", json={"version": "v001"}).json()["approved"] == "v001"
    palm = next(x for x in c.get("/api/library").json()["classes"] if x["slug"] == "palm")
    assert palm["models"][0]["approved"] is True
    assert c.post("/api/library/..%2Fx/approve", json={"version": None}).status_code == 404


def test_only_the_owner_builds_and_approves(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.delenv("SYNTRA_AUTH", raising=False)
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    monkeypatch.delenv("SYNTRA_ALLOW_SIGNUP", raising=False)
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={"name": "Owner", "email": "o@example.com", "password": "owner-pass-1"})
    assert r.status_code == 200, r.text
    assert c.get("/api/library").json()["owner"] is True
    c.post("/api/auth/signout")
    assert c.post("/api/auth/signup", json={"name": "User", "email": "u@example.com", "password": "user-pass-12"}).status_code == 200
    assert c.get("/api/library").json()["owner"] is False
    assert c.post("/api/library/ship/build", json={}).status_code == 403
    assert c.post("/api/library/ship/approve", json={"version": None}).status_code == 403
