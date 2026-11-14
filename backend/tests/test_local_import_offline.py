"""Open a GeoTIFF from this computer: no upload, no copy, original never modified."""
import hashlib
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _tif(path, w=600, h=500, stripped=True):
    rng = np.random.default_rng(3)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint16",
                       crs="EPSG:32639", transform=from_origin(518260, 2897944, 2, 2),
                       **({} if stripped else {"tiled": True})) as f:
        f.write(rng.integers(100, 2000, (3, h, w), dtype=np.uint16))
    return path


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    data, home = tmp_path / "data", tmp_path / "home"
    (home / "Downloads" / "IMG-VHR").mkdir(parents=True)
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "UPLOAD_DIR", data / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", data / "results")
    monkeypatch.setenv("SYNTRA_IMPORT_ROOTS", str(home))
    settings.ensure_dirs()
    from app.main import app
    return TestClient(app), home / "Downloads" / "IMG-VHR", settings


def test_browse_lists_folders_and_geotiffs_only(env):
    c, folder, _ = env
    _tif(folder / "scene-MUX.tiff")
    (folder / "scene-MUX.xml").write_text("<x/>")
    (folder / "notes.txt").write_text("x")
    (folder / "sub").mkdir()
    r = c.get("/api/images/local/browse", params={"path": str(folder)}).json()
    assert [(e["name"], e["kind"]) for e in r["entries"]] == [("sub", "dir"), ("scene-MUX.tiff", "file")]
    assert r["entries"][1]["has_xml"] is True and r["parent"]


def test_outside_the_import_roots_is_refused(env, tmp_path):
    c, _, _ = env
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _tif(outside / "x.tif")
    assert c.get("/api/images/local/browse", params={"path": str(outside)}).status_code == 403
    assert c.post("/api/images/local", json={"path": str(outside / "x.tif")}).status_code == 403
    assert c.get("/api/images/local/browse", params={"path": str(env[1] / ".." / ".." / ".." / "elsewhere")}).status_code == 403


def test_only_from_this_computer(env, monkeypatch):
    import app.api.images as images

    c, folder, _ = env
    monkeypatch.setattr(images, "is_loopback", lambda req: False)
    assert c.get("/api/images/local/browse", params={"path": str(folder)}).status_code == 403
    assert c.post("/api/images/local", json={"path": str(folder / "a.tif")}).status_code == 403


def test_import_links_the_file_and_never_changes_the_original(env):
    from app.engine.optimize import optimize

    c, folder, settings = env
    src = _tif(folder / "scene-MUX.tiff", w=4400, h=300, stripped=False)   # tiled + long: optimise adds overviews in place
    (folder / "scene-MUX.xml").write_text("<x/>")
    before = _sha(src)
    r = c.post("/api/images/local", json={"path": str(src)})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["imported"] == "linked" and e["source_path"] == str(src.resolve())
    lib = settings.UPLOAD_DIR / f"{e['image_id']}_scene-MUX.tiff"
    assert lib.with_suffix(".xml").exists()                    # sensor sidecar came along
    # the background optimise has already run (TestClient runs background tasks):
    # it wrote overviews into the library file; run it again to be sure it is a no-op
    optimize(str(lib), lambda p: None)
    assert _sha(src) == before                                 # the user's original is byte-identical
    with rasterio.open(lib) as ds:
        assert ds.overviews(1)                                 # the write did happen, in the library's own copy
    with rasterio.open(src) as ds:
        assert not ds.overviews(1)


def test_same_file_opened_twice_reuses_the_entry(env):
    c, folder, _ = env
    src = _tif(folder / "a.tif")
    first = c.post("/api/images/local", json={"path": str(src)}).json()
    lib = env[2].UPLOAD_DIR / f"{first['image_id']}_a.tif"
    assert os.stat(lib).st_nlink == 2                          # small, nothing written: still one shared file
    again = c.post("/api/images/local", json={"path": str(src)}).json()
    assert again["image_id"] == first["image_id"] and again.get("reused") is True


def test_copy_when_hard_link_is_not_possible(env, monkeypatch):
    c, folder, _ = env
    src = _tif(folder / "b.tif")

    def no_link(a, b):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    e = c.post("/api/images/local", json={"path": str(src)}).json()
    assert e["imported"] == "copied" and e["width"] == 600


def test_detach_breaks_the_link_before_a_write(tmp_path):
    from app.core.image_library import detach_if_linked

    a = tmp_path / "a.bin"
    a.write_bytes(b"original")
    b = tmp_path / "b.bin"
    os.link(a, b)
    assert detach_if_linked(b) is True and os.stat(a).st_nlink == 1
    b.write_bytes(b"changed")
    assert a.read_bytes() == b"original"
    assert detach_if_linked(b) is False                        # already its own


def test_preview_is_cached_and_refreshed_when_the_file_changes(env, monkeypatch):
    import app.engine.raster_io as rio

    c, folder, settings = env
    e = c.post("/api/images/local", json={"path": str(_tif(folder / "p.tif"))}).json()
    calls = []
    real = rio.read_preview_png
    monkeypatch.setattr(rio, "read_preview_png", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    url = f"/api/images/{e['image_id']}/preview.png"
    first = c.get(url)
    assert first.status_code == 200 and first.headers["etag"]
    second = c.get(url)                                        # a refresh: served from the disk cache
    assert second.content == first.content and len(calls) == 1
    assert c.get(url, headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    lib = settings.UPLOAD_DIR / f"{e['image_id']}_p.tif"
    os.utime(lib, ns=(os.stat(lib).st_atime_ns, os.stat(lib).st_mtime_ns + 10**9))   # the file changed
    third = c.get(url)
    assert third.status_code == 200 and len(calls) == 2 and third.headers["etag"] != first.headers["etag"]
