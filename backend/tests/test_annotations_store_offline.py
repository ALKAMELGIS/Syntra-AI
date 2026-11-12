"""
Offline tests for app.annotations.store: lon/lat <-> pixel conversion
(matching the linear stretch the frontend's Leaflet <ImageOverlay> uses
across an image's bounds_wgs84 rectangle) and the sidecar-JSON persistence
for per-image class-labeled annotations. Pure stdlib — no rasterio/PIL.
"""
import json

from app.annotations import store

BOUNDS = {"south": 10.0, "west": 20.0, "north": 11.0, "east": 22.0}
WIDTH, HEIGHT = 2000, 1000


def test_lonlat_to_pixel_corners():
    # top-left of the image == north-west corner
    x, y = store.lonlat_to_pixel(BOUNDS["west"], BOUNDS["north"], BOUNDS, WIDTH, HEIGHT)
    assert (round(x), round(y)) == (0, 0)
    # bottom-right == south-east corner
    x, y = store.lonlat_to_pixel(BOUNDS["east"], BOUNDS["south"], BOUNDS, WIDTH, HEIGHT)
    assert (round(x), round(y)) == (WIDTH, HEIGHT)


def test_lonlat_pixel_round_trip():
    lon, lat = 21.3, 10.6
    x, y = store.lonlat_to_pixel(lon, lat, BOUNDS, WIDTH, HEIGHT)
    lon2, lat2 = store.pixel_to_lonlat(x, y, BOUNDS, WIDTH, HEIGHT)
    assert round(lon2, 9) == round(lon, 9)
    assert round(lat2, 9) == round(lat, 9)


def test_add_list_delete_clear_round_trip(tmp_path):
    img = tmp_path / "ortho.tif"
    img.write_bytes(b"fake")  # content irrelevant; only the path is used as a key

    assert store.load_annotations(img) == []

    box = [[20.1, 10.9], [20.2, 10.9], [20.2, 10.8], [20.1, 10.8]]
    a1 = store.add_annotation(img, "tree", box, BOUNDS, WIDTH, HEIGHT)
    a2 = store.add_annotation(img, "building", box, BOUNDS, WIDTH, HEIGHT)

    loaded = store.load_annotations(img)
    assert len(loaded) == 2
    assert {a["id"] for a in loaded} == {a1["id"], a2["id"]}
    assert all(len(a["points_px"]) == 4 for a in loaded)

    # sidecar actually persisted to disk, not just in-memory
    sidecar = store.sidecar_path(img)
    assert sidecar.exists()
    on_disk = json.loads(sidecar.read_text())
    assert len(on_disk) == 2

    assert store.class_names_used(loaded) == ["tree", "building"]

    assert store.delete_annotation(img, a1["id"]) is True
    assert store.delete_annotation(img, "not-a-real-id") is False
    assert len(store.load_annotations(img)) == 1

    store.clear_annotations(img)
    assert store.load_annotations(img) == []


def test_add_annotation_rejects_degenerate_shape(tmp_path):
    img = tmp_path / "ortho.tif"
    import pytest
    with pytest.raises(ValueError):
        store.add_annotation(img, "tree", [[20.0, 10.0], [20.1, 10.1]], BOUNDS, WIDTH, HEIGHT)


def test_to_export_payload_assigns_stable_class_ids(tmp_path):
    img = tmp_path / "ortho.tif"
    box = [[20.1, 10.9], [20.2, 10.9], [20.2, 10.8], [20.1, 10.8]]
    store.add_annotation(img, "tree", box, BOUNDS, WIDTH, HEIGHT)
    store.add_annotation(img, "building", box, BOUNDS, WIDTH, HEIGHT)
    store.add_annotation(img, "tree", box, BOUNDS, WIDTH, HEIGHT)

    anns = store.load_annotations(img)
    names, payload = store.to_export_payload(anns)
    assert names == ["tree", "building"]
    assert [p["class_id"] for p in payload] == [0, 1, 0]
    assert all("points_px" in p for p in payload)
