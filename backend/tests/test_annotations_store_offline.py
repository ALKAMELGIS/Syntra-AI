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

    # saved per class in the library, as GeoJSON with real coordinates
    from app.core.config import settings
    key = store.image_key_for(img)
    tree = json.loads((settings.DATA_DIR / "classes" / "tree" / "labels" / f"{key}.geojson").read_text())
    assert len(tree["features"]) == 1 and tree["features"][0]["geometry"]["type"] == "Polygon"
    assert (settings.DATA_DIR / "classes" / "building" / "class.json").exists()

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


def test_update_keeps_id_and_moves_the_label_between_classes(tmp_path):
    img = tmp_path / "ortho.tif"
    img.write_bytes(b"fake")
    box = [[20.1, 10.9], [20.2, 10.9], [20.2, 10.8], [20.1, 10.8]]
    a = store.add_annotation(img, "boat", box, BOUNDS, WIDTH, HEIGHT)
    moved = [[20.3, 10.7], [20.4, 10.7], [20.4, 10.6], [20.3, 10.6], [20.35, 10.65]]
    u = store.update_annotation(img, a["id"], BOUNDS, WIDTH, HEIGHT, points_lonlat=moved, class_name="ship")
    assert u["id"] == a["id"] and u["created_at"] == a["created_at"] and u["shape"] == "edited"
    back = store.load_annotations(img)
    assert len(back) == 1 and back[0]["class_name"] == "ship" and back[0]["points_lonlat"] == moved
    assert round(back[0]["points_px"][0][0]) == 300                      # pixel grid follows the new outline


def test_update_refuses_bad_shapes_and_unknown_ids(tmp_path):
    import pytest
    img = tmp_path / "ortho.tif"
    img.write_bytes(b"fake")
    box = [[20.1, 10.9], [20.2, 10.9], [20.2, 10.8]]
    a = store.add_annotation(img, "boat", box, BOUNDS, WIDTH, HEIGHT)
    assert store.update_annotation(img, "nope", BOUNDS, WIDTH, HEIGHT, class_name="x") is None
    with pytest.raises(ValueError):
        store.update_annotation(img, a["id"], BOUNDS, WIDTH, HEIGHT, points_lonlat=box[:2])
    with pytest.raises(ValueError):
        store.update_annotation(img, a["id"], BOUNDS, WIDTH, HEIGHT, points_lonlat=[[30, 30], [31, 30], [31, 31]])
    assert store.load_annotations(img)[0]["points_lonlat"] == box        # unchanged after a refusal
