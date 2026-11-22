"""Click to segment, without the real SAM weights: the API and caching run against
a stand-in engine, and the ONNX wrapper runs against two tiny graphs with the
SAM 2.1 export's input and output names."""
import io
import json
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("cv2")
pytest.importorskip("shapely")
from rasterio.transform import from_origin  # noqa: E402


class DiscEngine:
    """Encodes nothing; decodes a disc of radius 30 px (window pixels) around the first add-click."""

    def __init__(self):
        self.encodes = 0

    def encode(self, rgb):
        self.encodes += 1
        return {"size": rgb.shape[:2]}

    def decode(self, emb, points, labels):
        h, w = emb["size"]
        yy, xx = np.mgrid[:h, :w]
        x, y = points[labels == 1][0]
        return (xx - x) ** 2 + (yy - y) ** 2 <= 30 ** 2, 0.9


@pytest.fixture()
def app_img(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "site.tif"
    with rasterio.open(p, "w", driver="GTiff", width=2000, height=1500, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(300000, 2800000, 0.3, 0.3)) as f:
        f.write(np.full((3, 1500, 2000), 120, np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("site.tif", fh, "image/tiff")}).json()
    return c, img


def lonlat(img, x, y):
    from app.annotations import store
    return list(store.pixel_to_lonlat(x, y, img["bounds_wgs84"], img["width"], img["height"]))


def test_not_installed_says_how_to_install(app_img):
    from app.engine import click_segment

    c, img = app_img
    click_segment.set_engine(None, None)
    st = c.get("/api/segment/status").json()
    assert st["ready"] is False and "Install the click tool" in st["reason"]
    b = img["bounds_wgs84"]
    r = c.post(f"/api/images/{img['image_id']}/segment",
               json={"clicks": [lonlat(img, 500, 500) + [1]], "view": [b["west"], b["south"], b["east"], b["north"]]})
    assert r.status_code == 409 and "Install" in r.json()["detail"]


def test_click_returns_a_fitted_outline_and_reuses_the_view(app_img):
    from app.engine import click_segment

    c, img = app_img
    eng = DiscEngine()
    click_segment.set_engine(eng)
    w0, n0 = lonlat(img, 0, 0)
    e1, s1 = lonlat(img, 1000, 1000)                       # view: a 1000 px square, no resampling
    body = {"clicks": [lonlat(img, 400, 300) + [1]], "view": [w0, s1, e1, n0], "class_name": "tank"}
    r = c.post(f"/api/images/{img['image_id']}/segment", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["shape"] == "circle" and res["vertices"] == 64             # tank -> circle
    xs, ys = np.array(res["points_px"]).T
    assert xs.mean() == pytest.approx(400, abs=1.5) and ys.mean() == pytest.approx(300, abs=1.5)
    assert res["area_m2"] == pytest.approx(np.pi * 30 ** 2 * 0.09, rel=0.05)  # GSD 0.3 m
    assert res["timings"]["cached"] is False
    body["clicks"].append(lonlat(img, 420, 300) + [0])
    r2 = c.post(f"/api/images/{img['image_id']}/segment", json=body).json()
    assert eng.encodes == 1 and r2["timings"]["cached"] is True          # same view: decoder only


def test_big_views_are_resampled_and_mapped_back(app_img):
    from app.engine import click_segment

    c, img = app_img
    click_segment.set_engine(DiscEngine())
    b = img["bounds_wgs84"]                                  # whole image: 2000 px -> 1024 window
    r = c.post(f"/api/images/{img['image_id']}/segment", json={
        "clicks": [lonlat(img, 1500, 1000) + [1]], "view": [b["west"], b["south"], b["east"], b["north"]], "shape": "polygon"})
    res = r.json()
    xs, ys = np.array(res["points_px"]).T
    assert res["timings"]["scale"] == pytest.approx(2000 / 1024, rel=1e-3)
    assert xs.mean() == pytest.approx(1500, abs=3) and ys.mean() == pytest.approx(1000, abs=3)
    assert (xs.max() - xs.min()) == pytest.approx(60 * 2000 / 1024, rel=0.08)   # 30 px radius in window pixels


def test_outline_fits_a_drawn_ring(app_img):
    c, img = app_img
    ring = [lonlat(img, x, y) for x, y in [(100, 100), (300, 104), (298, 200), (180, 199), (179, 260), (99, 258)]]
    r = c.post(f"/api/images/{img['image_id']}/outline", json={"points_lonlat": ring, "class_name": "building"}).json()
    assert r["shape"] == "right_angles" and r["vertices"] == 6


def test_install_unpacks_the_zip_and_refuses_unsafe_paths(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.engine import click_segment

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    def zipped(names):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for n in names:
                z.writestr(n, b"onnx")
        data = buf.getvalue()

        class R:
            headers = {"Content-Length": str(len(data))}
            def __init__(self): self.b = io.BytesIO(data)
            def read(self, n): return self.b.read(n)
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return lambda req, timeout=None: R()

    seen = []
    out = click_segment.install("sam2.1_tiny", progress=seen.append,
                                opener=zipped(["sam2.1_tiny/sam2.1_hiera_tiny.encoder.onnx", "sam2.1_tiny/sam2.1_hiera_tiny.decoder.onnx"]))
    assert click_segment.model_files("sam2.1_tiny") and seen[-1] == 100.0 and out["already"] is False
    with pytest.raises(click_segment.SegmentUnavailable, match="unsafe"):
        click_segment.install("sam2.1_small", opener=zipped(["../../evil.onnx"]))
    assert not (tmp_path / "evil.onnx").exists()


def test_onnx_wrapper_matches_the_sam21_interface(tmp_path):
    """Two tiny graphs with the export's tensor names: the decoder draws a disc
    of radius 40 (in 1024-space) around the first point, so a right answer
    proves names, scaling to 1024 and resizing back are all correct."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from app.engine.click_segment import SamOnnx

    f = TensorProto.FLOAT
    enc = helper.make_graph(
        [helper.make_node("ReduceMean", ["image"], ["image_embed"], keepdims=1),
         helper.make_node("Identity", ["image_embed"], ["high_res_feats_0"]),
         helper.make_node("Identity", ["image_embed"], ["high_res_feats_1"])],
        "enc", [helper.make_tensor_value_info("image", f, [1, 3, 1024, 1024])],
        [helper.make_tensor_value_info(n, f, None) for n in ("high_res_feats_0", "high_res_feats_1", "image_embed")])
    yy, xx = np.mgrid[:256, :256].astype(np.float32) * 4 + 2           # low-res mask grid in 1024-space
    grid = np.stack([xx, yy], -1)[None, None]                             # 1,1,256,256,2
    nodes = [
        helper.make_node("Slice", ["point_coords", "s0", "s1", "ax"], ["p0"]),          # first point (1,1,2)
        helper.make_node("Reshape", ["p0", "shp"], ["p"]),                               # 1,1,1,1,2
        helper.make_node("Sub", ["grid", "p"], ["d"]),
        helper.make_node("Mul", ["d", "d"], ["d2"]),
        helper.make_node("ReduceSum", ["d2", "ax4"], ["r2"], keepdims=0),               # 1,1,256,256
        helper.make_node("Sub", ["rr", "r2"], ["masks"]),                                # >0 inside radius 40
        helper.make_node("ReduceMax", ["masks"], ["mx"], keepdims=1, axes=[2, 3]),
        helper.make_node("Reshape", ["mx", "shp2"], ["iou_predictions"]),
        helper.make_node("ReduceSum", ["point_labels"], ["_l"], keepdims=1),            # inputs used, so kept
        helper.make_node("ReduceSum", ["image_embed"], ["_e"], keepdims=1),
    ]
    inits = [numpy_helper.from_array(v, n) for n, v in {
        "grid": grid, "s0": np.array([0], np.int64), "s1": np.array([1], np.int64), "ax": np.array([1], np.int64),
        "shp": np.array([1, 1, 1, 1, 2], np.int64), "ax4": np.array([4], np.int64),
        "rr": np.array(40.0 ** 2, np.float32), "shp2": np.array([1, 1], np.int64)}.items()]
    ins = [helper.make_tensor_value_info("image_embed", f, None), helper.make_tensor_value_info("high_res_feats_0", f, None),
           helper.make_tensor_value_info("high_res_feats_1", f, None), helper.make_tensor_value_info("point_coords", f, [1, None, 2]),
           helper.make_tensor_value_info("point_labels", f, [1, None]), helper.make_tensor_value_info("mask_input", f, [1, 1, 256, 256]),
           helper.make_tensor_value_info("has_mask_input", f, [1])]
    dec = helper.make_graph(nodes, "dec", ins, [helper.make_tensor_value_info("masks", f, None),
                                                helper.make_tensor_value_info("iou_predictions", f, None)], inits)
    for g, name in ((enc, "m.encoder.onnx"), (dec, "m.decoder.onnx")):
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, tmp_path / name)
    sam = SamOnnx(tmp_path / "m.encoder.onnx", tmp_path / "m.decoder.onnx")
    emb = sam.encode(np.zeros((500, 800, 3), np.uint8))                  # a 800x500 window
    mask, score = sam.decode(emb, np.array([[400.0, 250.0], [10.0, 10.0]]), np.array([1, 0]))
    assert mask.shape == (500, 800) and score > 0
    ys, xs = np.nonzero(mask)
    assert xs.mean() == pytest.approx(400, abs=2) and ys.mean() == pytest.approx(250, abs=2)
    # radius 40 in 1024-space is 40*800/1024 = 31 px across x, 40*500/1024 = 20 px down y
    assert (xs.max() - xs.min()) / 2 == pytest.approx(31, abs=2) and (ys.max() - ys.min()) / 2 == pytest.approx(20, abs=2)


@pytest.fixture(autouse=True)
def _reset_engine():
    yield
    from app.engine import click_segment
    click_segment.set_engine(None, None)
    json.dumps({})


class BoxEngine:
    """Decodes the box it is given (labels 2 and 3) as the mask; records the labels."""

    def __init__(self):
        self.labels = None

    def encode(self, rgb):
        return {"size": rgb.shape[:2]}

    def decode(self, emb, points, labels):
        self.labels = labels.tolist()
        h, w = emb["size"]
        (x0, y0), (x1, y1) = points[labels == 2][0], points[labels == 3][0]
        yy, xx = np.mgrid[:h, :w]
        return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1), 0.8


def test_a_box_alone_segments_the_object_inside_it(app_img):
    from app.engine import click_segment

    c, img = app_img
    eng = BoxEngine()
    click_segment.set_engine(eng)
    b = img["bounds_wgs84"]
    view = [b["west"], b["south"], b["east"], b["north"]]
    r = c.post(f"/api/images/{img['image_id']}/segment", json={
        "clicks": [lonlat(img, 400, 300) + [2], lonlat(img, 600, 500) + [3]], "view": view, "shape": "box"})
    assert r.status_code == 200, r.text
    assert eng.labels == [2.0, 3.0]                                   # SAM's box corner labels reach the decoder
    xs = [p[0] for p in r.json()["points_px"]]
    ys = [p[1] for p in r.json()["points_px"]]
    assert r.json()["shape"] == "box" and len(xs) == 4
    assert abs(min(xs) - 400) < 12 and abs(max(xs) - 600) < 12 and abs(min(ys) - 300) < 12 and abs(max(ys) - 500) < 12
    # one corner alone is not a box, and labels outside 0-3 are refused
    assert c.post(f"/api/images/{img['image_id']}/segment",
                  json={"clicks": [lonlat(img, 400, 300) + [2]], "view": view}).status_code == 422
    assert c.post(f"/api/images/{img['image_id']}/segment",
                  json={"clicks": [lonlat(img, 400, 300) + [7]], "view": view}).status_code == 400


def test_box_shape_is_axis_aligned():
    from shapely.geometry import Polygon

    from app.engine.shapes import RefineOptions, refine

    diamond = Polygon([(50, 0), (100, 50), (50, 100), (0, 50)])
    res = refine(diamond, RefineOptions(shape="box", px_m=1.0, simplify_m=0.5))
    assert res["shape"] == "box" and res["polygon"].bounds == (0.0, 0.0, 100.0, 100.0)
    assert res["warning"] is None                                     # a box is asked for: no fall-back to the trace


class BlobsEngine:
    """Three round objects (radius 25 px in window pixels); a point inside one returns it."""

    CENTRES = [(100, 100), (300, 150), (200, 300)]

    def encode(self, rgb):
        return {"size": rgb.shape[:2]}

    def decode(self, emb, points, labels):
        h, w = emb["size"]
        yy, xx = np.mgrid[:h, :w]
        x, y = points[0]
        for cx, cy in self.CENTRES:
            if (x - cx) ** 2 + (y - cy) ** 2 <= 25 ** 2:
                return (xx - cx) ** 2 + (yy - cy) ** 2 <= 25 ** 2, 0.95
        return np.zeros((h, w), bool), 0.2


def test_everything_in_view_finds_each_object_once(app_img):
    from app.engine import click_segment

    c, img = app_img
    click_segment.set_engine(BlobsEngine())
    view = [*lonlat(img, 0, 400)[:1], lonlat(img, 0, 400)[1], *lonlat(img, 400, 0)]     # west, south, east, north
    r = c.post(f"/api/images/{img['image_id']}/segment-all", json={"view": view, "grid": 24, "shape": "polygon"})
    assert r.status_code == 200, r.text
    objs = r.json()["objects"]
    assert len(objs) == 3                                         # many grid points hit each blob: kept once
    centres = sorted((round(sum(p[0] for p in o["points_px"]) / len(o["points_px"])),
                      round(sum(p[1] for p in o["points_px"]) / len(o["points_px"]))) for o in objs)
    assert all(abs(a - b) <= 3 for (cx, cy), (ex, ey) in zip(centres, sorted(BlobsEngine.CENTRES)) for a, b in ((cx, ex), (cy, ey)))
    assert all(o["points_lonlat"] and o["score"] >= 0.9 for o in objs)
