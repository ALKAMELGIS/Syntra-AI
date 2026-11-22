"""
Click to segment: SAM 2.1 (Apache 2.0) as two ONNX graphs, run with onnxruntime.

    encoder  the visible part of the image (resampled to 1024 px) -> features
             heavy; done once per view and cached
    decoder  features + clicks (add / remove) -> mask scores
             light; runs on every click

The ONNX files are the SAM 2.1 exports AnyLabeling publishes on Hugging Face
(encoder input: float32 1x3x1024x1024, ImageNet-normalised RGB resized
without padding; encoder outputs high_res_feats_0, high_res_feats_1,
image_embed; decoder inputs image_embed, high_res_feats_0/1, point_coords and
point_labels in 1024-space, mask_input, has_mask_input; decoder outputs mask
logits and their scores). Inputs and outputs are matched by name, so a
re-export with the same names works too. No AnyLabeling code is used (it is
GPL); this file only follows the published tensor interface.

The mask then goes through app.engine.shapes: traced, and fitted to the
shape the class has (right angles, rectangle, circle, smooth).
"""
from __future__ import annotations

import io
import threading
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

SIZE = 1024
LARGE_MARGIN = 0.12      # one click: a mask this close to the best score may win by being larger
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODELS = {
    "sam2.1_tiny": {
        "label": "SAM 2.1 tiny",
        "url": "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main/sam2.1_hiera_tiny_20260221.zip",
        "approx_mb": 120,
        "licence": "Apache-2.0 (Meta SAM 2.1 weights)",
    },
    "sam2.1_small": {
        "label": "SAM 2.1 small",
        "url": "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main/sam2.1_hiera_small_20260221.zip",
        "approx_mb": 150,
        "licence": "Apache-2.0 (Meta SAM 2.1 weights)",
    },
}
DEFAULT_MODEL = "sam2.1_tiny"
PREFERRED = ("sam2.1_small", "sam2.1_tiny")


class SegmentUnavailable(RuntimeError):
    pass


def models_dir() -> Path:
    from app.core.config import settings

    return settings.DATA_DIR / "engines" / "segment"      # models/ holds the classes' trained models


def model_files(name: str = DEFAULT_MODEL) -> tuple[Path, Path] | None:
    d = models_dir() / name
    if not d.is_dir():
        return None
    enc = sorted(d.rglob("*encoder*.onnx"))
    dec = sorted(d.rglob("*decoder*.onnx"))
    return (enc[0], dec[0]) if enc and dec else None


def status() -> dict:
    installed = {n: model_files(n) is not None for n in MODELS}
    name = next((n for n in PREFERRED if installed[n]), None)     # the more accurate one once it is there
    try:
        import onnxruntime  # noqa: F401
        ort_ok = True
    except ImportError:
        ort_ok = False
    reason = None
    if not ort_ok:
        reason = "onnxruntime is missing: run the start script again"
    elif not name and _installing:
        reason = "Installing the click tool… try again in a minute"
    elif not name:
        reason = f"Install the click tool once ({MODELS[DEFAULT_MODEL]['label']}, about {MODELS[DEFAULT_MODEL]['approx_mb']} MB)"
    if _pinned and _engine is not None:                       # a stand-in engine (tests, demos): ready
        name, reason, ort_ok = None, None, True
    return {
        "ready": (bool(name) and ort_ok) or (_pinned and _engine is not None),
        "model": name,
        "label": MODELS[name]["label"] if name else None,
        "models": {n: {**{k: v for k, v in m.items() if k != "url"}, "installed": installed[n]} for n, m in MODELS.items()},
        "reason": reason,
        "installing": bool(_installing),
        "loaded": _engine is not None,
        "timings": dict(_last_timings),
    }


_install_lock = threading.Lock()
_installing: set = set()


def install(name: str = DEFAULT_MODEL, progress=None, opener=None) -> dict:
    """Download the model zip once and unpack it into the data folder (one download at a time)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'")
    _installing.add(name)
    try:
        with _install_lock:
            return _install(name, progress, opener)
    finally:
        _installing.discard(name)


def _install(name: str, progress=None, opener=None) -> dict:
    dest = models_dir() / name
    if model_files(name):
        return {"model": name, "path": str(dest), "already": True}
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(MODELS[name]["url"], headers={"User-Agent": "Syntra-AI"})
    buf = io.BytesIO()
    try:
        with opener(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                if progress and total:
                    progress(min(99.0, 100.0 * buf.tell() / total))
    except OSError as exc:
        reason = str(getattr(exc, "reason", exc))
        if any(w in reason for w in ("Tunnel", "403", "407", "roxy")):
            raise SegmentUnavailable("A proxy or firewall blocks huggingface.co on this network") from None
        raise SegmentUnavailable(f"Could not download the model: {reason}") from None
    tmp = dest.with_name(dest.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(buf) as z:
            for info in z.infolist():
                target = (tmp / info.filename).resolve()
                if not str(target).startswith(str(tmp.resolve())):     # zip slip
                    raise SegmentUnavailable("The model archive has an unsafe path")
                if info.is_dir() or not info.filename.lower().endswith((".onnx", ".json", ".txt", ".yaml", ".data")):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
    except zipfile.BadZipFile:
        raise SegmentUnavailable("The download is not a model archive") from None
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    tmp.rename(dest)
    if not model_files(name):
        raise SegmentUnavailable("The archive has no encoder/decoder ONNX files")
    if progress:
        progress(100.0)
    return {"model": name, "path": str(dest), "already": False}


class SamOnnx:
    """The two graphs, with inputs and outputs matched by name."""

    def __init__(self, encoder: Path, decoder: Path, providers=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = providers or ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(encoder), so, providers=providers)
        self.dec = ort.InferenceSession(str(decoder), so, providers=providers)
        self.enc_in = self.enc.get_inputs()[0].name
        self.enc_out = [o.name for o in self.enc.get_outputs()]
        self.dec_in = [i.name for i in self.dec.get_inputs()]

    def encode(self, rgb: np.ndarray) -> dict:
        import cv2

        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.enc.run(None, {self.enc_in: x})
        feats = dict(zip(self.enc_out, outs))
        return {"feats": feats, "size": (h, w)}

    def _feat(self, emb: dict, want: str):
        feats = emb["feats"]
        if want in feats:
            return feats[want]
        for k, v in feats.items():
            if want.replace("image_embed", "embed") in k:
                return v
        return list(feats.values())[{"high_res_feats_0": 0, "high_res_feats_1": 1}.get(want, -1)]

    def decode(self, emb: dict, points: np.ndarray, labels: np.ndarray, prefer_large: bool = False) -> tuple[np.ndarray, float]:
        """points (N, 2) in window pixels, labels (N,) 1 add / 0 remove / 2-3 box corners -> (mask bool HxW, score)."""
        import cv2

        h, w = emb["size"]
        pc = points.astype(np.float32).copy()
        pc[:, 0] *= SIZE / w
        pc[:, 1] *= SIZE / h
        feed_all = {
            "image_embed": self._feat(emb, "image_embed"),
            "high_res_feats_0": self._feat(emb, "high_res_feats_0"),
            "high_res_feats_1": self._feat(emb, "high_res_feats_1"),
            "point_coords": pc[None],
            "point_labels": labels.astype(np.float32)[None],
            "mask_input": np.zeros((1, 1, SIZE // 4, SIZE // 4), np.float32),
            "has_mask_input": np.zeros((1,), np.float32),
            "orig_im_size": np.array([SIZE, SIZE], np.int32),
        }
        feed = {}
        for name in self.dec_in:
            key = name if name in feed_all else next((k for k in feed_all if k in name), None)
            if key is None:
                raise SegmentUnavailable(f"This decoder needs an input Syntra does not know: {name}")
            feed[name] = feed_all[key]
        masks, scores = self.dec.run(None, feed)[:2]
        masks = np.asarray(masks)[0]
        scores = np.asarray(scores).reshape(-1)
        best = int(np.argmax(scores)) if scores.size else 0
        if prefer_large and masks.ndim == 3 and scores.size > 1:
            # one click: SAM's three answers are a part, the object and a wider area; the
            # best-scored is often only a part (one roof face). Take the largest of those
            # scoring close to the best: the whole object, not the background.
            areas = [(masks[i] > 0).sum() for i in range(masks.shape[0])]
            near = [i for i in range(masks.shape[0]) if scores[i] >= scores[best] - LARGE_MARGIN]
            best = max(near, key=lambda i: areas[i])
        logit = masks[best] if masks.ndim == 3 else masks
        logit = cv2.resize(logit.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return logit > 0, float(scores[best]) if scores.size else 0.0


_engine: SamOnnx | None = None
_engine_name: str | None = None
_pinned = False
_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_plans: dict = {}
_lock = threading.Lock()
_last_timings: dict = {}
CACHE_VIEWS = 4


def _get_engine(device: str = "cpu") -> SamOnnx:
    global _engine, _engine_name
    if _pinned:
        return _engine
    st = status()
    if not st["ready"]:
        raise SegmentUnavailable(st["reason"])
    if _engine is None or _engine_name != st["model"]:
        from app.core.onnx_runtime import session_providers

        enc, dec = model_files(st["model"])
        _engine = SamOnnx(enc, dec, session_providers(device))
        _engine_name = st["model"]
        _cache.clear()
    return _engine


def set_engine(engine, name: str = "test") -> None:
    """Tests: use a stand-in engine (None goes back to the installed model)."""
    global _engine, _engine_name, _pinned
    _engine, _engine_name, _pinned = engine, name, engine is not None
    _cache.clear()


def _plan(path: str):
    from app.engine.raster_prep import plan_rgb

    key = (path, Path(path).stat().st_mtime)
    if key not in _plans:
        _plans.clear()
        _plans[key] = plan_rgb(path)
    return _plans[key]


def view_window(view_px: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) in raster pixels -> integer window clipped to the image."""
    x0, y0, x1, y1 = view_px
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    c1, r1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    if c1 - c0 < 8 or r1 - r0 < 8:
        raise ValueError("The view does not cover the image")
    return c0, r0, c1 - c0, r1 - r0


def _embedded(path: str, window: tuple[int, int, int, int], device: str, engine=None):
    """The view encoded once (cached per view): (engine, embedding, scale, encode_ms). Holds _lock."""
    col, row, w, h = window
    scale = max(1.0, max(w, h) / SIZE)
    out_h, out_w = max(1, round(h / scale)), max(1, round(w / scale))
    eng = engine or _get_engine(device)
    key = (path, window, (out_h, out_w), _engine_name)
    t0 = time.perf_counter()
    emb = _cache.get(key)
    encode_ms = 0.0
    if emb is None:
        import rasterio
        from rasterio.windows import Window

        from app.engine.raster_prep import read_rgb

        with rasterio.open(path) as ds:
            rgb, _valid = read_rgb(ds, _plan(path), window=Window(col, row, w, h), out_shape=(out_h, out_w))
        emb = eng.encode(rgb)
        _cache[key] = emb
        while len(_cache) > CACHE_VIEWS:
            _cache.popitem(last=False)
        encode_ms = (time.perf_counter() - t0) * 1000
    else:
        _cache.move_to_end(key)
    return eng, emb, scale, encode_ms


def prepare(path: str, window: tuple[int, int, int, int], device: str = "cpu", engine=None) -> dict:
    """Encode the view ahead of the first click (the slow part, seconds on a CPU)."""
    with _lock:
        _eng, _emb, _scale, encode_ms = _embedded(path, window, device, engine)
    return {"encode_ms": round(encode_ms, 1), "cached": encode_ms == 0.0}


def segment_everything(path: str, window: tuple[int, int, int, int], opts, device: str = "cpu",
                       engine=None, grid: int = 16, min_score: float = 0.7) -> dict:
    """Every object in the view, SAM's automatic way: one add-point at each node of a
    grid x grid lattice, masks below `min_score` dropped, the rest de-duplicated
    (higher score first; a mask overlapping a kept one by more than half, or lying
    mostly inside it, is dropped), then fitted to the class's shape. Masks covering
    more than a third of the view are background (roads, fields), not objects."""
    from shapely.geometry import box as sbox

    from app.engine.shapes import mask_to_polygon, refine

    col, row, w, h = window
    with _lock:
        eng, emb, scale, encode_ms = _embedded(path, window, device, engine)
        eh, ew = emb["size"]
        t1 = time.perf_counter()
        found = []
        for gy in range(grid):
            for gx in range(grid):
                pt = np.array([[(gx + 0.5) * ew / grid, (gy + 0.5) * eh / grid]], np.float32)
                mask, score = eng.decode(emb, pt, np.array([1], np.float32))
                if score < min_score or not mask.any():
                    continue
                poly = mask_to_polygon(mask, seed_xy=tuple(pt[0]), fill_holes_px2=opts.px2(opts.fill_holes_m2),
                                       offset=(col, row), scale=scale)
                if poly is not None:
                    found.append((score, poly))
        decode_ms = (time.perf_counter() - t1) * 1000
    view_area = sbox(col, row, col + w, row + h).area
    kept = []
    for score, poly in sorted(found, key=lambda t: -t[0]):
        if poly.area > view_area / 3 or poly.area < 16:
            continue
        if any(poly.intersection(k).area > 0.5 * min(poly.area, k.area) for _, k in kept):
            continue
        kept.append((score, poly))
    objects = []
    for score, poly in kept:
        try:
            res = refine(poly, opts)
        except ValueError:
            continue
        res.update(score=round(float(score), 3))
        objects.append(res)
    timings = {"encode_ms": round(encode_ms, 1), "decode_ms": round(decode_ms, 1), "prompts": grid * grid}
    _last_timings.update(timings)
    return {"objects": objects, "timings": timings}


def segment(path: str, window: tuple[int, int, int, int], clicks: list[tuple[float, float, int]],
            opts, device: str = "cpu", engine=None, prefer_large: bool = True) -> dict:
    """Clicks in raster pixels -> refined outline in raster pixels + stats."""
    from app.engine.shapes import mask_to_polygon, refine

    labels = [lbl for _, _, lbl in clicks]
    boxed = labels.count(2) == 1 and labels.count(3) == 1       # a box prompt: top-left 2, bottom-right 3 (SAM)
    if not (any(l == 1 for l in labels) or boxed):
        raise ValueError("Click inside the object, or draw a box around it")
    col, row, w, h = window
    with _lock:
        eng, emb, scale, encode_ms = _embedded(path, window, device, engine)
        pts = np.array([[(x - col) / scale, (y - row) / scale] for x, y, _ in clicks], dtype=np.float32)
        lbl = np.array([l if l in (1, 2, 3) else 0 for _, _, l in clicks], dtype=np.float32)
        t1 = time.perf_counter()
        single = prefer_large and int((lbl == 1).sum()) == 1 and not boxed and not (lbl == 0).any()
        try:
            mask, score = eng.decode(emb, pts, lbl, prefer_large=single)
        except TypeError:                                         # a stand-in engine without the option
            mask, score = eng.decode(emb, pts, lbl)
        decode_ms = (time.perf_counter() - t1) * 1000
    seed = next((p for p, l in zip(pts, lbl) if l == 1), None)
    if seed is None and boxed:                                   # a box alone: keep the part at its centre
        corners = [p for p, l in zip(pts, lbl) if l in (2, 3)]
        seed = (corners[0] + corners[1]) / 2
    poly = mask_to_polygon(mask, seed_xy=tuple(seed) if seed is not None else None,
                           fill_holes_px2=opts.px2(opts.fill_holes_m2), offset=(col, row), scale=scale)
    if poly is None:
        raise ValueError("Nothing found under the click. Zoom in or click nearer the middle")
    res = refine(poly, opts)
    timings = {"encode_ms": round(encode_ms, 1), "decode_ms": round(decode_ms, 1),
               "cached": encode_ms == 0.0, "scale": round(scale, 3)}
    _last_timings.update(timings)
    res.update(score=round(score, 3), timings=timings)
    return res
