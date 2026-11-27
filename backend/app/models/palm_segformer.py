"""
Date palm crowns from VHR satellite RGB: the released SegFormer-B3 RGB model
of "Date Palm Segmentation from WorldView-3 Satellite Data"
(Al-Ruzouq et al., 2024; github.com/brakuta/Date-Palm-Tree-Segmentation-from-
WorldView3-Satellite-Data), weights CC BY 4.0 on Hugging Face
(brakuta/date-palm-wv3-models, segformer_b3_rgb.safetensors).

Trained on 20 WorldView-3 scenes over Ajman, Sharjah and Fujairah,
pansharpened to about 0.31 m; the paper reports 76.16 % mIoU and 84.59 %
mean F-score for this RGB model on its held-out tiles.

The network is written here in plain PyTorch with the module names
MMSegmentation 1.x uses (MixVisionTransformer + SegformerHead), so the
released file loads with `strict=True` and no MMSegmentation install: a
missing or extra tensor is an error, never a silent partial load.

Input contract (from the publisher's PREPROCESSING.md, RGB models): 8-bit
RGB, ImageNet mean/std, the network sees R, G, B. The released head has
150 outputs (a stale config value); training used 0 = background and
1 = palm, so palm probability is the softmax of channel 1 over all outputs.

Each tile is read at the training scale (~0.31 m), segmented, and the palm
mask is split into one crown per tree with a seeded watershed on the
distance to the crown edge, written here (not copied from the toolkit).
"""
from __future__ import annotations

import threading

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, RawDetection

TRAINING_GSD_M = 0.31
MODEL_PX = 512
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
PALM = 1
MIN_CROWN_M = 2.0           # crowns narrower than this are dropped
SEED_SPACING_M = 3.0        # two trees' centres are at least this far apart

# MiT-B3
EMBED = (64, 128, 320, 512)
DEPTHS = (3, 4, 18, 3)
HEADS = (1, 2, 5, 8)
SR = (8, 4, 2, 1)
PATCH = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))   # kernel, stride, padding

_CACHE: dict = {}
_LOCK = threading.Lock()


def build_network(num_classes: int = 150):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class PatchEmbed(nn.Module):
        def __init__(self, cin, cout, k, s, p):
            super().__init__()
            self.projection = nn.Conv2d(cin, cout, k, s, p)
            self.norm = nn.LayerNorm(cout, eps=1e-6)

        def forward(self, x):
            x = self.projection(x)
            h, w = x.shape[2:]
            return self.norm(x.flatten(2).transpose(1, 2)), (h, w)

    class EfficientAttention(nn.Module):
        def __init__(self, dim, heads, sr):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.sr_ratio = sr
            if sr > 1:
                self.sr = nn.Conv2d(dim, dim, sr, sr)
                self.norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x, hw, identity):
            kv = x
            if self.sr_ratio > 1:
                b, n, c = x.shape
                kv = x.transpose(1, 2).reshape(b, c, *hw)
                kv = self.norm(self.sr(kv).flatten(2).transpose(1, 2))
            return identity + self.attn(x, kv, kv, need_weights=False)[0]

    class MixFFN(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(dim, hidden, 1), nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
                nn.GELU(), nn.Dropout(0.0), nn.Conv2d(hidden, dim, 1), nn.Dropout(0.0))

        def forward(self, x, hw, identity):
            b, n, c = x.shape
            out = self.layers(x.transpose(1, 2).reshape(b, c, *hw))
            return identity + out.flatten(2).transpose(1, 2)

    class Block(nn.Module):
        def __init__(self, dim, heads, sr):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = EfficientAttention(dim, heads, sr)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.ffn = MixFFN(dim, dim * 4)

        def forward(self, x, hw):
            x = self.attn(self.norm1(x), hw, identity=x)
            return self.ffn(self.norm2(x), hw, identity=x)

    class MiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            cin = 3
            for i in range(4):
                self.layers.append(nn.ModuleList([
                    PatchEmbed(cin, EMBED[i], *PATCH[i]),
                    nn.ModuleList([Block(EMBED[i], HEADS[i], SR[i]) for _ in range(DEPTHS[i])]),
                    nn.LayerNorm(EMBED[i], eps=1e-6),
                ]))
                cin = EMBED[i]

        def forward(self, x):
            outs = []
            for embed, blocks, norm in self.layers:
                x, hw = embed(x)
                for blk in blocks:
                    x = blk(x, hw)
                x = norm(x)
                x = x.transpose(1, 2).reshape(x.shape[0], -1, *hw)
                outs.append(x)
            return outs

    class ConvModule(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([ConvModule(c, 256) for c in EMBED])
            self.fusion_conv = ConvModule(256 * 4, 256)
            self.conv_seg = nn.Conv2d(256, num_classes, 1)

        def forward(self, feats):
            size = feats[0].shape[2:]
            outs = [F.interpolate(conv(f), size=size, mode="bilinear", align_corners=False)
                    for conv, f in zip(self.convs, feats)]
            return self.conv_seg(self.fusion_conv(torch.cat(outs, dim=1)))

    class SegFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = MiT()
            self.decode_head = Head()

        def forward(self, x):
            logits = self.decode_head(self.backbone(x))
            return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    return SegFormer()


def load_network(path: str, device: str = "cpu"):
    """The released file, loaded strictly. Raises on any mismatch."""
    import torch
    from safetensors.torch import load_file

    sd = load_file(path)
    head = sd.get("decode_head.conv_seg.weight")
    if head is None or sd.get("backbone.layers.0.0.projection.weight") is None:
        raise ValueError("Not a SegFormer palm model (no MiT backbone or SegFormer head).")
    if tuple(sd["backbone.layers.0.0.projection.weight"].shape[:2]) != (EMBED[0], 3):
        raise ValueError("Expected the RGB (3-band) SegFormer-B3 model.")
    net = build_network(num_classes=int(head.shape[0]))
    net.load_state_dict(sd, strict=True)
    net.eval()
    dev = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    return net.to(dev), dev


def verify_weights(path) -> dict:
    net, _ = load_network(str(path))
    n = sum(p.numel() for p in net.parameters())
    return {"model_type": "Segmentor", "architecture": "SegFormer-B3 (MiT-B3)", "parameters": n,
            "tile_px": MODEL_PX, "resolution_cm": TRAINING_GSD_M * 100, "class_names": ["background", "date_palm"]}


def _cached(path: str, device: str):
    import os

    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime, device)
    with _LOCK:
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = load_network(path, device)
        return _CACHE[key]


def palm_probability(net_dev, rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 at ~0.31 m, H and W <= MODEL_PX -> palm probability (H, W)."""
    import torch

    net, dev = net_dev
    h, w = rgb.shape[:2]
    x = np.zeros((MODEL_PX, MODEL_PX, 3), np.float32)               # zero padding, as in training
    x[:h, :w] = (rgb.astype(np.float32) - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(dev)
    with torch.inference_mode():
        prob = torch.softmax(net(t), dim=1)[0, PALM]
    return prob[:h, :w].float().cpu().numpy()


def split_crowns(prob: np.ndarray, gsd_m: float, threshold: float = 0.5) -> list[tuple[np.ndarray, float]]:
    """Palm probability -> one (contour Nx2 in prob pixels, mean probability) per tree."""
    import cv2
    from scipy import ndimage as ndi

    mask = (prob >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    r = max(2, int(round(SEED_SPACING_M / 2 / gsd_m)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    smooth = cv2.GaussianBlur(dist, (0, 0), 1.0)
    peaks = (smooth >= cv2.dilate(smooth, k)) & (dist >= MIN_CROWN_M / 2 / gsd_m * 0.6) & (mask > 0)
    markers, n = ndi.label(peaks)
    comps, nc = ndi.label(mask)
    has_seed = set(np.unique(comps[markers > 0]).tolist())
    for c in range(1, nc + 1):                                          # a blob with no peak still gets one seed
        if c not in has_seed:
            yy, xx = np.unravel_index(np.argmax(np.where(comps == c, dist, -1)), dist.shape)
            n += 1
            markers[yy, xx] = n
    if n == 0:
        return []
    # each crown pixel goes to its nearest seed within the same blob (Voronoi split of touching crowns)
    _, (iy, ix) = ndi.distance_transform_edt(markers == 0, return_indices=True)
    labels = markers[iy, ix]
    labels[mask == 0] = 0
    labels[comps != comps[iy, ix]] = 0
    min_area = np.pi * (MIN_CROWN_M / 2 / gsd_m) ** 2
    out = []
    for i in range(1, n + 1):
        inst = (labels == i).astype(np.uint8)
        area = int(inst.sum())
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        out.append((c.astype(np.float64), float(prob[inst > 0].mean())))
    return out


class PalmSegformerDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, weights_path: str, training_gsd_m: float = TRAINING_GSD_M):
        self.metadata = metadata
        self.weights_path = weights_path
        self.training_gsd_m = training_gsd_m
        self._net = None
        self.nominal_tile = None

    def load_model(self, device: str) -> None:
        self._net = _cached(self.weights_path, device)

    def validate_model(self) -> tuple[bool, str]:
        return (True, "OK") if self._net is not None else (False, "Model not loaded.")

    def preferred_tile_size(self, gsd_m: float | None) -> int:
        if not gsd_m or gsd_m <= 0:
            return MODEL_PX
        return int(np.clip(round(MODEL_PX * self.training_gsd_m / gsd_m), 64, 4096))

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        import cv2

        if self._net is None:
            raise RuntimeError("Model not loaded.")
        conf = confidence_threshold if confidence_threshold is not None else self.metadata.default_confidence_threshold
        h, w = tile_rgb.shape[:2]
        f = MODEL_PX / float(self.nominal_tile or max(h, w))              # native px -> model px
        nh, nw = min(MODEL_PX, max(1, round(h * f))), min(MODEL_PX, max(1, round(w * f)))
        rgb = cv2.resize(np.ascontiguousarray(tile_rgb[..., :3]), (nw, nh), interpolation=cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA)
        prob = palm_probability(self._net, rgb)
        sx, sy = w / nw, h / nh
        out = []
        for contour, score in split_crowns(prob, self.training_gsd_m):
            if score < conf:
                continue
            poly = [(float(x * sx), float(y * sy)) for x, y in contour]
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(RawDetection(box=(min(xs), min(ys), max(xs), max(ys)), score=score, class_id=0, polygon=poly))
        return out

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {"model_version": self.metadata.version, "model_status": self.metadata.status.value,
                "weights": self.weights_path, "notes": self.metadata.notes,
                "weights_licence": "CC BY 4.0, Al-Ruzouq et al. 2024 (brakuta/date-palm-wv3-models)", **(extra_meta or {})}
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)
