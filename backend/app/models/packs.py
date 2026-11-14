"""
Model packs: open pretrained weights a category can install on demand.

A pack is one ONNX file from a fixed, reviewed URL. Until it is installed
the category shows as "Not installed" with an Install action; once the file
is in `WEIGHTS_DIR/packs/` and loads, the category becomes runnable
(VALIDATION: pretrained, not benchmarked on the user's imagery).

Every pack was chosen from the research report "Open models for satellite
detection" (Nov 2026). Licence facts are recorded per pack and shown in the
UI before installing; non-commercial packs are labelled as such.

Offline machines can install a pack by uploading the same file (the UI's
"From file" action), so a blocked network never makes a pack unusable.
Only the URLs in this table are ever downloaded: clients name a pack id,
never a URL.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

HF_GEODEEP = "https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/"


@dataclass(frozen=True)
class ModelPack:
    pack_id: str
    model_ids: tuple[str, ...]          # catalog entries this pack makes runnable
    title: str
    filename: str
    url: str
    runtime: str                        # "geodeep_detector" | "geodeep_segmentor"
    licence: str
    commercial_use: bool | None         # None = licence not stated by the publisher
    source: str                         # what it is and what it was trained on
    training_gsd_m: float | None
    approx_size_mb: float | None
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, installed: bool, path: Path | None) -> dict:
        d = asdict(self)
        d["model_ids"] = list(self.model_ids)
        d["caveats"] = list(self.caveats)
        d["installed"] = installed
        d["installed_size_mb"] = round(path.stat().st_size / 1e6, 1) if installed and path else None
        return d


PACKS: dict[str, ModelPack] = {
    "deepforest-trees": ModelPack(
        pack_id="deepforest-trees",
        model_ids=("tree_detection",),
        title="DeepForest tree crowns",
        filename="tree_crown_detection_retinanet_deepforest.onnx",
        url=HF_GEODEEP + "tree_crown_detection_retinanet_deepforest.onnx",
        runtime="geodeep_detector",
        licence="MIT",
        commercial_use=True,
        source="DeepForest (weecology) RetinaNet, ONNX export by GeoDeep. Trained on NEON aerial RGB at ~10 cm.",
        training_gsd_m=0.10,
        approx_size_mb=None,
        caveats=(
            "Trained at ~10 cm: on 30-50 cm satellite imagery tiles are upsampled, so small crowns are missed.",
            "Forest-trained; palms and sparse desert trees are not its training data.",
        ),
    ),
    "ramp-buildings": ModelPack(
        pack_id="ramp-buildings",
        model_ids=("building_detection",),
        title="RAMP building footprints",
        filename="buildings_ramp_XUnet_256.onnx",
        url=HF_GEODEEP + "buildings_ramp_XUnet_256.onnx",
        runtime="geodeep_segmentor",
        licence="CC BY-NC 4.0",
        commercial_use=False,
        source=(
            "RAMP baseline (DevGlobal / Radiant Earth) U-Net, ONNX export by GeoDeep. Trained on <=50 cm RGB "
            "from eight regions including Oman: the only open building model with arid Middle East training data."
        ),
        training_gsd_m=0.5,
        approx_size_mb=None,
        caveats=("Non-commercial licence: research and public-good use only.",),
    ),
}


def packs_dir() -> Path:
    from app.core.config import settings

    d = settings.WEIGHTS_DIR / "packs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pack_path(pack: ModelPack) -> Path:
    return packs_dir() / pack.filename


def is_installed(pack: ModelPack) -> bool:
    p = pack_path(pack)
    return p.is_file() and p.stat().st_size > 0


def pack_for_model(model_id: str) -> ModelPack | None:
    for pack in PACKS.values():
        if model_id in pack.model_ids:
            return pack
    return None


def list_packs() -> list[dict]:
    return [p.to_dict(is_installed(p), pack_path(p)) for p in PACKS.values()]


def verify_onnx(path: Path, runtime: str) -> dict:
    """Load the file and check it is the kind of model the pack says it is."""
    from app.models.geodeep_onnx import load_session

    _, cfg = load_session(str(path))
    want = "Detector" if runtime == "geodeep_detector" else "Segmentor"
    if cfg["model_type"] != want:
        raise ValueError(f"Expected a {want} model, the file is a {cfg['model_type']}.")
    return {"model_type": cfg["model_type"], "resolution_cm": cfg["resolution"], "tile_px": cfg["tiles_size"],
            "class_names": cfg["class_names"]}
