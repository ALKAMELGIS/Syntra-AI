"""
Training history and accuracy of a model, for the Model performance panel.

Sources, in order:
  * a model trained here (custom registry): the `results.csv` ultralytics
    writes next to its weights, one row per epoch;
  * a model pack: `app/models/pack_metrics/<pack_id>.json`, written when the
    pack was trained and checked, so the numbers ship with the code.

Detection models are not measured with "accuracy". What the panel shows:
  * loss per epoch, training against validation (a widening gap = the model
    memorises its training images);
  * precision, recall and mAP50 per epoch on the held-out validation images;
  * mAP50 per class from the final check, when it was run.
Mask metrics are used for segmentation models, box metrics otherwise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PACK_METRICS_DIR = Path(__file__).resolve().parent.parent / "models" / "pack_metrics"


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None   # NaN -> None


def parse_results_csv(path: str | Path) -> list[dict]:
    """ultralytics results.csv -> [{epoch, train_loss, val_loss, precision, recall, map50, map50_95}]."""
    with open(path, newline="") as fh:
        rows = [{k.strip(): v for k, v in r.items() if k} for r in csv.DictReader(fh)]
    if not rows:
        return []
    keys = rows[0].keys()
    kind = "(M)" if any(k.endswith("(M)") for k in keys) else "(B)"
    loss_parts = [k.split("/", 1)[1] for k in keys if k.startswith("train/") and k.endswith("_loss")]
    out = []
    for r in rows:
        tl = [_f(r.get(f"train/{p}")) for p in loss_parts]
        vl = [_f(r.get(f"val/{p}")) for p in loss_parts]
        out.append({
            "epoch": int(float(r["epoch"])),
            "train_loss": round(sum(x for x in tl if x is not None), 4) if any(x is not None for x in tl) else None,
            "val_loss": round(sum(x for x in vl if x is not None), 4) if any(x is not None for x in vl) else None,
            "precision": _f(r.get(f"metrics/precision{kind}")),
            "recall": _f(r.get(f"metrics/recall{kind}")),
            "map50": _f(r.get(f"metrics/mAP50{kind}")),
            "map50_95": _f(r.get(f"metrics/mAP50-95{kind}")),
        })
    return out


def _summary(model_id: str, name: str, epochs: list[dict], **extra) -> dict:
    scored = [e for e in epochs if e.get("map50") is not None]
    best = max(scored, key=lambda e: e["map50"]) if scored else None
    return {"model_id": model_id, "name": name, "epochs": epochs,
            "best": {"epoch": best["epoch"], "map50": round(best["map50"], 3)} if best else None, **extra}


def model_performance(model_id: str) -> dict | None:
    from app.core.config import settings
    from app.models.custom_registry import get_custom_model
    from app.models.packs import pack_for_model

    entry = get_custom_model(settings.DATA_DIR, model_id)
    if entry is not None:
        run_dir = Path(entry.weights_path).resolve().parent.parent
        csv_path = run_dir / "results.csv"
        if not csv_path.is_file():
            return None
        return _summary(model_id, entry.name, parse_results_csv(csv_path),
                        source="Trained in Syntra", per_class=None, hidden_classes=[],
                        task=entry.task, classes=list(entry.class_names.values()))

    pack = pack_for_model(model_id)
    if pack is not None:
        p = PACK_METRICS_DIR / f"{pack.pack_id}.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text())
        return _summary(model_id, data.get("name", pack.title), data["epochs"], source=data.get("source", pack.source),
                        per_class=data.get("per_class"), hidden_classes=data.get("hidden_classes", []),
                        task=data.get("task"), classes=data.get("classes"), licence=pack.licence,
                        training=data.get("training"))
    return None


def models_with_performance() -> list[dict]:
    from app.models.registry import list_catalog

    out = []
    for m in list_catalog():
        perf = model_performance(m.model_id)
        if perf is not None:
            out.append({"model_id": m.model_id, "name": m.name, "best": perf["best"]})
    return out
