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
    """Training history (if any) plus the accuracy checks run on the user's own images."""
    from app.core.config import settings
    from app.engine.accuracy import checks

    perf = _training_performance(model_id)
    done = checks(settings.DATA_DIR, model_id)
    if perf is None and not done:
        return None
    if perf is None:
        from app.models.registry import get_metadata
        perf = _summary(model_id, get_metadata(model_id).name, [], source=None, per_class=None, hidden_classes=[])
    perf["checks"] = done
    return perf


def _training_performance(model_id: str) -> dict | None:
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
            f1 = perf["checks"][0]["overall"]["f1"] if perf.get("checks") else None
            out.append({"model_id": m.model_id, "name": m.name, "best": perf["best"], "f1": f1})
    return out


# --- training runs, live while they train ---------------------------------
LIVE = {"queued", "processing", "starting", "training", "running"}


def _epochs_from_history(history: list[dict]) -> list[dict]:
    by_epoch: dict[int, dict] = {}
    for ev in history or []:
        if ev.get("type") != "epoch" or ev.get("epoch") is None:
            continue
        tl = ev.get("train_loss")
        if tl is None:                                  # events written before train_loss existed
            parts = [ev.get(k) for k in ("box_loss", "cls_loss", "dfl_loss") if ev.get(k) is not None]
            tl = round(sum(parts), 4) if parts else None
        by_epoch[int(ev["epoch"])] = {"epoch": int(ev["epoch"]), "train_loss": tl, "val_loss": ev.get("val_loss"),
                                      "precision": ev.get("precision"), "recall": ev.get("recall"),
                                      "map50": ev.get("map50"), "map50_95": ev.get("map50_95")}
    return [by_epoch[k] for k in sorted(by_epoch)]


def _run_name(job: dict) -> str:
    group, version = job.get("model_group"), job.get("model_version")
    if group:
        return f"{group} v{int(version):03d}" if isinstance(version, int) else f"{group} {version or ''}".strip()
    return f"Run {job['job_id'][:8]}"


def run_performance(job: dict) -> dict:
    """A training job's history: results.csv once it is written, the epoch events until then."""
    epochs = []
    csv_path = (job.get("result") or {}).get("results_csv")
    if csv_path and Path(csv_path).is_file():
        try:
            epochs = parse_results_csv(csv_path)
        except (OSError, ValueError, KeyError):
            epochs = []
    if not epochs:
        epochs = _epochs_from_history(job.get("history"))
    cfg = job.get("config") or {}
    planned = (job.get("progress") or {}).get("epochs") or cfg.get("epochs")
    test = (job.get("result") or {}).get("test")
    out = _summary(f"run:{job['job_id']}", _run_name(job), epochs, source=None, per_class=None, hidden_classes=[])
    out.update({
        "run_id": job["job_id"], "status": job.get("status"), "live": job.get("status") in LIVE,
        "planned_epochs": int(planned) if planned else None, "where": job.get("where") or "this computer",
        "device": cfg.get("device"), "elapsed_s": (job.get("progress") or {}).get("elapsed_s"),
        "error": job.get("error"), "message": job.get("message"),
        "test": test if isinstance(test, dict) and "error" not in test else None,
        "updated_at": job.get("updated_at"), "checks": [],
        "training": {"note": None},
    })
    return out


def training_runs(limit: int = 10) -> list[dict]:
    from app.core.jobs import get_store

    out = []
    for job in get_store().list("train", limit=limit):
        perf = run_performance(job)
        if not perf["epochs"] and not perf["live"]:
            continue                                    # never reached an epoch: nothing to show
        out.append({"model_id": perf["model_id"], "run_id": job["job_id"], "name": perf["name"], "best": perf["best"],
                    "f1": None, "status": perf["status"], "live": perf["live"],
                    "epoch": perf["epochs"][-1]["epoch"] if perf["epochs"] else 0, "planned_epochs": perf["planned_epochs"]})
    return out
