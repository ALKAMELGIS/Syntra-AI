"""
Offline tests for the expanded model catalog (Phase 5 — IPSAR SAT-style
category list) and the factory's honesty guard.

Locks down two things that would otherwise be easy to get wrong when adding
a new catalog entry:
  1. Every catalog entry is internally consistent (unique id, valid status).
  2. build_model() never silently returns a runnable object for a model
     that has no real weights — it must raise NotProductionReadyError.

Run directly: python tests/test_registry_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.base import ModelStatus  # noqa: E402
from app.models.factory import build_model  # noqa: E402
from app.models.registry import CATALOG, get_metadata, list_catalog  # noqa: E402
from app.models.segmentation import NotProductionReadyError  # noqa: E402


def test_catalog_has_20_unique_entries():
    cat = list_catalog()
    assert len(cat) == 20
    ids = [m.model_id for m in cat]
    assert len(ids) == len(set(ids)), "duplicate model_id in catalog"


def test_every_catalog_entry_key_matches_its_model_id():
    for key, meta in CATALOG.items():
        assert key == meta.model_id


def test_get_metadata_rejects_unknown_id():
    try:
        get_metadata("not_a_real_model")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_not_production_ready_models_refuse_to_build():
    # field_boundary_segmentation is the one documented exception: its
    # postprocessing math (app/models/segmentation.py) is real and scaffolded,
    # so the factory legitimately builds a FieldBoundarySegmenter instance —
    # the honesty refusal for it happens inside predict() itself (no trained
    # weights), not at build time. Every other NOT_PRODUCTION_READY model has
    # no runtime at all and must refuse right here.
    not_ready = [
        m.model_id for m in list_catalog()
        if m.status == ModelStatus.NOT_PRODUCTION_READY and m.model_id != "field_boundary_segmentation"
    ]
    assert len(not_ready) >= 12
    for model_id in not_ready:
        try:
            build_model(model_id)
            assert False, f"{model_id}: expected NotProductionReadyError, got a built model instead"
        except NotProductionReadyError:
            pass


def test_dota_backed_models_build_as_aerial_obb_detectors():
    # These five are VALIDATION: real aerial-pretrained weights (DOTA v1),
    # not yet benchmarked on the user's imagery. They must build, unlike the
    # NOT_PRODUCTION_READY models above.
    from app.models.aerial_obb_detector import DOTA_CLASS_MAP, AerialObbDetector

    assert set(DOTA_CLASS_MAP) == {
        "car_detection", "ship_detection", "aircraft_detection", "oil_tank_detection", "bridge_detection",
    }
    for model_id in DOTA_CLASS_MAP:
        meta = get_metadata(model_id)
        assert meta.status == ModelStatus.VALIDATION, model_id
        assert "DOTA" in meta.notes, model_id
        # every DOTA class a model asks for maps to one of its own class ids
        assert set(DOTA_CLASS_MAP[model_id].values()) == set(meta.class_names), model_id
        assert isinstance(build_model(model_id), AerialObbDetector), model_id


def test_change_detection_and_measurement_types_are_present_and_honest():
    change = get_metadata("change_detection")
    assert change.model_type == "change_detection"
    assert change.status == ModelStatus.NOT_PRODUCTION_READY

    oil_shadow = get_metadata("oil_tank_shadow_analysis")
    tank_volume = get_metadata("tank_volume_estimation")
    assert oil_shadow.model_type == "measurement"
    assert tank_volume.model_type == "measurement"
    assert oil_shadow.status == ModelStatus.NOT_PRODUCTION_READY
    assert tank_volume.status == ModelStatus.NOT_PRODUCTION_READY


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  OK  {t.__name__}")
    print(f"\n{passed}/{len(tests)} registry tests passed.")


if __name__ == "__main__":
    _run_all()
