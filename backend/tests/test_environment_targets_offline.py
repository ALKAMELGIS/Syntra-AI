"""
Offline tests for the per-library (targeted) install resolution used by the
Library Status panel's "pick a library, install just that one" flow — no pip
subprocess runs here, only the pure requirements.txt-parsing / module-name-
resolution helpers in app.diagnostics.environment (fastapi-free, so these run
without fastapi installed, same as the rest of this offline test suite).
"""
import pytest

from app.diagnostics.environment import parse_requirements, resolve_install_targets


def test_parse_requirements_strips_comments_and_extras():
    lines = parse_requirements()
    assert lines["fastapi"] == "fastapi==0.115.6"
    # bracketed extra ("uvicorn[standard]") must still key on the bare name
    assert lines["uvicorn"].startswith("uvicorn[standard]==")
    # inline trailing comment must be stripped
    assert lines["ultralytics"] == "ultralytics==8.3.49"


def test_resolve_install_targets_single_known_module():
    resolved = resolve_install_targets(["cv2"])
    assert resolved == ["opencv-python-headless==4.10.0.84"]


def test_resolve_install_targets_multiple_known_modules():
    resolved = resolve_install_targets(["ultralytics", "segmentation_models_pytorch"])
    assert resolved == [
        "ultralytics==8.3.49",
        "segmentation-models-pytorch==0.3.4",
    ]


def test_resolve_install_targets_unknown_module_raises():
    with pytest.raises(ValueError, match="totally_not_a_real_module"):
        resolve_install_targets(["totally_not_a_real_module"])


def test_resolve_install_targets_mixed_known_and_unknown_reports_only_unknown():
    with pytest.raises(ValueError) as exc_info:
        resolve_install_targets(["cv2", "bogus_pkg"])
    assert "bogus_pkg" in str(exc_info.value)
    assert "cv2" not in str(exc_info.value)
