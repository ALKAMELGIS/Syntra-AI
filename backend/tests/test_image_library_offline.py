"""Offline tests for app.core.image_library — no rasterio/GDAL required."""
from __future__ import annotations

from pathlib import Path

from app.core.image_library import (
    build_entry,
    compute_area_km2,
    find_entry_by_image_id,
    list_library,
    read_entry,
    sidecar_path,
    write_entry,
)


def test_compute_area_km2_real_geometry():
    # 1000 x 1000 px at 0.5 m GSD -> 500m x 500m = 0.25 km^2
    assert compute_area_km2(1000, 1000, 0.5, 0.5) == 0.25


def test_compute_area_km2_missing_inputs_returns_none():
    assert compute_area_km2(None, 1000, 0.5, 0.5) is None
    assert compute_area_km2(1000, 1000, None, 0.5) is None


def test_build_entry_with_raster_meta():
    entry = build_entry(
        image_id="abc123",
        filename="scene.tif",
        path=Path("/data/uploads/abc123_scene.tif"),
        size_bytes=4096,
        uploaded_at="2026-01-01T00:00:00+00:00",
        raster_meta={"width": 2000, "height": 2000, "bands": 3, "dtype": "uint8", "crs": "EPSG:4326", "gsd_x_m": 0.3, "gsd_y_m": 0.3},
    )
    assert entry.image_id == "abc123"
    assert entry.width == 2000
    assert entry.area_km2 == compute_area_km2(2000, 2000, 0.3, 0.3)
    assert entry.raster_meta_note is None


def test_build_entry_without_raster_meta_notes_why():
    entry = build_entry(
        image_id="xyz",
        filename="scene.tif",
        path=Path("/data/uploads/xyz_scene.tif"),
        size_bytes=100,
        uploaded_at="2026-01-01T00:00:00+00:00",
        raster_meta=None,
        raster_meta_note="rasterio not installed",
    )
    assert entry.width is None
    assert entry.area_km2 is None
    assert entry.raster_meta_note == "rasterio not installed"


def test_write_and_read_entry_roundtrip(tmp_path):
    img_path = tmp_path / "id1_scene.tif"
    img_path.write_bytes(b"not-a-real-tiff")
    entry = build_entry(
        image_id="id1",
        filename="scene.tif",
        path=img_path,
        size_bytes=img_path.stat().st_size,
        uploaded_at="2026-01-01T00:00:00+00:00",
        raster_meta={"width": 100, "height": 100, "bands": 1, "dtype": "uint8", "crs": None, "gsd_x_m": 1.0, "gsd_y_m": 1.0},
    )
    sc = write_entry(img_path, entry)
    assert sc == sidecar_path(img_path)
    assert sc.exists()

    loaded = read_entry(img_path)
    assert loaded["image_id"] == "id1"
    assert loaded["area_km2"] == 0.01


def test_list_library_sorts_newest_first(tmp_path):
    for image_id, filename, ts in [
        ("a", "first.tif", "2026-01-01T00:00:00+00:00"),
        ("b", "second.tif", "2026-02-01T00:00:00+00:00"),
    ]:
        p = tmp_path / f"{image_id}_{filename}"
        p.write_bytes(b"x")
        entry = build_entry(image_id, filename, p, p.stat().st_size, ts)
        write_entry(p, entry)

    out = list_library(tmp_path)
    assert [e["filename"] for e in out] == ["second.tif", "first.tif"]


def test_list_library_handles_missing_sidecar_gracefully(tmp_path):
    orphan = tmp_path / "orphan.tif"
    orphan.write_bytes(b"x")
    out = list_library(tmp_path)
    assert len(out) == 1
    assert out[0]["image_id"] is None
    assert "No metadata sidecar" in out[0]["raster_meta_note"]


def test_list_library_empty_dir_returns_empty(tmp_path):
    assert list_library(tmp_path / "does_not_exist") == []


def test_find_entry_by_image_id(tmp_path):
    p = tmp_path / "id42_scene.tif"
    p.write_bytes(b"x")
    entry = build_entry("id42", "scene.tif", p, p.stat().st_size, "2026-01-01T00:00:00+00:00")
    write_entry(p, entry)

    found = find_entry_by_image_id(tmp_path, "id42")
    assert found is not None
    assert found["filename"] == "scene.tif"

    assert find_entry_by_image_id(tmp_path, "missing") is None
