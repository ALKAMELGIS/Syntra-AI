"""Rendered tiles come back from memory; a changed file is rendered again."""
import os
import time


def test_second_request_is_from_memory_and_a_changed_file_is_not(tmp_path, monkeypatch):
    from app.engine import tiles
    calls = []
    monkeypatch.setattr(tiles, "render_tile", lambda p, z, x, y, b=None, s=None: calls.append((z, x, y, b, s)) or b"png%d" % len(calls))
    monkeypatch.setattr(tiles, "_TILE_CACHE", None)
    f = tmp_path / "scene.tif"
    f.write_bytes(b"II*\0")
    assert tiles.cached_tile(str(f), 15, 1, 2) == b"png1"
    assert tiles.cached_tile(str(f), 15, 1, 2) == b"png1" and len(calls) == 1      # from memory
    assert tiles.cached_tile(str(f), 15, 1, 2, (3, 2, 1), None) == b"png2"           # another display: its own tile
    time.sleep(0.01)
    f.write_bytes(b"II*\0changed")
    os.utime(f, None)
    assert tiles.cached_tile(str(f), 15, 1, 2) == b"png3"                            # file changed: rendered again


def test_cache_stays_within_its_size(tmp_path, monkeypatch):
    from app.engine import tiles
    monkeypatch.setattr(tiles, "render_tile", lambda *a, **k: b"x" * 1000)
    monkeypatch.setattr(tiles, "_TILE_CACHE", None)
    monkeypatch.setattr(tiles, "_TILE_CACHE_BYTES", 0)
    monkeypatch.setattr(tiles, "_TILE_CACHE_MAX", 10_000)
    f = tmp_path / "scene.tif"
    f.write_bytes(b"II*\0")
    for i in range(50):
        tiles.cached_tile(str(f), 15, i, 0)
    assert tiles._TILE_CACHE_BYTES <= 10_000 and len(tiles._TILE_CACHE) == 10


def test_map_display_is_stretched_but_detection_is_not(tmp_path):
    """An 8-bit photo using only 40..235 is drawn with a percent clip on the map
    (like ArcGIS), while the detector still reads the stored values."""
    import numpy as np
    import pytest
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from app.engine.raster_prep import DISPLAY_STRETCH, plan_rgb
    from app.engine.tiles import parse_display
    rng = np.random.default_rng(0)
    data = rng.integers(40, 236, size=(3, 256, 256), dtype=np.uint8)
    p = tmp_path / "photo.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
                       crs="EPSG:32639", transform=from_origin(500000, 2900000, 0.2, 0.2)) as d:
        d.write(data)
    detector = plan_rgb(p)
    assert detector.lo == [0.0, 0.0, 0.0] and detector.hi == [255.0, 255.0, 255.0]      # untouched
    shown = plan_rgb(p, stretch=DISPLAY_STRETCH)
    assert all(39 <= v <= 45 for v in shown.lo) and all(230 <= v <= 236 for v in shown.hi)   # spread to full range
    stored = plan_rgb(p, stretch="none")
    assert stored.hi == [255.0, 255.0, 255.0]
    assert parse_display(None, "none") == (None, "none")
