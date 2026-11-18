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
