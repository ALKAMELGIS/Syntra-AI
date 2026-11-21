"""A PROJ_LIB left by another program (PostGIS, QGIS) must not be used."""
import os


def test_foreign_proj_lib_is_replaced_or_removed(monkeypatch, tmp_path):
    from app.core.proj_env import ensure_own_geodata

    foreign = tmp_path / "PostgreSQL" / "share" / "proj"
    foreign.mkdir(parents=True)
    (foreign / "proj.db").write_bytes(b"old")
    monkeypatch.setenv("PROJ_LIB", str(foreign))
    monkeypatch.setenv("PROJ_DATA", str(foreign))
    monkeypatch.setenv("GDAL_DATA", str(tmp_path / "PostgreSQL" / "gdal"))
    ensure_own_geodata()
    for k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        assert os.environ.get(k) != str(foreign) and "PostgreSQL" not in (os.environ.get(k) or "")


def test_value_inside_this_environment_is_kept(monkeypatch):
    import sys

    from app.core.proj_env import ensure_own_geodata

    own = os.path.join(sys.prefix, "share", "proj")
    monkeypatch.setenv("PROJ_LIB", own)
    ensure_own_geodata()
    assert os.environ["PROJ_LIB"] == own


def test_rasterio_reads_crs_after_a_foreign_proj_lib(tmp_path):
    import subprocess
    import sys

    import pytest
    pytest.importorskip("rasterio")
    bad = tmp_path / "proj"
    bad.mkdir()
    (bad / "proj.db").write_bytes(b"not a database")
    env = dict(os.environ, PROJ_LIB=str(bad), PROJ_DATA=str(bad))
    code = ("import app, rasterio.crs; print(rasterio.crs.CRS.from_epsg(32640).to_epsg())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.stdout.strip() == "32640", out.stderr[-400:]
