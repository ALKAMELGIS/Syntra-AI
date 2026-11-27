"""A hosted install serves the built interface itself (SYNTRA_STATIC_DIR): the page is
never cached, so a new deploy shows at once, and /api/* never falls back to it."""
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_index_is_not_cached_and_api_is_not_the_page(tmp_path):
    (tmp_path / "static" / "assets").mkdir(parents=True)
    (tmp_path / "static" / "index.html").write_text("<html>syntra</html>")
    (tmp_path / "static" / "assets" / "a.js").write_text("x")
    code = """
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)
for p in ("/", "/index.html", "/some/route"):
    r = c.get(p)
    assert r.status_code == 200 and "syntra" in r.text and r.headers["cache-control"] == "no-cache", p
assert c.get("/assets/a.js").text == "x"
assert c.get("/api/nothing-here").status_code == 404
print("ok")
"""
    env = {"SYNTRA_STATIC_DIR": str(tmp_path / "static"), "SYNTRA_DATA_DIR": str(tmp_path / "data"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip().endswith("ok"), r.stderr[-2000:]
