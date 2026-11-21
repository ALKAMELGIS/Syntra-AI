"""
Deploy Syntra to its Hugging Face Space (Alkamelgis/SYNTRA-AI) from this computer.

Builds nothing itself: run it after `npm run build` in frontend/ (deploy-space.bat
does both). It assembles the Space (backend/app, the requirements, frontend/dist
as static/, deploy/space/*) in a temporary folder and uploads it as one commit.

The Hugging Face token comes from, in order: HF_TOKEN in the environment, the
token saved in Syntra (Settings > Services > Hugging Face), or `hf auth login`.
It is never printed.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = os.getenv("SYNTRA_SPACE", "Alkamelgis/SYNTRA-AI")


def token() -> str | None:
    if os.getenv("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    secrets = Path(os.getenv("SYNTRA_DATA_DIR", ROOT / "backend" / "data")) / "secrets.json"
    try:
        t = json.loads(secrets.read_text(encoding="utf-8")).get("hf_token")
        if t:
            return t
    except (OSError, ValueError):
        pass
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # noqa: BLE001
        return None


def assemble(dst: Path) -> int:
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        sys.exit("Build the interface first: cd frontend && npm run build")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "backend" / "app", dst / "backend" / "app", ignore=ignore)
    for f in ("requirements-core.txt", "requirements-ml.txt"):
        shutil.copy2(ROOT / "backend" / f, dst / "backend" / f)
    shutil.copytree(dist, dst / "static")
    for f in (ROOT / "deploy" / "space").iterdir():
        shutil.copy2(f, dst / f.name)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def main() -> None:
    from huggingface_hub import HfApi

    tok = token()
    if not tok and sys.stdin.isatty():
        import getpass
        tok = getpass.getpass("Hugging Face token (Write, hidden as you paste): ").strip() or None
    if not tok:
        sys.exit("No Hugging Face token. Run: hf auth login   (or save one in Syntra: Settings > Services)")
    with tempfile.TemporaryDirectory() as tmp:
        n = assemble(Path(tmp))
        print(f"Uploading {n} files to {SPACE} …")
        info = HfApi(token=tok).upload_folder(folder_path=tmp, repo_id=SPACE, repo_type="space",
                                              commit_message="Deploy from this computer",
                                              delete_patterns=["backend/**", "static/**"])
    print(f"Done: https://huggingface.co/spaces/{SPACE}  (commit {str(getattr(info, 'oid', ''))[:7]})")
    print(f"The app: https://{SPACE.replace('/', '-').lower()}.hf.space  (building takes a few minutes)")


if __name__ == "__main__":
    main()
