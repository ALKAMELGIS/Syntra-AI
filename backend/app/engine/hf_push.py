"""
Save an exported training dataset to a private Hugging Face dataset repo.

One push = one folder in the repo (`<date>_<image>_<task>/`) and one git
commit, so every version stays addressable and nothing is overwritten. The
folder is the YOLO dataset exactly as Syntra exported it (data.yaml, images/,
labels/) plus `syntra_source.json` with what the tiles do not carry: the
source image, its CRS, bounds, ground resolution, the classes and counts.

Privacy: a new repo is always created private, and a push into an existing
public repo is refused unless `allow_public` is set, because samples cut from
licensed commercial imagery must not end up public by accident.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

MAX_FILES_PER_FOLDER = 10_000          # Hugging Face limit per directory


class PushError(RuntimeError):
    pass


def _api(token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise PushError("The huggingface_hub package is missing. In the Syntra environment run: pip install huggingface_hub")
    return HfApi(token=token)


def whoami(token: str) -> str:
    """The account name the token belongs to; raises PushError if it is not valid."""
    try:
        info = _api(token).whoami()
    except PushError:
        raise
    except Exception as exc:  # noqa: BLE001 - shown to the user, worded by cause
        raise PushError(_explain(exc))
    return info.get("name") or ""


def _explain(exc: Exception) -> str:
    """A wrong token and an unreachable site need different fixes: say which one it is."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc)
    if status is None:
        for code in (401, 403, 407, 429, 500, 502, 503):
            if text.startswith(str(code)) or f" {code} " in f" {text} ":
                status = code
                break
    if status == 401:
        return "Hugging Face did not accept this token. Create a token with Write access in Settings > Access Tokens"
    if status == 403 and "huggingface" in text.lower() and "proxy" not in text.lower():
        return "This token has no write access. Create one with Write permission in Settings > Access Tokens"
    if status in (403, 407):
        return f"Could not reach Hugging Face (HTTP {status}): a proxy or firewall blocks huggingface.co on this network"
    if status:
        return f"Hugging Face answered HTTP {status}. Try again in a minute"
    return f"Could not reach Hugging Face: check the internet connection ({type(exc).__name__})"


def check_folder_sizes(dataset_dir: Path) -> None:
    for d in [dataset_dir, *[p for p in dataset_dir.rglob("*") if p.is_dir()]]:
        n = sum(1 for _ in d.iterdir())
        if n >= MAX_FILES_PER_FOLDER:
            raise PushError(f"{d.relative_to(dataset_dir)} holds {n} files; Hugging Face allows under "
                            f"{MAX_FILES_PER_FOLDER} per folder. Export with a larger tile or fewer areas.")


SHARD_FILES = 2000          # files per tar shard: few, large files instead of thousands of small ones


def stage(dataset_dir: Path, staging: Path, source: dict) -> int:
    """What goes up: the small files as they are, each split's tiles and labels
    packed into tar shards (shards/<split>-000.tar, unpacked by the GPU), and
    the owner's original labels (library/<class>/labels/<image>.geojson)."""
    import shutil
    import tarfile

    staging.mkdir(parents=True, exist_ok=True)
    for f in dataset_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, staging / f.name)
    n = 0
    for split in ("train", "val", "test"):
        files = sorted([*(dataset_dir / "images" / split).glob("*"), *(dataset_dir / "labels" / split).glob("*")])
        files = [f for f in files if f.is_file()]
        for k in range(0, len(files), SHARD_FILES):
            (staging / "shards").mkdir(exist_ok=True)
            with tarfile.open(staging / "shards" / f"{split}-{k // SHARD_FILES:03d}.tar", "w") as tar:
                for f in files[k:k + SHARD_FILES]:
                    tar.add(f, arcname=str(f.relative_to(dataset_dir)).replace("\\", "/"))
            n += 1
    for rel in source.get("label_files") or []:
        src = Path(source["classes_dir"]) / rel
        if src.is_file():
            dst = staging / "library" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return n


def push_dataset(dataset_dir: Path, repo_id: str, token: str, source: dict, *,
                 allow_public: bool = False, progress=lambda pct, msg=None: None) -> dict:
    import tempfile

    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "data.yaml").exists():
        raise PushError("This export has no data.yaml: export the dataset again")
    api = _api(token)
    progress(5, "Checking the repo")
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        if not getattr(info, "private", True) and not allow_public:
            raise PushError(f"{repo_id} is public. Samples from licensed imagery go to a private repo: "
                            "choose another name, or make this repo private on Hugging Face")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if source.get("dataset") and source.get("version"):          # datasets/<class>/vNNN -> <class>/vNNN
        folder = f"{source['dataset']}/{source['version']}"
    else:
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in Path(source.get("image_name") or "image").stem)[:48]
        folder = f"{stamp}_{stem}_{source.get('task', 'detect')}"
    meta = {k: v for k, v in source.items() if k not in ("classes_dir",)}
    with tempfile.TemporaryDirectory(prefix="syntra-push-") as tmp:
        staging = Path(tmp) / "up"
        progress(10, "Packing tiles")
        shards = stage(dataset_dir, staging, source)
        (staging / "syntra_source.json").write_text(json.dumps({**meta, "pushed_at": stamp, "folder": folder}, indent=1))
        n_files = sum(1 for p in staging.rglob("*") if p.is_file())
        check_folder_sizes(staging)
        progress(20, f"Uploading {n_files} files ({shards} tile packs)")
        commit = api.upload_folder(folder_path=str(staging), path_in_repo=folder, repo_id=repo_id,
                                   repo_type="dataset", commit_message=f"Syntra samples: {folder}")
    progress(100, "Saved")
    return {"repo_id": repo_id, "folder": folder, "files": n_files,
            "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{folder}",
            "commit": getattr(commit, "oid", None) or str(commit)}
