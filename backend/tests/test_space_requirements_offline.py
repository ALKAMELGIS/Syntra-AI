"""The Hugging Face Space build mounts only deploy/space/requirements.txt, so it can't use
-r includes; it must list every backend requirement itself."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _reqs(p: Path) -> set[str]:
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.add(line)
    return out


def test_space_requirements_are_flat_and_complete():
    space = ROOT / "deploy" / "space" / "requirements.txt"
    assert not any(l.strip().startswith("-r") for l in space.read_text(encoding="utf-8").splitlines())
    need = _reqs(ROOT / "backend" / "requirements-core.txt") | _reqs(ROOT / "backend" / "requirements-ml.txt")
    assert need <= _reqs(space)
