"""Windows reads text in its own code page (cp1252) unless told otherwise: training
failed there with "'charmap' codec can't decode byte 0x81". Every text read, write
and subprocess pipe in the engine must name its encoding."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"


def _kw(call: ast.Call, name: str) -> bool:
    return any(k.arg == name for k in call.keywords)


def _binary_mode(call: ast.Call, pos: int) -> bool:
    mode = call.args[pos] if len(call.args) > pos else next((k.value for k in call.keywords if k.arg == "mode"), None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def test_every_text_io_names_its_encoding():
    bad = []
    for f in APP.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
            where = f"{f.relative_to(APP)}:{node.lineno}"
            if name in ("read_text", "write_text") and not _kw(node, "encoding"):
                bad.append(where)
            elif name == "open" and isinstance(fn, ast.Name) and not _kw(node, "encoding") and not _binary_mode(node, 1):
                bad.append(where)
            elif name == "open" and isinstance(fn, ast.Attribute) and fn.attr == "open" and isinstance(fn.value, ast.Name) \
                    and fn.value.id not in ("rasterio", "zipfile", "tarfile", "_tar", "Image", "gzip", "io") \
                    and not _kw(node, "encoding") and not _binary_mode(node, 0) and node.args and isinstance(node.args[0], ast.Constant):
                bad.append(where)
            elif name in ("Popen", "run", "check_output") and (_kw(node, "text") or _kw(node, "universal_newlines")) and not _kw(node, "encoding"):
                bad.append(where)
    assert not bad, f"name the encoding (utf-8): {bad}"
