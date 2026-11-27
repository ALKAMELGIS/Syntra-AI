"""The interface stays on its design tokens (frontend/src/styles): type scale,
4px spacing grid, named stacking layers and shadows. Fails on a raw value so
drift is caught when it is written, not found later as overlap or uneven UI."""
import re
from pathlib import Path

STYLES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "styles"
APP = (STYLES / "app.css").read_text(encoding="utf-8")
TOKENS = (STYLES / "tokens.css").read_text(encoding="utf-8")


def test_font_sizes_come_from_the_scale():
    raw = re.findall(r"font-size:\s*[0-9.]+px", APP)
    assert not raw, f"use var(--fs-*): {raw[:5]}"
    used = set(re.findall(r"var\(--fs-(\d+)\)", APP))
    defined = set(re.findall(r"--fs-(\d+):", TOKENS))
    assert used <= defined and len(defined) <= 7


def test_spacing_is_on_the_4px_grid():
    bad = []
    for prop, val in re.findall(r"((?:padding|margin|gap|row-gap|column-gap)(?:-[a-z]+)?)\s*:([^;{}]+)", APP):
        for n in re.findall(r"(?<![\w.])-?(\d+)px", val):
            if int(n) > 2 and int(n) % 4:
                bad.append(f"{prop}: {val.strip()}")
    assert not bad, f"off the 4px grid: {bad[:5]}"


def test_stacking_layers_are_named():
    raw = [int(v) for v in re.findall(r"z-index:\s*(\d+)", APP) if int(v) >= 100]
    assert not raw, f"use var(--z-*) for page-level layers: {raw}"


def test_shadows_are_tokens():
    assert not re.findall(r"box-shadow:\s*0 \d+px \d+px rgba\(0", APP)
