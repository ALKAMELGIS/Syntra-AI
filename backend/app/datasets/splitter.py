"""
Train/validation/test splitting — with geographic separation to avoid
spatial data leakage (spec Section 9: "Use geographically separated
evaluation datasets where possible").

Why this matters: if two tiles from the same plantation/parking lot end up
one in train and one in val, the model can effectively memorize that scene
and validation metrics become optimistic. Instead of a plain random split,
when samples carry `geo_bounds`, we bin samples into spatial grid cells and
assign *whole cells* to a split, so neighboring tiles never straddle the
train/val boundary.
"""
from __future__ import annotations

import random
from collections import defaultdict

from app.datasets.schema import ClassBalanceReport, DatasetSample, DatasetSplit


def class_balance(samples: list[DatasetSample]) -> ClassBalanceReport:
    counts: dict[int, int] = defaultdict(int)
    for s in samples:
        for b in s.boxes:
            counts[b.class_id] += 1

    total = sum(counts.values())
    if not counts:
        return ClassBalanceReport(counts={}, total=0, imbalance_ratio=1.0, underrepresented_classes=[])

    mean = total / len(counts)
    underrepresented = [c for c, n in counts.items() if n < 0.10 * mean]
    imbalance_ratio = max(counts.values()) / max(1, min(counts.values()))

    return ClassBalanceReport(counts=dict(counts), total=total, imbalance_ratio=imbalance_ratio,
                               underrepresented_classes=sorted(underrepresented))


def _grid_cell(bounds: tuple[float, float, float, float], cell_size: float) -> tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return int(cx // cell_size), int(cy // cell_size)


def geographically_separated_split(
    samples: list[DatasetSample],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    cell_size: float = 1000.0,     # in the same units as geo_bounds (e.g. meters for a projected CRS)
    seed: int = 42,
) -> DatasetSplit:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    geo_samples = [s for s in samples if s.geo_bounds is not None]
    non_geo_samples = [s for s in samples if s.geo_bounds is None]

    rng = random.Random(seed)

    if geo_samples:
        cells: dict[tuple[int, int], list[str]] = defaultdict(list)
        for s in geo_samples:
            cells[_grid_cell(s.geo_bounds, cell_size)].append(s.sample_id)

        cell_keys = list(cells.keys())
        rng.shuffle(cell_keys)

        n_cells = len(cell_keys)
        n_train = max(1, round(n_cells * train_frac)) if n_cells >= 3 else n_cells
        n_val = max(0, round(n_cells * val_frac)) if n_cells >= 3 else 0
        train_cells = cell_keys[:n_train]
        val_cells = cell_keys[n_train:n_train + n_val]
        test_cells = cell_keys[n_train + n_val:]

        train_ids = [sid for c in train_cells for sid in cells[c]]
        val_ids = [sid for c in val_cells for sid in cells[c]]
        test_ids = [sid for c in test_cells for sid in cells[c]]
        notes = f"{len(cell_keys)} spatial cells ({cell_size} units) assigned whole to one split each — no tile straddles train/val/test."
    else:
        train_ids, val_ids, test_ids = [], [], []
        notes = "No geo_bounds on any sample — geographic separation unavailable; falling back to random split."

    # Samples without geo info (or the whole dataset, in the fallback case) get a plain random split.
    pool = non_geo_samples if geo_samples else samples
    ids = [s.sample_id for s in pool]
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    train_ids += ids[:n_train]
    val_ids += ids[n_train:n_train + n_val]
    test_ids += ids[n_train + n_val:]

    return DatasetSplit(
        train=train_ids, val=val_ids, test=test_ids,
        geographically_separated=bool(geo_samples), notes=notes,
    )
