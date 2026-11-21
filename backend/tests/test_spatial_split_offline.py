"""Train / val / test split by area, not per tile: neighbouring, overlapping
tiles must never sit on both sides of a split."""
import csv
import os

import numpy as np

from app.training.dataset_builder import DatasetBuilder
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_training_dataset_builder_offline import FakeReader, _box  # noqa: E402


def build(tmp_path, test_areas=None, test_anns=None, val_split=0.25):
    r = FakeReader(width=6000, height=4000)
    anns = [{"cls": 0, "pts": _box(x, y, 20)} for x in range(150, 5900, 300) for y in range(150, 3900, 300)]
    out = tmp_path / "ds"
    b = DatasetBuilder(r, anns, ["ship"], str(out), tile=640, overlap=64, val_split=val_split, augment=False,
                       neg_ratio=0.0, test_areas=test_areas, test_annotations=test_anns, img_fmt="jpg")
    b.run()
    rows = list(csv.DictReader(open(out / "tiles_index.csv")))
    return b, rows, out


def grid(rows, step=576):
    return {(int(r["y_offset_px"]) // step, int(r["x_offset_px"]) // step): r["split"] for r in rows}


def test_no_train_tile_touches_a_val_tile(tmp_path):
    b, rows, _ = build(tmp_path)
    g = grid(rows)
    val = [k for k, s in g.items() if s == "val"]
    assert val and any(s == "train" for s in g.values())
    for (j, i) in val:
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                assert g.get((j + dj, i + di)) in (None, "val"), "a train tile overlaps a val tile"
    assert b.stats["buffer_dropped"] > 0 and b.split_info["method"] == "spatial blocks"
    share = b.stats["val_images"] / (b.stats["val_images"] + b.stats["train_images"])
    assert 0.1 < share < 0.45


def test_accuracy_areas_become_the_labelled_test_split(tmp_path):
    area = np.array([[4300, 2600], [5800, 2600], [5800, 3800], [4300, 3800]], float)
    inside = [{"cls": 0, "pts": _box(x, y, 20)} for x in range(4350, 5800, 300) for y in range(2650, 3800, 300)]
    b, rows, out = build(tmp_path, test_areas=[area], test_anns=inside)
    g = grid(rows)
    test = [k for k, s in g.items() if s == "test"]
    assert test and b.stats["test_objects"] >= len(inside)
    for (j, i) in test:                                     # nothing trained right next to the test area
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                assert g.get((j + dj, i + di)) in (None, "test")
    assert os.listdir(out / "labels" / "test")
    assert "test: images/test" in (out / "data.yaml").read_text()


def test_a_small_image_still_trains_and_the_manifest_says_the_gap_was_dropped(tmp_path):
    r = FakeReader(width=1920, height=1080)
    anns = [{"cls": 0, "pts": _box(x, y, 15)} for x in range(80, 1900, 160) for y in range(80, 1060, 160)]
    area = np.array([[1056, 54], [1824, 54], [1824, 486], [1056, 486]], float)
    inside = [a for a in anns if 1056 < a["pts"][0][0] < 1800 and a["pts"][0][1] < 470]
    b = DatasetBuilder(r, anns, ["boat"], str(tmp_path / "ds"), augment=False, neg_ratio=0.0,
                       test_areas=[area], test_annotations=inside)
    b.run()
    assert b.stats["train_images"] > 0 and b.stats["test_images"] > 0
    assert b.split_info["test_buffer_tiles"] == 0 and b.split_info["notes"]
