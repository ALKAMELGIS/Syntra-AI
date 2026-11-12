"""
Offline test for app.training.dataset_builder.DatasetBuilder using a fake,
in-memory raster reader — no rasterio, no real GeoTIFF needed. Tile image
writing goes through app.training.raster_window.write_image, which uses PIL
for jpg/png (already a project dependency), so this still runs without
rasterio installed as long as the output format stays "jpg"/"png".

Mirrors how the GeoSyntra AI QGIS plugin's own dataset_builder.py was
designed: "No QGIS dependency: annotations arrive already converted to pixel
space" — so the same holds here for rasterio/GDAL.
"""
import json
import os

import numpy as np
import pytest

from app.training.dataset_builder import DatasetBuilder
from app.training.raster_window import RasterReader


class FakeReader:
    """Duck-typed stand-in for app.training.raster_window.RasterReader."""

    def __init__(self, width=1200, height=900, path="/fake/ortho.tif"):
        self.width = width
        self.height = height
        self.count = 3
        self.dtype = "uint8"
        self.path = path
        self.wkt = ""
        self.gt = (0.0, 0.1, 0.0, 0.0, 0.0, -0.1)  # arbitrary georeference

    def read_window(self, xoff, yoff, w, h, out_size=None):
        ow, oh = out_size if out_size else (w, h)
        # mid-gray tile (never "black") with a bit of per-tile variation so
        # augmentation doesn't operate on a perfectly uniform image
        rgb = np.full((oh, ow, 3), 120, dtype=np.uint8)
        rgb[: oh // 2, : ow // 2] = 160
        valid = np.ones((oh, ow), dtype=bool)
        return rgb, valid

    black_ratio = staticmethod(RasterReader.black_ratio)

    def tile_geotransform(self, xoff, yoff, scale=1.0):
        gt = self.gt
        ox = gt[0] + xoff * gt[1] + yoff * gt[2]
        oy = gt[3] + xoff * gt[4] + yoff * gt[5]
        return (ox, gt[1] * scale, gt[2] * scale, oy, gt[4] * scale, gt[5] * scale)


def _box(cx, cy, half):
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                      [cx + half, cy + half], [cx - half, cy + half]], dtype=float)


def _annotations():
    # three boxes spread across the fake 1200x900 raster, two classes
    return [
        {"cls": 0, "pts": _box(100, 100, 30)},
        {"cls": 0, "pts": _box(700, 400, 25)},
        {"cls": 1, "pts": _box(1000, 800, 40)},
    ]


def test_dataset_builder_produces_expected_outputs(tmp_path):
    reader = FakeReader()
    builder = DatasetBuilder(
        reader, _annotations(), class_names=["tree", "building"], out_dir=str(tmp_path),
        tile=256, overlap=32, val_split=0.3, neg_ratio=0.2, augment=True, seed=7,
    )
    yaml_path, info = builder.run()

    assert os.path.isfile(yaml_path)
    assert os.path.isfile(tmp_path / "classes.txt")
    assert os.path.isfile(tmp_path / "tiles_index.csv")
    assert os.path.isfile(tmp_path / "dataset_info.json")

    with open(tmp_path / "dataset_info.json", encoding="utf-8") as fh:
        saved_info = json.load(fh)
    assert saved_info["classes"] == ["tree", "building"]
    assert saved_info["tile_size"] == 256

    # at least one image + one matching label file were written for training
    train_images = os.listdir(tmp_path / "images" / "train")
    train_labels = os.listdir(tmp_path / "labels" / "train")
    assert len(train_images) > 0
    assert len(train_labels) > 0
    stems_img = {os.path.splitext(f)[0] for f in train_images}
    stems_lab = {os.path.splitext(f)[0] for f in train_labels}
    assert stems_img == stems_lab          # every tile has exactly one label file

    # every positive (non-background) tile's label file references a valid class id
    found_any_label = False
    for f in train_labels:
        with open(tmp_path / "labels" / "train" / f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                found_any_label = True
                cls_id = int(line.split()[0])
                assert cls_id in (0, 1)
    assert found_any_label

    # validation set is never empty — DatasetBuilder backfills from train if needed
    val_images = os.listdir(tmp_path / "images" / "val")
    assert len(val_images) > 0

    assert info["stats"]["class_counts"]["tree"] >= 1
    assert info["stats"]["class_counts"]["building"] >= 1


def test_dataset_builder_backfills_empty_validation_when_too_few_tiles(tmp_path):
    reader = FakeReader(width=400, height=400)
    builder = DatasetBuilder(
        reader, [{"cls": 0, "pts": _box(50, 50, 20)}], class_names=["tree"], out_dir=str(tmp_path),
        tile=256, overlap=32, val_split=0.2, neg_ratio=0.0, augment=False, seed=1,
    )
    yaml_path, info = builder.run()
    assert os.path.isfile(yaml_path)
    # only one annotated tile exists -> val would naturally be empty without the backfill
    assert info["stats"]["val_images"] > 0
    assert len(os.listdir(tmp_path / "images" / "val")) > 0


def test_dataset_builder_respects_cancellation(tmp_path):
    reader = FakeReader()
    builder = DatasetBuilder(
        reader, _annotations(), class_names=["tree", "building"], out_dir=str(tmp_path),
        tile=256, overlap=32, canceled=lambda: True,
    )
    with pytest.raises(Exception):
        builder.run()
