"""Offline tests for the dataset module: GeoJSON parsing, validation, class
balance, and the geographically separated splitter. Pure stdlib + the repo's
own code — no rasterio/torch needed."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.datasets.annotations import parse_geojson_annotations, validate_dataset
from app.datasets.schema import BoundingBox, DatasetSample
from app.datasets.splitter import class_balance, geographically_separated_split


def _write_geojson(tmp: Path, cells: list[tuple[float, float]]) -> Path:
    """One small square polygon per (x, y) origin, 10x10 units."""
    features = []
    for i, (x, y) in enumerate(cells):
        ring = [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10], [x, y]]
        features.append({"type": "Feature", "id": i, "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}})
    path = tmp / "fields.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_geojson_parsing_and_validation():
    tmp = Path(tempfile.mkdtemp())
    path = _write_geojson(tmp, [(0, 0), (100, 0), (0, 100)])
    samples = parse_geojson_annotations(path)
    assert len(samples) == 3
    assert samples[0].geo_bounds == (0.0, 0.0, 10.0, 10.0)

    # validate_dataset flags polygon-only samples as fine (they have polygon_rings)
    problems = validate_dataset(samples)
    assert problems == [], f"unexpected problems: {problems}"
    print("GeoJSON parse + validate: OK")


def test_class_balance():
    samples = [
        DatasetSample(sample_id="a", image_path="a.tif", boxes=[
            BoundingBox(0.5, 0.5, 0.1, 0.1, class_id=0),
            BoundingBox(0.3, 0.3, 0.1, 0.1, class_id=0),
        ]),
        DatasetSample(sample_id="b", image_path="b.tif", boxes=[
            BoundingBox(0.5, 0.5, 0.1, 0.1, class_id=1),
        ]),
    ]
    report = class_balance(samples)
    assert report.total == 3
    assert report.counts == {0: 2, 1: 1}
    assert 1 in report.underrepresented_classes or report.imbalance_ratio == 2.0
    print("Class balance:", report.summary())


def test_geographic_split_keeps_cells_together():
    # Two tight clusters far apart (different grid cells at cell_size=1000);
    # within each cluster the samples must never straddle different splits.
    samples = []
    cluster_a = [(0 + dx, 0 + dy) for dx in (0, 50) for dy in (0, 50)]           # 4 tiles, same cell
    cluster_b = [(5000 + dx, 5000 + dy) for dx in (0, 50) for dy in (0, 50)]     # 4 tiles, different cell
    for i, (x, y) in enumerate(cluster_a):
        samples.append(DatasetSample(sample_id=f"a{i}", image_path="", geo_bounds=(x, y, x + 10, y + 10)))
    for i, (x, y) in enumerate(cluster_b):
        samples.append(DatasetSample(sample_id=f"b{i}", image_path="", geo_bounds=(x, y, x + 10, y + 10)))

    split = geographically_separated_split(samples, train_frac=0.5, val_frac=0.5, test_frac=0.0, cell_size=1000.0, seed=1)
    assert split.geographically_separated is True

    all_ids = set(split.train) | set(split.val) | set(split.test)
    assert all_ids == {s.sample_id for s in samples}

    cluster_a_ids = {f"a{i}" for i in range(4)}
    cluster_b_ids = {f"b{i}" for i in range(4)}
    for cluster in (cluster_a_ids, cluster_b_ids):
        in_train = cluster & set(split.train)
        in_val = cluster & set(split.val)
        in_test = cluster & set(split.test)
        occupied = [s for s in (in_train, in_val, in_test) if s]
        assert len(occupied) == 1, f"cluster {cluster} was split across multiple partitions: {occupied}"
    print("Geographic split: both clusters stayed intact ->", split.notes)


if __name__ == "__main__":
    test_geojson_parsing_and_validation()
    test_class_balance()
    test_geographic_split_keeps_cells_together()
    print("test_datasets_offline: PASSED")
