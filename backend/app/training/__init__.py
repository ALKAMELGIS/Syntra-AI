"""
Model training pipeline — ported from the GeoSyntra AI QGIS plugin's
methodology (Dataset Exporter -> Model Trainer -> Object Detector), with the
QGIS-specific GUI/GDAL glue replaced by this project's own rasterio-based
raster I/O (see app/engine/raster_io.py) and plain Python elsewhere.

Submodules:
  geometry.py        pure-NumPy polygon/tile-grid math (no GIS dependency)
  augment.py         90/180/270 rotation + flip augmentation for tiles
  raster_window.py   windowed raster reads + tile image writing (rasterio)
  dataset_builder.py tiles a large orthomosaic + annotations into a YOLO
                      dataset (images/labels/train+val, data.yaml)
"""
