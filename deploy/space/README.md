---
title: Syntra AI
emoji: 🛰️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Objects in satellite and drone GeoTIFFs, as GIS layers
---

# Syntra AI

Finds objects in satellite and drone GeoTIFFs and hands them back as GIS layers
(GeoJSON, KML, Shapefile). This Space runs the engine and the interface in one
process; open it at its own address (`https://alkamelgis-syntra-ai.hf.space`),
not inside the Hugging Face page, so the sign-in cookie works.

Deployed from GitHub by `.github/workflows/deploy-space.yml`; do not edit here.
