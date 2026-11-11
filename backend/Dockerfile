# Syntra AI — Detection Engine backend image.
#
# CPU-only base; for GPU inference, build from an NVIDIA CUDA base image
# instead (e.g. nvidia/cuda:12.4.1-runtime-ubuntu22.04) and install a
# matching torch build. GDAL/rasterio need the system libgdal package.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV SYNTRA_DATA_DIR=/srv/data
VOLUME ["/srv/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
