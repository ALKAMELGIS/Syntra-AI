"""
Global configuration for the Syntra AI Detection Engine.

Values here are safe defaults; everything can be overridden with environment
variables (see `.env.example`) so the same code runs unmodified in dev,
on-prem GPU servers, or containers.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    # Load backend/.env (if present) into the process environment *before*
    # Settings reads any os.getenv() below, and -- critically -- before
    # rasterio/pyproj get imported anywhere (they're lazy-imported per
    # module, but always after app.core.config is first imported). This is
    # also how PROJ_LIB/PROJ_DATA overrides in .env actually take effect;
    # see .env.example for why that one matters on Windows.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed yet -- plain OS env vars still work


class Settings:
    # --- Paths -----------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = Path(os.getenv("SYNTRA_DATA_DIR", BASE_DIR / "data"))
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    RESULTS_DIR: Path = DATA_DIR / "results"
    WEIGHTS_DIR: Path = Path(os.getenv("SYNTRA_WEIGHTS_DIR", BASE_DIR / "weights"))

    # --- Tiling ------------------------------------------------------------
    DEFAULT_TILE_SIZE: int = int(os.getenv("SYNTRA_TILE_SIZE", 1024))
    DEFAULT_TILE_OVERLAP: float = float(os.getenv("SYNTRA_TILE_OVERLAP", 0.2))  # fraction of tile size
    MAX_TILE_SIZE: int = 4096
    MIN_TILE_SIZE: int = 256

    # --- Inference ---------------------------------------------------------
    DEFAULT_BATCH_SIZE: int = int(os.getenv("SYNTRA_BATCH_SIZE", 4))
    DEFAULT_NMS_IOU: float = float(os.getenv("SYNTRA_NMS_IOU", 0.45))
    USE_FP16_IF_AVAILABLE: bool = os.getenv("SYNTRA_FP16", "1") == "1"

    # --- Compute profiles ----------------------------------------------------
    # CPU0 / CPU1 are user-configurable CPU execution profiles (e.g. thread
    # counts / core affinity); GPU uses CUDA when available; AUTO picks the
    # best option for the current hardware + workload (see core/compute.py).
    CPU0_THREADS: int = int(os.getenv("SYNTRA_CPU0_THREADS", 4))
    CPU1_THREADS: int = int(os.getenv("SYNTRA_CPU1_THREADS", os.cpu_count() or 8))

    # --- API ---------------------------------------------------------------
    CORS_ORIGINS: list[str] = os.getenv("SYNTRA_CORS_ORIGINS", "http://localhost:5173").split(",")

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.UPLOAD_DIR, self.RESULTS_DIR, self.WEIGHTS_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
