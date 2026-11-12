"""
Auto-tuning heuristics for the Train step: suggest a batch size, a worker
(DataLoader) count, and an image size that fit the machine actually running
the job, instead of making the user guess. Pure Python/stdlib (`os`,
`shutil`) — no torch/ultralytics needed, so these run everywhere, including
the API-only process and this offline test suite.

Ported from the GeoSyntra AI QGIS plugin's core/autoparams.py: same
reasoning (CPU-count-based workers, memory-based batch size, device-aware
defaults for "works great on CPU"), re-expressed against this project's own
app.core.compute device profile instead of QGIS's own hardware probing.
"""
from __future__ import annotations

import os


def suggest_workers(cpu_threads: int | None = None) -> int:
    """DataLoader workers: leave at least one core free, cap at 8 — more than
    that mostly adds process overhead for typical tile-sized datasets."""
    n = cpu_threads if cpu_threads is not None else (os.cpu_count() or 4)
    return max(0, min(8, n - 1))


def suggest_batch_size(imgsz: int, device: str, free_vram_mb: int | None = None,
                        available_ram_mb: int | None = None) -> int:
    """A conservative batch size: scales down for larger images, and down
    further on CPU (no VRAM headroom to lean on, and a huge batch just makes
    each CPU step slower without a speed benefit)."""
    base = {320: 32, 416: 24, 512: 20, 640: 16, 768: 12, 896: 8, 1024: 6}
    tiers = sorted(base)
    size = next((base[t] for t in tiers if imgsz <= t), base[tiers[-1]])

    if device == "cpu":
        size = max(2, size // 2)
        if available_ram_mb is not None and available_ram_mb < 4096:
            size = max(2, size // 2)
        return size

    if free_vram_mb is not None:
        # roughly, each sample at `imgsz` costs ~ (imgsz/640)^2 * 90MB at fp32
        # train-time (activations + gradients) for a small/medium model
        per_sample_mb = max(20, 90 * (imgsz / 640.0) ** 2)
        vram_cap = max(1, int((free_vram_mb * 0.7) // per_sample_mb))
        size = min(size, vram_cap)
    return max(1, size)


def suggest_imgsz(median_object_px: float | None, tile_size: int = 640) -> int:
    """Round the dataset's tile size up to the nearest multiple of 32
    (Ultralytics requires stride-32-aligned sizes); small objects benefit
    from not downscaling below the tile size."""
    size = max(int(tile_size), 320)
    return int(((size + 31) // 32) * 32)


def suggest_epochs(num_train_images: int) -> int:
    """More epochs for small datasets (which converge on less signal per
    epoch and benefit from more passes), fewer for large ones."""
    if num_train_images < 200:
        return 150
    if num_train_images < 1000:
        return 100
    if num_train_images < 5000:
        return 60
    return 40


def suggest_training_params(device: str, imgsz: int, num_train_images: int,
                             cpu_threads: int | None = None,
                             free_vram_mb: int | None = None,
                             available_ram_mb: int | None = None) -> dict:
    """One-call convenience bundling the suggestions above, mirroring what
    the QGIS plugin's Train step auto-fills before the user can override it."""
    return {
        "batch": suggest_batch_size(imgsz, device, free_vram_mb, available_ram_mb),
        "workers": suggest_workers(cpu_threads),
        "epochs": suggest_epochs(num_train_images),
        "imgsz": suggest_imgsz(imgsz if imgsz else None, tile_size=imgsz or 640),
    }
