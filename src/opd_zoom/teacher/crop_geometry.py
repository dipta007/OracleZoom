# src/opd_zoom/teacher/crop_geometry.py
"""Map a CoZ recursion index to the native ground-truth pixel window.

CoZ center-crops 1/upscale each recursion from a 512 canvas that maps to the GT's center
min(W,H) square. So the window viewed at recursion i is the center min(W,H)/upscale**i square
of the native GT, resized to process_size. We report whether that crop is REAL detail
(native side >= process_size, i.e. a downscale) or UPSAMPLED (native side < process_size).
"""
from PIL import Image


def native_side(min_side: int, i: int, upscale: int = 4) -> float:
    """Side length (native GT px) of the window viewed at recursion i."""
    return min_side / (upscale ** i)


def gt_window(gt: Image.Image, i: int, process_size: int = 512, upscale: int = 4):
    """Return (crop_process_size, is_real, upsample_factor) for recursion i.

    is_real: True if native window side >= process_size (real detail, no upsampling).
    upsample_factor: process_size / side when upsampled, else 1.0.
    """
    w, h = gt.size
    side = native_side(min(w, h), i, upscale)
    side_px = max(1, int(round(side)))
    left = (w - side_px) // 2
    top = (h - side_px) // 2
    crop = gt.crop((left, top, left + side_px, top + side_px)).resize(
        (process_size, process_size), Image.LANCZOS
    )
    is_real = side_px >= process_size
    upsample_factor = 1.0 if is_real else process_size / side_px
    return crop, is_real, upsample_factor
