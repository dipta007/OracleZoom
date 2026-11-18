"""Render the zoom levels into one continuous zoom clip.

Each segment pushes in on level i until its frame is exactly the blurry crop that level i+1
was made from, then holds still and cross-fades blurry into sharp. The hold is the point:
it is where the viewer sees what the model actually added.
"""
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from geometry import window_rect

FPS = 30
PUSH_FRAMES = 30
FADE_FRAMES = 14
TAIL_FRAMES = 24


def _ease(t):
    return t * t * (3 - 2 * t)


def _lerp_rect(a, b, t):
    return tuple(round(p + (q - p) * t) for p, q in zip(a, b))


def _frames(levels, upscale, center):
    size = levels[0].size
    full = (0, 0, size[0], size[1])
    for i in range(len(levels) - 1):
        target = window_rect(size, upscale, center)
        for f in range(PUSH_FRAMES):
            # f / (PUSH_FRAMES - 1), not f / PUSH_FRAMES: the last frame has to land exactly on
            # `target`, which is the crop the next level was built from. Otherwise it jumps.
            rect = _lerp_rect(full, target, _ease(f / (PUSH_FRAMES - 1)))
            yield levels[i].crop(rect).resize(size, Image.BICUBIC)
        blurry = np.asarray(levels[i].crop(target).resize(size, Image.BICUBIC), dtype=np.float32)
        sharp = np.asarray(levels[i + 1], dtype=np.float32)
        for f in range(FADE_FRAMES):
            a = _ease((f + 1) / FADE_FRAMES)
            yield Image.fromarray((blurry * (1 - a) + sharp * a).astype(np.uint8))
    for _ in range(TAIL_FRAMES):
        yield levels[-1]


def render(levels, upscale=4, center=(0.5, 0.5)):
    """Write an mp4 of the zoom and return its path. Needs at least two levels."""
    if len(levels) < 2:
        return None
    path = tempfile.mkstemp(suffix=".mp4")[1]
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1) as out:
        for frame in _frames(levels, upscale, center):
            out.append_data(np.asarray(frame.convert("RGB")))
    return path
