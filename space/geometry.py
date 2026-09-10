"""Crop geometry for the zoom. Pure PIL, no model code, so it stays cheap to test."""
from PIL import Image

PROCESS_SIZE = 512


def resize_and_center_crop(img, size=PROCESS_SIZE):
    """CoZ's entry crop: shortest side to `size`, then the centre square."""
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def window_rect(size, upscale, center=(0.5, 0.5)):
    """The next level's field of view: a 1/upscale square around `center`, clamped inside."""
    w, h = size
    nw, nh = w // upscale, h // upscale
    left = min(max(int(center[0] * w - nw / 2), 0), w - nw)
    top = min(max(int(center[1] * h - nh / 2), 0), h - nh)
    return left, top, left + nw, top + nh


def zoom_window(img, upscale, center=(0.5, 0.5)):
    return img.crop(window_rect(img.size, upscale, center))


def nested_rects(size, levels, upscale=4, center=(0.5, 0.5)):
    """Where each zoom step lands, in the coordinates of the first canvas.

    Stops early once a window would be under a pixel wide. The recursion itself never
    degenerates, because every step resizes its crop back up to the full canvas, but drawing
    all those steps onto the one original canvas does run out of pixels.
    """
    rects, rect = [], (0, 0, size[0], size[1])
    for _ in range(levels):
        side = rect[2] - rect[0]
        if side // upscale < 1:
            break
        l, t, r, b = window_rect((side, side), upscale, center)
        rect = (rect[0] + l, rect[1] + t, rect[0] + r, rect[1] + b)
        rects.append(rect)
    return rects
