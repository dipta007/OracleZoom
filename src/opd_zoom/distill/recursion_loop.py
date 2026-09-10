"""Recursion-in-the-loop helpers for B3 (multi-scale on-policy).

CoZ 16x = 4x applied twice: crop the center 1/upscale of the 4x output, bicubic-upscale it
back to process_size, run the 4x SR again. These ops are differentiable so pass-2 loss flows
back through the crop/resize into pass-1 and the LoRA.
"""
import torch.nn.functional as F


def center_quarter(img, factor: int = 4):
    """Center 1/factor square crop. img: (1,3,H,W). CoZ crops center min-side/factor each
    recursion; here H==W==process_size so it is a clean center square. Differentiable (slice)."""
    _, _, h, w = img.shape
    nh, nw = h // factor, w // factor
    top, left = (h - nh) // 2, (w - nw) // 2
    return img[..., top:top + nh, left:left + nw]


def bicubic_up(img, size: int):
    """Bicubic upscale to (size,size). Differentiable. Matches CoZ's per-recursion resize."""
    return F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def recursive_deep_pair(deep_render, gt4x, factor: int = 4):
    """Align a deep (e.g. 16x) render with the real 4x GT for a cycle-consistency loss.

    deep_render: (1,3,H,W) the student's SECOND-pass output (a zoom into the center 1/factor of
    the 4x frame), in [0,1]. gt4x: (1,3,H,W) the real 4x GT, in [0,1].

    Returns (a_deep, a_gt), both (1,3,H//factor,W//factor):
      a_deep = deep_render downsampled by factor  -> back to the 4x-frame scale
      a_gt   = center 1/factor crop of gt4x        -> the SAME region the deep render zoomed into
    LPIPS(a_deep, a_gt) is then a differentiable, region-aligned reward. Identical geometry to
    anchor.anchored_pair; named for the recursion context and kept beside center_quarter so the
    crop convention is defined once.
    """
    _, _, h, w = deep_render.shape
    nh, nw = h // factor, w // factor
    a_deep = F.interpolate(deep_render, size=(nh, nw), mode="bicubic", align_corners=False).clamp(0, 1)
    top, left = (h - nh) // 2, (w - nw) // 2
    a_gt = gt4x[..., top:top + nh, left:left + nw]
    return a_deep, a_gt
