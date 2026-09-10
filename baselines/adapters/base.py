"""SRAdapter: uniform x4 super-resolution interface for baseline models (task #37).

Each baseline implements this so the model-agnostic recursive_render.py can drive any of them
identically. ISOLATED: adapters never import into or modify src/opd_zoom (our running eval code).
"""
from abc import ABC, abstractmethod

from PIL import Image


class SRAdapter(ABC):
    """One baseline SR model, exposed as a single blind x4 op (no prompt / no privilege)."""

    name = "base"

    @abstractmethod
    def __init__(self, ckpt_dir: str, device: str = "cuda"):
        """Load weights from ckpt_dir onto device."""

    @abstractmethod
    def sr_x4(self, img: "Image.Image") -> "Image.Image":
        """Super-resolve a PIL RGB image by 4x (H,W -> 4H,4W). Deterministic, no text prompt.
        recursive_render normalizes the output back to process_size if a model returns another size."""
