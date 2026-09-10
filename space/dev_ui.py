"""Run the interface locally with the models stubbed out, for laying out the UI.

No GPU, no weights, no downloads. The fake zoom just bicubic-enlarges, so the pictures are
meant to be blurry: this is for arranging the page, not for judging output.

    uv run --no-project --python 3.12 --with "gradio==5.50.0" --with spaces --with pillow \
        --with numpy --with imageio --with imageio-ffmpeg python space/dev_ui.py
"""
import os
import sys
import types

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import geometry  # noqa: E402  (after the path insert, on purpose)

FAKE_PROMPTS = [
    "spotted fur, whiskers, amber eye, fine hairs",
    "dense guard hairs, rosette edge, warm tones",
    "single hair shafts, follicle, soft focus",
    "fine keratin fibre, grain, pale strands",
    "fibre bundle, faint striations",
    "surface texture, low contrast",
    "granular detail, muted",
    "flat texture, little structure",
]


def fake_zoom(models, image, levels=4, upscale=4, center=(0.5, 0.5)):
    cur = geometry.resize_and_center_crop(image)
    yield 0, 1, "", cur, cur
    for i in range(levels):
        blurry = geometry.zoom_window(cur, upscale, center).resize(cur.size, Image.BICUBIC)
        yield i + 1, upscale ** (i + 1), FAKE_PROMPTS[i % len(FAKE_PROMPTS)], blurry, blurry
        cur = blurry


stub = types.ModuleType("zoom")
stub.Models = type("Models", (), {"__init__": lambda self, weights=None: None})
stub.zoom = fake_zoom
sys.modules["zoom"] = stub

import app  # noqa: E402  (must follow the stub)

BANNER = (
    '<div style="background:#7f1d1d;color:#fff;padding:10px 14px;border-radius:8px;'
    'margin:0 0 12px;text-align:center;font-weight:600">'
    'LAYOUT PREVIEW ONLY. The model is stubbed out and every zoom is a plain bicubic '
    'enlargement, so the pictures are meant to look blurry. Judge the layout here, '
    'never the output.</div>'
)


def _add_banner():
    """Say it on the page, not just in the terminal. A blurry zoom here is the stub, not a bug."""
    for block in app.demo.blocks.values():
        if isinstance(block, __import__("gradio").HTML) and "OracleZoom" in str(block.value):
            block.value = BANNER + block.value
            return True
    return False


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    if not _add_banner():
        print("WARNING: could not add the stub banner; the header markup must have changed")
    print(f"stubbed UI on http://127.0.0.1:{port}  (blurry output is expected, it is not the model)")
    app.demo.queue().launch(server_port=port, quiet=True)
