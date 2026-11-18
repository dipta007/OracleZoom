"""Build the whole Gradio graph with the models stubbed out.

Catches component, parameter and event-wiring mistakes without a GPU, which is most of what
breaks a Space on first deploy. The model path itself is not covered here.

    uv run --no-project --python 3.12 --with pytest --with "gradio==5.50.0" --with spaces \
        --with pillow --with numpy --with imageio --with imageio-ffmpeg \
        pytest space/test_app_smoke.py -q
"""
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def app():
    """Import app.py with `zoom` replaced by a stub that skips the GPU work."""
    import geometry

    stub = types.ModuleType("zoom")
    stub.REPO = "dipta007/OracleZoom"
    stub.Models = type("Models", (), {"__init__": lambda self, weights=None: None})

    def fake_zoom(models, image, levels=4, upscale=4, center=(0.5, 0.5)):
        cur = geometry.resize_and_center_crop(image)
        yield 0, 1, "", cur, cur
        for i in range(levels):
            blurry = geometry.zoom_window(cur, upscale, center).resize(cur.size, Image.BICUBIC)
            yield i + 1, upscale ** (i + 1), f"tag {i}", blurry, blurry
            cur = blurry

    stub.zoom = fake_zoom
    sys.modules["zoom"] = stub
    import app as module
    return module


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (700, 900, 3), dtype=np.uint8))


def test_the_interface_builds(app):
    assert app.demo.blocks and app.demo.fns


def test_preview_returns_a_canvas_for_an_upload_and_for_nothing_yet(app, photo):
    assert app.preview(photo, 4, 0.3, 0.7).size == (512, 512)
    # Never None: a visitor touching a slider before uploading must not blank the overlay.
    assert app.preview(None, 4, 0.5, 0.5).size == (512, 512)


def test_declared_duration_grows_with_depth(app, photo):
    got = [app.estimate_duration(photo, n, 0.5, 0.5) for n in (1, 2, 3, 4)]
    assert got == sorted(got) and all(isinstance(v, int) for v in got)


def test_the_shipped_example_assets_exist(app):
    for path in [app.EXAMPLE_CLIP, *app.EXAMPLE_COMPARE, *(p for p, _ in app.EXAMPLE_LEVELS)]:
        assert os.path.getsize(path) > 1000, path
    assert [label for _, label in app.EXAMPLE_LEVELS] == ["input", "4x", "16x", "64x", "256x"]


def test_labels_cover_every_depth_the_slider_offers(app):
    got = [app.label(app.UPSCALE ** n) for n in range(app.MAX_STEPS + 1)]
    assert got[0] == "input"
    assert got[-1] == "65536x"


def test_the_citation_is_the_one_from_the_readme(app):
    readme = open(os.path.join(os.path.dirname(app.__file__), "..", "README.md")).read()
    assert "2609.06490" in app.BIBTEX and "dipta2026oraclezoom" in app.BIBTEX
    for line in app.BIBTEX.splitlines():
        assert line.strip() in " ".join(readme.split()) or line.strip() in readme, line


def test_every_sample_named_for_the_examples_is_present(app):
    for name in app.SAMPLES:
        assert os.path.exists(f"samples/{name}.png")


def test_run_streams_one_update_per_level_then_the_clip(app, photo):
    yields = list(app.run(photo, 3, 0.5, 0.5))
    assert len(yields) == 5                       # entry level, 3 zoom steps, then the clip
    assert all(len(y) == 6 for y in yields)
    assert yields[0][1] is None                   # first yield clears any stale clip
    status, clip, gallery, compare, prompts, _scale = yields[-1]
    assert [label for _, label in gallery] == ["input", "4x", "16x", "64x"]
    assert compare is not None and len(compare) == 2
    assert prompts.count("tag") == 3
    try:
        assert os.path.getsize(clip) > 10_000
    finally:
        os.remove(clip)


def test_the_comparison_shows_the_deepest_level(app, photo):
    """Left is what the model started that step from, right is what it produced."""
    *_, (_, _, gallery, compare, _, scale) = list(app.run(photo, 2, 0.5, 0.5))
    assert compare[1] is gallery[-1][0]
    assert scale["choices"] == ["4x", "16x"] and scale["value"] == "16x"


def test_the_scale_picker_offers_one_option_per_step(app):
    assert app.scale_choices(1) == ["4x"]
    assert app.scale_choices(4) == ["4x", "16x", "64x", "256x"]
    assert app.scale_choices(app.MAX_STEPS)[-1] == "65536x"
    assert "input" not in app.scale_choices(app.MAX_STEPS)


def as_gallery_input(levels):
    """What Gradio actually hands a handler for a filepath Gallery: (path, caption) tuples.
    Passing dicts here instead is what hid a real bug: the picker silently did nothing."""
    return [(path, caption) for path, caption in levels]


def test_a_gallery_input_is_parsed_as_media_caption_tuples(app):
    got = app._gallery_paths(as_gallery_input(app.EXAMPLE_LEVELS))
    assert got == [(p, c) for p, c in app.EXAMPLE_LEVELS]


def test_switching_scale_rebuilds_the_pair_without_the_gpu(app):
    """Runs off the level images already on screen, so it must not touch zoom.zoom."""
    import numpy as np
    from PIL import Image
    left, right = app.pick_scale(as_gallery_input(app.EXAMPLE_LEVELS), 0.5, 0.5, "16x")
    assert left.size == right.size == (512, 512)
    expected = np.asarray(Image.open("samples/example_16x.png").convert("RGB"))
    assert np.array_equal(np.asarray(right), expected)


def test_every_offered_scale_can_be_switched_to(app):
    for _, caption in app.EXAMPLE_LEVELS[1:]:
        pair = app.pick_scale(as_gallery_input(app.EXAMPLE_LEVELS), 0.5, 0.5, caption)
        assert pair[0].size == pair[1].size == (512, 512), caption


def test_switching_to_a_scale_that_was_not_run_changes_nothing(app):
    import gradio as gr
    gallery = as_gallery_input(app.EXAMPLE_LEVELS)
    assert app.pick_scale(gallery, 0.5, 0.5, "65536x") == gr.skip()
    assert app.pick_scale(gallery, 0.5, 0.5, "input") == gr.skip()


def test_the_level_strip_wraps_to_two_rows_instead_of_scrolling(app):
    """Six levels or more will not fit one row at a readable size."""
    for steps in (1, 2, 3, 4):
        shape = app.gallery_shape(steps)
        assert shape["rows"] == 1 and shape["columns"] == steps + 1
    for steps in (5, 6, 7, 8):
        shape = app.gallery_shape(steps)
        items = steps + 1
        assert shape["rows"] == 2
        # every item has a cell, and no more than one cell goes spare
        assert shape["columns"] * 2 >= items > shape["columns"] * 2 - 2


def test_the_header_links_every_sibling_artifact(app):
    """A visitor who lands on the Space should be able to reach the rest from here."""
    for url in ["https://arxiv.org/abs/2609.06490",
                "https://github.com/dipta007/OracleZoom",
                "https://dipta007.github.io/OracleZoom/",
                "https://huggingface.co/dipta007/OracleZoom",
                "https://huggingface.co/datasets/dipta007/OracleZoom-4KLSDB-train",
                "https://huggingface.co/collections/dipta007/oraclezoom"]:
        assert url in app.HEADER, url


def test_dark_mode_is_forced(app):
    """The light theme washes the amber accents out against the photos."""
    assert "__theme" in app.FORCE_DARK and "dark" in app.FORCE_DARK
    assert app.demo.js is app.FORCE_DARK


def test_the_strip_never_pins_a_height(app):
    """A fixed height is wrong at some width: too short scrolls, too tall leaves a dead band.
    CSS hugs the content instead, so nothing here may set height."""
    assert all("height" not in app.gallery_shape(n) for n in range(1, app.MAX_STEPS + 1))
    assert "#levels" in app.CSS and "height:auto !important" in app.CSS


def test_a_deep_run_labels_every_level(app, photo):
    _, clip, gallery, _, _, scale = list(app.run(photo, app.MAX_STEPS, 0.5, 0.5))[-1]
    try:
        assert [label for _, label in gallery] == [
            "input", "4x", "16x", "64x", "256x", "1024x", "4096x", "16384x", "65536x"]
        assert scale["value"] == "65536x" and len(scale["choices"]) == app.MAX_STEPS
    finally:
        os.remove(clip)


def test_run_rejects_an_empty_upload(app):
    import gradio as gr
    with pytest.raises(gr.Error):
        next(app.run(None, 4, 0.5, 0.5))
