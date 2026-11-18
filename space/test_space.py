"""Tests for the parts of the Space that need no GPU: crop geometry and the video render.

    uv run --with pytest --with pillow --with numpy --with imageio --with imageio-ffmpeg \
        pytest space/test_space.py -q
"""
import os

import numpy as np
import pytest
from PIL import Image

import geometry
import video


def _img(seed, size=512):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_entry_crop_is_square_and_centred():
    out = geometry.resize_and_center_crop(_img(0, 900).resize((1200, 900)))
    assert out.size == (512, 512)


def test_window_is_one_quarter_wide_and_centred_by_default():
    assert geometry.window_rect((512, 512), 4) == (192, 192, 320, 320)


def test_window_stays_inside_the_frame_at_the_edges():
    for center in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        l, t, r, b = geometry.window_rect((512, 512), 4, center)
        assert 0 <= l < r <= 512 and 0 <= t < b <= 512
        assert (r - l, b - t) == (128, 128)


def test_nested_rects_shrink_by_upscale_and_nest():
    rects = geometry.nested_rects((512, 512), 4, 4)
    assert [r[2] - r[0] for r in rects] == [128, 32, 8, 2]
    for inner, outer in zip(rects[1:], rects):
        assert outer[0] <= inner[0] and inner[2] <= outer[2]
        assert outer[1] <= inner[1] and inner[3] <= outer[3]


def test_nested_rects_stop_before_a_window_goes_sub_pixel():
    """Eight steps is offered in the UI, but a 512 canvas only holds four boxes."""
    rects = geometry.nested_rects((512, 512), 8, 4)
    assert [r[2] - r[0] for r in rects] == [128, 32, 8, 2]
    assert all(r[2] > r[0] and r[3] > r[1] for r in rects)


def test_zoom_window_matches_window_rect():
    img = _img(1)
    assert geometry.zoom_window(img, 4).size == (128, 128)
    assert np.array_equal(np.asarray(geometry.zoom_window(img, 4)),
                          np.asarray(img.crop(geometry.window_rect(img.size, 4))))


def test_video_needs_two_levels():
    assert video.render([_img(2)]) is None


def test_video_renders_an_mp4_with_the_expected_frame_count():
    levels = [_img(i) for i in range(5)]
    path = video.render(levels, 4, (0.5, 0.5))
    try:
        assert os.path.getsize(path) > 10_000
        import imageio.v2 as imageio
        frames = imageio.mimread(path, memtest=False)
        expected = 4 * (video.PUSH_FRAMES + video.FADE_FRAMES) + video.TAIL_FRAMES
        assert len(frames) == expected
        assert frames[0].shape[:2] == (512, 512)
    finally:
        os.remove(path)


def test_the_push_ends_exactly_on_the_next_level_input():
    """The last push frame must equal the blurry crop the next level was built from.
    If it does not, the clip jumps at every transition."""
    levels = [_img(3), _img(4)]
    frames = list(video._frames(levels, 4, (0.5, 0.5)))
    last_push = np.asarray(frames[video.PUSH_FRAMES - 1], dtype=np.int16)
    target = geometry.window_rect(levels[0].size, 4, (0.5, 0.5))
    blurry = np.asarray(levels[0].crop(target).resize(levels[0].size, Image.BICUBIC),
                        dtype=np.int16)
    assert np.array_equal(last_push, blurry)


def test_the_fade_ends_on_the_sharp_level():
    levels = [_img(5), _img(6)]
    frames = list(video._frames(levels, 4, (0.5, 0.5)))
    end_of_fade = np.asarray(frames[video.PUSH_FRAMES + video.FADE_FRAMES - 1], dtype=np.int16)
    assert np.abs(end_of_fade - np.asarray(levels[1], dtype=np.int16)).max() <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
