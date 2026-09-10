"""OracleZoom demo: upload a photo, watch it zoom 4x and deeper."""
import gradio as gr
import spaces
from PIL import Image, ImageDraw

import geometry
import video
import zoom

MODELS = zoom.Models()

UPSCALE = 4
MAX_STEPS = 8
ACCENT = "#f5b942"
SAMPLES = ["0479", "0064", "0245", "0393", "0457"]

# Shipped output of a real run, so the page shows the payoff before anyone spends any quota.
EXAMPLE_INPUT = "samples/example_1x.png"
EXAMPLE_CLIP = "samples/example_zoom.mp4"
EXAMPLE_LEVELS = [(EXAMPLE_INPUT, "input")] + \
                 [(f"samples/example_{f}x.png", f"{f}x") for f in (4, 16, 64, 256)]
EXAMPLE_COMPARE = ("samples/example_256x_input.png", "samples/example_256x.png")
EXAMPLE_NOTE = "_An example run. Upload a photo above to make your own._"

HEADER = """
<div style="text-align:center;max-width:820px;margin:0 auto 4px">
  <h1 style="margin:0;font-size:2.1em;letter-spacing:-.02em">OracleZoom 🔎</h1>
  <p style="margin:.5em 0 .9em;font-size:1.08em;line-height:1.5;opacity:.85">
    Zoom into any photo far past what it holds. Four steps of 4x take you to
    <b>256x</b>, each one drawn from the last. Push it to eight steps if you want.
  </p>
  <p style="margin:0;font-size:.95em">
    <a href="https://arxiv.org/abs/2609.06490">Paper</a> &nbsp;·&nbsp;
    <a href="https://github.com/dipta007/OracleZoom">Code</a> &nbsp;·&nbsp;
    <a href="https://dipta007.github.io/OracleZoom/">Project page</a> &nbsp;·&nbsp;
    <a href="https://huggingface.co/dipta007/OracleZoom">Model</a>
  </p>
</div>
"""

BIBTEX = """@misc{dipta2026oraclezoomonpolicyselfdistillationinspired,
  title         = {OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution},
  author        = {Shubhashis Roy Dipta and Sourajit Saha and Shaswati Saha and Nobin Sarwar},
  year          = {2026},
  eprint        = {2609.06490},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.06490}
}"""

# Dark only. Gradio otherwise follows the visitor's system setting, and the light theme washes
# the amber accents out against the photos. One redirect on first load, then it sticks.
FORCE_DARK = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.href);
  }
}
"""

CSS = """
#hero video {border-radius:12px}
.contain {max-width:1400px !important}
footer {display:none !important}
#zoom-target img {cursor:crosshair}
/* Let the level strip size itself. Any fixed height is wrong at some window width or depth:
   too short and it scrolls, too tall and it leaves a dead band. .gallery-container is the one
   that actually carries the height; styling only the outer block leaves the band behind. */
#levels, #levels .gallery-container, #levels .grid-wrap {
  height:auto !important; max-height:none !important; min-height:0 !important;
  overflow:visible !important; flex-grow:0 !important}
"""


def label(factor):
    return "input" if factor == 1 else f"{factor}x"


def preview(image, levels, cx, cy):
    """Show where the zoom will go, so nobody spends a run to find out.

    Falls back to the shipped example when nothing is uploaded yet. Returning None here instead
    would blank the box the moment a visitor touched any slider, taking the one thing that
    explains what the amber boxes mean with it.
    """
    canvas = geometry.resize_and_center_crop(
        image if image is not None else Image.open(EXAMPLE_INPUT)).convert("RGB")
    rects = geometry.nested_rects(canvas.size, int(levels), UPSCALE, (cx, cy))
    dimmed = Image.blend(canvas, Image.new("RGB", canvas.size, (0, 0, 0)), 0.45)
    dimmed.paste(canvas.crop(rects[0]), rects[0][:2])
    draw = ImageDraw.Draw(dimmed)
    for rect in rects:
        if rect[2] - rect[0] >= 3:
            draw.rectangle(rect, outline=ACCENT, width=2)
    return dimmed


def pick_point(evt: gr.SelectData):
    x, y = evt.index
    # rounded to the sliders' own step, so a click does not read 0.427 on a 0.01 slider
    return (round(x / geometry.PROCESS_SIZE, 2), round(y / geometry.PROCESS_SIZE, 2))


def _gallery_paths(gallery):
    """Gradio hands a Gallery back to a handler as (media, caption) tuples, media being a path
    because the component is filepath-typed. Not dicts, which is what the payload looks like."""
    return [(item[0], item[1] or "") for item in gallery or []
            if isinstance(item, (tuple, list)) and len(item) == 2]


def pick_scale(gallery, cx, cy, choice):
    """Rebuild the comparison at whichever scale was asked for.

    The left side is the plain enlargement that fed that step, which is the previous level
    cropped and scaled back up. Recomputing it from the levels already on screen keeps this
    on the CPU, so switching scales costs no GPU quota.
    """
    levels = _gallery_paths(gallery)
    labels = [lbl for _, lbl in levels]
    if choice not in labels:
        return gr.skip()
    i = labels.index(choice)
    if i == 0:
        return gr.skip()
    prev = Image.open(levels[i - 1][0]).convert("RGB")
    blurry = geometry.zoom_window(prev, UPSCALE, (cx, cy)).resize(prev.size, Image.BICUBIC)
    return blurry, Image.open(levels[i][0]).convert("RGB")


def scale_choices(levels):
    return [label(UPSCALE ** n) for n in range(1, int(levels) + 1)]


CLIP_H = 480          # tuned so the right column ends level with the Zoom in button


def gallery_shape(levels):
    """One row up to five levels, two beyond. Height is left to CSS, which hugs the content."""
    items = int(levels) + 1
    rows = 1 if items <= 5 else 2
    return gr.update(columns=-(-items // rows), rows=rows)


def estimate_duration(image, levels, cx, cy):
    # Checked against the visitor's remaining quota BEFORE the run, so a loose number locks out
    # low-quota visitors and a tight one gets the run killed mid-way. Measured warm on ZeroGPU:
    # 6.4s fixed, 3.2s per level. Near double that, to survive a cold worker streaming weights.
    return int(12 + 6 * int(levels))


def _stream(image, levels, cx, cy):
    if image is None:
        raise gr.Error("Upload a photo first.")
    levels = int(levels)
    gallery, prompts, frames, compare = [], [], [], None
    for step, factor, prompt, blurry, result in zoom.zoom(MODELS, image, levels, UPSCALE, (cx, cy)):
        frames.append(result)
        gallery.append((result, label(factor)))
        if step:
            compare = (blurry, result)
            prompts.append(f"**{label(factor)}** &nbsp; {prompt or '_(no prompt)_'}")
        left = levels - step
        if not step:
            status = f"Warmed up. Zooming {levels} step{'s' if levels > 1 else ''}…"
        elif left:
            status = f"At **{label(factor)}**, {left} step{'s' if left > 1 else ''} to go…"
        else:
            status = f"At **{label(factor)}**. Rendering the clip…"
        # First yield clears any clip left from the previous run; later ones leave it alone.
        yield (status, (None if not step else gr.skip()), gallery, compare,
               "\n\n".join(prompts), gr.skip())
    yield ("Done. 🔎", video.render(frames, UPSCALE, (cx, cy)), gallery, compare,
           "\n\n".join(prompts),
           gr.update(choices=scale_choices(levels), value=label(UPSCALE ** levels)))


@spaces.GPU(duration=estimate_duration)
def run(image, levels, cx, cy):
    yield from _stream(image, levels, cx, cy)


@spaces.GPU(duration=estimate_duration(None, 4, 0, 0))
def run_example(image):
    """One input, so the examples render as thumbnails instead of a four-column table.
    Its results are cached, which is how a visitor with no quota left still sees output."""
    yield from _stream(image, 4, 0.5, 0.5)


def _example_scale_options():
    """Keep the picker's starting options honest about the shipped example's depth."""
    return scale_choices(len(EXAMPLE_LEVELS) - 1)


with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber"), css=CSS, js=FORCE_DARK,
               title="OracleZoom: zoom past 256x") as demo:
    gr.HTML(HEADER)
    # Inputs down the left, results down the right. Both columns carry real content the whole
    # way, which is what stops one of them stretching a component into an empty band.
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            image = gr.Image(label="Your photo", type="pil", height=300,
                             sources=["upload", "clipboard"])
            # Inputs only, no fn: the outputs it would fill live further down the page, and
            # Gradio can only wire components that already exist.
            gr.Examples(examples=[f"samples/{n}.png" for n in SAMPLES], inputs=[image],
                        label="Or try any of these")
            # Seeded with the example so the box overlay explains itself before any upload.
            target = gr.Image(value=preview(None, 4, 0.5, 0.5), elem_id="zoom-target",
                              label="Click to move the zoom point", type="pil",
                              interactive=False, height=340, show_download_button=False)
            with gr.Row():
                cx = gr.Slider(0, 1, value=0.5, step=0.01, label="Horizontal", elem_id="zoom-x")
                cy = gr.Slider(0, 1, value=0.5, step=0.01, label="Vertical", elem_id="zoom-y")
            levels = gr.Slider(1, MAX_STEPS, value=4, step=1, label="Zoom steps",
                               info=f"Each step is {UPSCALE}x. 4 steps reach 256x, "
                                    f"{MAX_STEPS} reach {UPSCALE ** MAX_STEPS}x. "
                                    f"Fewer steps finish sooner.")
            go = gr.Button("🔎 Zoom in", variant="primary", size="lg")
            status = gr.Markdown("Showing an example. Upload a photo, then press **Zoom in**.")
            gr.Markdown("##### Vision Language Model at each step:")
            prompts = gr.Markdown(EXAMPLE_NOTE)
        with gr.Column(scale=6):
            clip = gr.Video(value=EXAMPLE_CLIP, label="The zoom", elem_id="hero", autoplay=True,
                            loop=True, show_share_button=True, height=CLIP_H)
            gallery = gr.Gallery(value=EXAMPLE_LEVELS, label="Every level, start to finish",
                                 elem_id="levels", object_fit="cover", show_download_button=True,
                                 **{k: v for k, v in gallery_shape(len(EXAMPLE_LEVELS) - 1).items()
                                    if k != "__type__"})
            scale = gr.Radio(_example_scale_options(), value=EXAMPLE_LEVELS[-1][1],
                             label="Compare at",
                             info="Left is the plain enlargement that step began from, "
                                  "right is what OracleZoom drew. Switching is free.")
            compare = gr.ImageSlider(value=EXAMPLE_COMPARE, height=470, elem_id="compare",
                                     show_label=False)

    controls = [image, levels, cx, cy]
    for c in controls:
        c.change(preview, controls, target, show_api=False)
    target.select(pick_point, None, [cx, cy], show_api=False)
    # Reshape before the run, not after, so the strip does not resize under the results.
    levels.change(gallery_shape, levels, gallery, show_api=False)
    go.click(run, controls, [status, clip, gallery, compare, prompts, scale])
    scale.change(pick_scale, [gallery, cx, cy, scale], compare, show_api=False)

    gr.Markdown("### Cite our work")
    gr.Code(value=BIBTEX, language=None, show_label=False, container=False)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
