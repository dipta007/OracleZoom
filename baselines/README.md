# Baselines

We do **not** copy the numbers other papers report. We run their models ourselves, inside our zoom
loop, and score them with our code. Same test images, same crops, same metrics. That is the only way
the comparison is fair.

## The five models

| Model | Type |
|---|---|
| SwinIR | regression |
| HiT-SR | regression |
| MambaIR | regression |
| SeeSR | diffusion |
| OSEDiff | diffusion |

Each one replaces the super-resolution step inside our zoom loop. Nothing else changes.

## How they are run

Each zoom step: cut the middle quarter (128x128), run the model's own 4x super-resolution, resize to
512, then zoom again. The crop geometry is identical to our own model's. **No prompt model and no
privileged information** are used. This is the fair "no OracleZoom" comparison at 4x, 16x, 64x
and 256x.

Expect the three regression models to look blurry at deep zoom. They cannot invent detail, so they
smooth instead. That is a real finding, not a bug.

## Commands

```bash
# 1. render all four zoom levels with a baseline model
python baselines/recursive_render.py --model swinir --ckpt <ckpt_dir> \
  --gt_dir <dataset>/gt --out renders/swinir

# 2. score it with the same scorer we use for our own model
python -m opd_zoom.eval.full_eval --render_dir renders/swinir \
  --names_json <dataset>/names.json:all --gt_list <dataset>/gt.txt \
  --arm swinir --method baseline --rank 0 --evalset <name> \
  --out results/swinir_s1.json --full_fr --only_scale 1
```

Repeat step 2 with `--only_scale 2`, `3`, and `4` for the deeper zoom levels.

## Getting the model code

Each model needs its own official code. Put the clones in one folder and point to it:

```bash
export OPDZ_REPOS=/path/to/upstream_repos
```

The adapter looks there first, then falls back to `baselines/repos/upstream`.

| Model | Code from |
|---|---|
| SwinIR, HiT-SR, MambaIR | their own repositories (BasicSR style) |
| SeeSR | the SeeSR repository |
| OSEDiff | the original OSEDiff repository (SD2.1 version) |

Some of these need their own Python environment, because their dependencies conflict with each
other. `recursive_render.py` runs each model in a separate process for this reason.

## Adding another model

Write `adapters/<name>.py` with a class following `adapters/base.SRAdapter`:

- `__init__(ckpt_dir, device)` loads the model.
- `sr_x4(image) -> image` does one 4x super-resolution on a PIL image. No prompt.

Nothing in this folder imports from or edits `src/opd_zoom/`. It only calls our scorer, read-only.
