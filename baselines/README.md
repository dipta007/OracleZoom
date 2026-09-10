# Baselines

We do **not** copy the numbers other papers report. We run their models ourselves, inside our zoom loop, and score them with our code. Same test images, same crops, same metrics. That is the only way the comparison is fair.

## The five models

| Model | Type |
|---|---|
| SwinIR | regression |
| HiT-SR | regression |
| MambaIR | regression |
| SeeSR | diffusion |
| OSEDiff | diffusion |

Each model uses the same crop-and-zoom geometry. OracleZoom and Chain-of-Zoom share a VLM prompter; the five baselines below use their native SR pipelines without that shared prompter.

## How they are run

At each step, crop the middle quarter (128x128), run the model's native 4x super-resolution, and resize the output to 512x512 before the next zoom. The renderer does not supply VLM prompts or privileged ground-truth information to these baselines. Native conditioning inside each adapter still applies.

## Commands

```bash
# 1. render all four zoom levels with a baseline model
uv run baselines/recursive_render.py --model swinir --ckpt <ckpt_dir> \
  --gt_dir <dataset>/gt --out renders/swinir

# 2. score it with the same scorer we use for our own model
uv run -m opd_zoom.eval.full_eval --render_dir renders/swinir \
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

Some models need their own Python environment because their dependencies conflict. Run each rendering command in the environment prepared for that model. The renderer loads one selected adapter in its own process.

## Adding another model

Write `adapters/<name>.py` with a class following `adapters/base.SRAdapter`:

- `__init__(ckpt_dir, device)` loads the model.
- `sr_x4(image) -> image` does one 4x super-resolution on a PIL image. No prompt.

Nothing in this folder imports from or edits `src/opd_zoom/`. It only calls our scorer, read-only.
