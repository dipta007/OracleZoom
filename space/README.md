---
title: OracleZoom
emoji: 🔎
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12.12"
app_file: app.py
startup_duration_timeout: 45m
pinned: true
license: mit
short_description: Zoom any photo to 256x, one 4x step at a time
tags:
  - super-resolution
  - image-to-image
  - diffusion
  - zoom
models:
  - dipta007/OracleZoom
  - stabilityai/stable-diffusion-3-medium-diffusers
  - Qwen/Qwen2.5-VL-3B-Instruct
datasets:
  - dipta007/OracleZoom-4KLSDB-train
---

# OracleZoom

Upload a photo, pick a point, and watch it zoom to 256x. Four steps of 4x, each one drawn
from the last.

- Paper: https://arxiv.org/abs/2609.06490
- Code: https://github.com/dipta007/OracleZoom
- Project page: https://dipta007.github.io/OracleZoom/
- Model: https://huggingface.co/dipta007/OracleZoom
- Training data: https://huggingface.co/datasets/dipta007/OracleZoom-4KLSDB-train
- Collection: https://huggingface.co/collections/dipta007/oraclezoom

## Running it yourself

The Space needs an `HF_TOKEN` secret. Stable Diffusion 3-medium is gated, so the token has
to come from an account that has accepted its licence. Nothing in the code reads the token
directly; `huggingface_hub` picks up `HF_TOKEN` on its own.

Hardware: ZeroGPU. The pipeline holds about 27 GB of weights, well inside the 48 GB that a
`large` slice gives. ZeroGPU is Gradio-only, so a Docker Space cannot host this.

## What is in here

`app.py` is the interface. `zoom.py` is the recursion, flattened from the research repo so it
runs in memory with no disk round-trips and lets the zoom window sit off centre. `video.py`
renders the clip. `vendor/` holds four files copied from
[Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom) (MIT, see `vendor/COZ-LICENSE`),
because Spaces do not support git submodules. The only edit to them pins `weights_only=True`
on three `torch.load` calls, whose default flipped in torch 2.6.

## Honest limits

Past the first step or two there is no ground truth left to recover, so the deep levels are
plausible detail rather than measured detail. The zoom is a crop of your photo, not a real
lens moving closer. The paper says which claims we do and do not make.
