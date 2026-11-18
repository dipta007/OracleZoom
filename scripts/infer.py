#!/usr/bin/env python3
"""Zoom an image (or a folder of images) 4x -> 16x -> 64x -> 256x with OracleZoom.

One command, no manual checkpoint shuffling:

    uv run scripts/infer.py --input photo.jpg --output out/
    uv run scripts/infer.py --input my_images/ --output out/

It downloads the weights from Hugging Face on first use if they are not already local. That one
download carries both our merged transformer AND the three Chain-of-Zoom checkpoints the pipeline
needs (SR_LoRA, SR_VAE, VLM_LoRA), so there is nothing else to fetch by hand.

Stable Diffusion 3-medium is gated: accept its licence once on Hugging Face and run `uv run hf auth login`.
It and Qwen2.5-VL-3B download themselves on the first run.

The released weights are a MERGED transformer, not a LoRA adapter, which is why this passes
--full_transformer. Passing --pld_lora instead fails with "Can't find 'adapter_config.json'".
"""
import argparse
import os
import subprocess
import sys

REPO = "dipta007/OracleZoom"
MERGED = "merged_transformer.safetensors"


def resolve_weights(weights, allow_download):
    """Return a local dir holding MERGED + ckpt/, downloading it if needed."""
    if weights and os.path.isfile(os.path.join(weights, MERGED)):
        return weights
    if not allow_download:
        sys.exit(f"error: {MERGED} not found under {weights!r}. "
                 f"Drop --no-download to fetch it from {REPO}.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("error: huggingface_hub is missing. Install it, or pass --weights "
                 "pointing at an existing download.")
    print(f"[infer] fetching {REPO} (~8 GB, once) ...", flush=True)
    return snapshot_download(repo_id=REPO, local_dir=weights or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="an image, or a folder of images")
    ap.add_argument("--output", default="out", help="where the zoom levels are written")
    ap.add_argument("--weights", default="weights/oraclezoom",
                    help="local weights dir; downloaded here if absent")
    ap.add_argument("--rec_num", type=int, default=4, help="zoom steps (4 -> up to 256x)")
    ap.add_argument("--upscale", type=int, default=4, help="magnification per step")
    ap.add_argument("--no-download", action="store_true", help="fail instead of fetching weights")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    if not os.path.isfile(args.input) and not os.path.isdir(args.input):
        ap.error(f"input does not exist: {args.input}")
    if os.path.isfile(args.input) and os.path.splitext(args.input)[1].lower() not in (".png", ".jpg", ".jpeg"):
        ap.error("input must be a PNG or JPEG image")
    w = resolve_weights(args.weights, not args.no_download)

    cmd = [sys.executable, "-m", "opd_zoom.teacher.oracle_infer",
           "--mode", "student",
           "--full_transformer", os.path.join(w, MERGED),
           "--coz_ckpt", os.path.join(w, "ckpt"),
           "--gt_dir", args.input,
           "--out", args.output,
           "--rec_num", str(args.rec_num),
           "--upscale", str(args.upscale)]

    print("[infer] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
