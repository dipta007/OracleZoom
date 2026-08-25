#!/usr/bin/env bash
# Train the reported OracleZoom model (the locked config from the paper).
#
#   scripts/train_final.sh <train_cache> <train_list> <val_cache> <out_dir>
#
# The five loss weights below are the reported configuration. They are written
# out explicitly rather than hidden in a default, so the run is self-documenting:
#   4x fidelity      w_4x        = 1.0   (trainer default)
#   cycle coherence  w_deep      = 1.0
#   sharpness reward beta_reward = 0.4   (TOPIQ-NR on the 16x output)
#   KL leash         beta_kl     = 8.0
#   EMA self-teacher lambda_ema  = 0.1   (decay 0.95, trainer default)
# LoRA rank 16 on the SD3 transformer (7.1M trainable parameters).
#
# Training stops early on held-out validation loss, so there is no fixed step
# count. Our reported run stopped near 9.3k steps (~37 epochs over the 1k tier).
set -euo pipefail

CACHE=${1:?usage: scripts/train_final.sh <train_cache> <train_list> <val_cache> <out_dir>}
LIST=${2:?}
VAL=${3:?}
OUT=${4:?}
shift 4          # anything left over is passed through to the trainer (e.g. --resume)

GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU python -u -m opd_zoom.distill.train_pld_b3_resumable \
  --cache "$CACHE" --pilot_list "$LIST" --val_cache "$VAL" --out "$OUT" \
  --lora_rank 16 \
  --w_deep 1.0 --full_grad \
  --beta_reward 0.4 --reward_metric topiq_nr \
  --beta_kl 8.0 \
  --lambda_ema 0.1 \
  --eval_every 100 --eval_batch 16 \
  --patience 8 --warmup_evals 2 \
  --ckpt_every 200 \
  "$@"

# The trainer is fully resumable (optimizer + step + RNG + data order + EMA).
# Re-run the same command with --resume to continue an interrupted run.
