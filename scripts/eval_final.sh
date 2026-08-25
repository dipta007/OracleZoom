#!/usr/bin/env bash
# Score one arm on one eval set, exactly as the paper does it.
#
#   scripts/eval_final.sh <dataset_root> <render_out> <json_prefix> [lora_path]
#
# dataset_root must be a prepped set, as produced by scripts/prep_eval_datasets.py:
#     <dataset_root>/gt/          the ground-truth images
#     <dataset_root>/names.json   {"all": [name, ...]}
#     <dataset_root>/gt.txt       one absolute GT path per line (for 4x full-reference)
#
# Pass "blind" as lora_path to score the CoZ baseline instead of our model.
#
# Two stages, matching the reported harness:
#   1. render      4 recursions -> 4x / 16x / 64x / 256x
#   2. full_eval   one JSON per scale. Full-reference only at 4x, because no
#                  ground truth exists deeper. That limit is the harness's, not
#                  a flag we chose per run.
set -euo pipefail

ROOT=${1:?usage: scripts/eval_final.sh <dataset_root> <render_out> <json_prefix> [lora_path]}
RENDER=${2:?}
PREFIX=${3:?}
LORA=${4:-${ORACLEZOOM_LORA:-release/oraclezoom}}
COZ_CKPT=${COZ_CKPT:-ref/coz/ckpt}
GPU=${GPU:-0}
ARM=${ARM:-ours}
EVALSET=${EVALSET:-$(basename "$ROOT")}

for f in "$ROOT/gt" "$ROOT/names.json" "$ROOT/gt.txt"; do
  [ -e "$f" ] || { echo "missing $f -- run scripts/prep_eval_datasets.py first" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=$GPU
LARGS=""
[ "$LORA" = "blind" ] || LARGS="--pld_lora $LORA"

# --- 1. render -------------------------------------------------------------
python -u -m opd_zoom.teacher.oracle_infer --mode student $LARGS \
  --gt_dir "$ROOT/gt" --out "$RENDER" --coz_ckpt "$COZ_CKPT" --rec_num 4 --upscale 4

# --- 2. metric panel, one JSON per scale ---------------------------------
for S in 1 2 3 4; do
  python -u -m opd_zoom.eval.full_eval \
    --render_dir "$RENDER" \
    --names_json "$ROOT/names.json:all" \
    --gt_list "$ROOT/gt.txt" \
    --arm "$ARM" --method oraclezoom --rank 16 --evalset "$EVALSET" \
    --out "${PREFIX}_s${S}.json" \
    --full_fr --skip_metrics topiq_fr --only_scale "$S"
done

echo "scores -> ${PREFIX}_s1..s4.json   (s1 = 4x, s4 = 256x)"
