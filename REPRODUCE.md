# Reproduce our numbers

Every number in the paper comes from **one** setting. There is no sweep and no per-dataset tuning.
This page lists that setting and the exact commands.

## The settings

| What | Value |
|---|---|
| SR model | OSEDiff on SD3-medium, **frozen** |
| Prompt model | Qwen2.5-VL-3B-Instruct + Chain-of-Zoom LoRA, **frozen** |
| Prompt length | 32 tokens |
| Precision | fp32 |
| **Adapter** | LoRA on the SD3 transformer, **rank 16**, alpha 32, dropout 0 |
| Adapter targets | `to_q, to_k, to_v, add_q_proj, add_k_proj, add_v_proj` |
| Trained weights | **7.1M** (nothing else is trained) |
| Optimizer | AdamW, learning rate 5e-5, weight decay 1e-2 |
| Schedule | 500 warmup steps, then flat |
| Batch | 4 (one image at a time, gradients added up) |
| Stopping | stop after 8 checks with no gain, check every 100 steps |
| Seed | 0 |
| **A** 4x fidelity | weight 1.0 |
| **B** cycle coherence | weight 1.0, deep zoom factor 4 |
| **C** sharpness reward | weight **0.4**, reward model TOPIQ-NR on the 16x image |
| **D** KL leash | weight **8.0**, base = adapter turned off |
| **E** slow-copy teacher | weight 0.1, decay 0.95 |
| Training set | 1,000 curated images (4KLSDB) |
| Validation set | 2,000 held-out images |
| Supervision | 4x only, the one level with a real answer |
| Crop | 512 x 512 |
| Zoom steps at test | 4, giving 4x / 16x / 64x / 256x |

We picked `beta_kl` using the validation set, not the test set.

Training stops on its own, so the number of steps is not a setting. Our run stopped near **9,300
steps** (about 37 passes over the 1,000 images), in under a day on one GPU.

## 1. Get the data

```bash
export HF_TOKEN=hf_...        # or: huggingface-cli login
python scripts/prepare_data.py --tier 1k --out data/train_1k --eval_layout data/eval_val
```

The dataset ships each high-resolution image plus its cached 4x prompt. The low-quality input is
rebuilt locally, using the same geometry the zoom loop uses at test time: centre-crop and resize to
512, take the middle quarter, then bicubic back up to 512. Tiers `1k`, `4k`, `16k` and `all` are the
dataset's own subsets. The paper uses `1k`.

## 2. Train

```bash
scripts/train_final.sh data/train_1k data/train_1k/train_list.txt data/train_1k_val out/run1
```

The script passes all five weights on the command line, so you can read the setting instead of
trusting defaults. The defaults in the code match them too.

Add `--resume` to continue an interrupted run. It restores the optimizer, scheduler, step, epoch,
data order, random state, and slow-copy weights, so the run continues exactly.

## 3. Prepare the paper's test sets

```bash
export OPDZ_ROOT=/where/your/datasets/are
python scripts/prep_eval_datasets.py div8k_1238 flickr2k drealsr realsr ffhq
```

For each set this makes three things, which the scorer needs:

| File | What it is |
|---|---|
| `gt/` | the real images |
| `names.json` | `{"all": [name, ...]}` |
| `gt.txt` | one full path per line, used for 4x scoring |

The six test sets in the paper are 4KLSDB, DIV8K (1238 images), DRealSR, FFHQ, Flickr2K, and RealSR.

## 4. Score

```bash
# our model
scripts/eval_final.sh $OPDZ_ROOT/data/eval/div8k_1238 renders/ours results/ours

# Chain-of-Zoom, through the same pipeline
ARM=blind scripts/eval_final.sh $OPDZ_ROOT/data/eval/div8k_1238 renders/coz results/coz blind
```

You get one file per zoom level: `results/ours_s1.json` is 4x, `_s4.json` is 256x.

**Full-reference scores only exist at 4x.** Deeper there is no real image, so LPIPS and DISTS are
not computed and we make no fidelity claim there. This is built into the code, not a flag.

## 5. Baselines

The five comparison models run through the **same** zoom loop and the **same** scorer. See
[baselines/README.md](baselines/README.md).

## Things that change your numbers

**Pin your metric library.** `pyiqa` changed its internals between releases, including a known
NRQM/PI fix. If your version differs from ours, your numbers will differ for reasons that have
nothing to do with the model. Pin one version and write it down.

Table 1 uses CLIPIQA (no reference needed), plus LPIPS and DISTS at 4x. `full_eval` also computes a
wider set, but only these are needed to rebuild Table 1.

**Match the protocol.**

- 512x512 centre crop in, zoom by 4 each step, 4 steps.
- Every method runs as the SR model **inside the same zoom loop**. This compares the models, not
  the zoom code around them.
- Time per zoom step is the same at every level, because each step returns to 512x512.
- Speed: reading the prompt takes about 8 times longer than the super-resolution itself, and our
  adapter does not change that part. The adapter adds 0.019 s per step, about 1.3% of the time per
  image. Merging it into the model removes even that.
