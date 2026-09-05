# Reproduce our numbers

This page gives the final model configuration and commands for training, recursive rendering, and image-quality scoring. Ablations vary individual settings. Projected-reference and VLM-judge results require separate evaluation; the commands below do not reproduce those results.

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
| **B** cross-scale consistency | weight 1.0, project 16x to the aligned 4x ground-truth region |
| **C** quality guidance | weight **0.4**, reward model TOPIQ-NR on the 16x image |
| **D** KL prior | weight **8.0**, base = adapter turned off |
| **E** EMA consistency | weight 0.1, decay 0.95 |
| Training set | 1,000 curated images (4KLSDB) |
| Validation set | 2,000 held-out images |
| Supervision | Direct targets at 4x; projected ground-truth reference at 16x |
| Crop | 512 x 512 |
| Zoom steps at test | 4, giving 4x / 16x / 64x / 256x |

We picked `beta_kl` using the validation set, not the test set.

Training stops on its own, so the number of steps is not a setting. Our run stopped near **9,300 steps** (about 37 passes over the 1,000 images), in under a day on one GPU.

## 1. Get the data

```bash
uv run scripts/prepare_data.py --tier 1k --out data/train_1k --eval_layout data/eval_val
```

The dataset ships each high-resolution image plus its cached 4x prompt. The low-quality input is rebuilt locally, using the same geometry the zoom loop uses at test time: centre-crop and resize to 512, take the middle quarter, then bicubic back up to 512. Tiers `1k`, `4k`, `16k` and `all` are the dataset's own subsets. The paper uses `1k`.

## 2. Train

```bash
scripts/train_final.sh data/train_1k data/train_1k/train_list.txt data/train_1k_val out/run1
```

The script passes all five weights on the command line, so you can read the setting instead of trusting defaults. The defaults in the code match them too.

Add `--resume out/run1/ckpt.pt` to continue an interrupted run. It restores the optimizer, scheduler, step, epoch, data order, random state, and slow-copy weights, so the run continues exactly.

## 3. Prepare the paper's test sets

```bash
export OPDZ_ROOT=/where/your/datasets/are
uv run scripts/prep_eval_datasets.py div8k_1238 flickr2k drealsr realsr ffhq
```

For each set this makes three things, which the scorer needs:

| File | What it is |
|---|---|
| `gt/` | the real images |
| `names.json` | `{"all": [name, ...]}` |
| `gt.txt` | one full path per line, used for 4x scoring |

The seven test sets in the paper are 4KLSDB, DIV2K, DIV8K (1238 images), DRealSR, FFHQ, Flickr2K, and RealSR.

## 4. Score

```bash
# our model
scripts/eval_final.sh $OPDZ_ROOT/data/eval/div8k_1238 renders/ours results/ours

# Chain-of-Zoom, through the same pipeline
ARM=blind scripts/eval_final.sh $OPDZ_ROOT/data/eval/div8k_1238 renders/coz results/coz blind
```

You get one file per zoom level: `results/ours_s1.json` is 4x, `_s4.json` is 256x.

The scorer computes direct-reference LPIPS and DISTS only at 4x. At deeper scales, the paper separately evaluates projected-reference fidelity and consistency with earlier zooms.

## 5. Baselines

The five comparison models use the same zoom geometry and scorer, but do not use the VLM prompter shared by OracleZoom and Chain-of-Zoom. See [baselines/README.md](baselines/README.md).

## Things that change your numbers

**Pin your metric library.** `pyiqa` changed its internals between releases, including a known NRQM/PI fix. If your version differs, your numbers will differ for reasons that have nothing to do with the model. This repo pins **`pyiqa==0.1.15`**, which we verified end to end on a clean machine.

We did not record the exact `pyiqa` patch release used to produce the published numbers, and the environment that produced them no longer exists. Expect the ranking and the size of the gaps to reproduce; expect the third decimal of individual NR-IQA values to move a little. `full_eval` now writes the installed version into every scores JSON so this cannot happen to you.

Table 1 uses CLIPIQA (no reference needed), plus LPIPS and DISTS at 4x. `full_eval` also computes a wider set, but only these are needed to rebuild Table 1.

**Match the protocol.**

- 512x512 centre crop in, zoom by 4 each step, 4 steps.
- Every method uses the same crop-and-zoom geometry and scorer. The baseline SR pipelines have different conditioning; see the baseline instructions above.
- Each zoom step returns to 512x512; runtime also depends on prompt generation and hardware.
- The adapter adds 0.019 s to the SR stage per recursive step on the measured H100 setup. Merging it reduces that SR-stage gap to 0.001 s. Prompt generation accounts for most of the total runtime.
