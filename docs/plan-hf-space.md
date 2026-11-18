# Plan: OracleZoom demo on Hugging Face Spaces

Goal: a stranger lands on a URL, uploads a photo, clicks one button, and watches it zoom 4x to 256x. No install, no login, no GPU of their own.

Status: **live and public.** https://huggingface.co/spaces/dipta007/OracleZoom, ZeroGPU,
`RUNNING`, dark mode forced. Source of truth is `space/` in this repo; the Space is a copy
pushed there. Written Sept 10, 2026.

Measured on the live Space, warm worker: **6.4 s fixed, 3.2 s per level.** So 9.6 s for one
level, 19.4 s for four, 40.4 s for eight. `estimate_duration` declares `12 + 6 * levels`, which
leaves about 20 s of headroom at the deepest setting for a cold worker streaming weights. The
startup log also settled the disk question: `Cleaned 17.39GB of tensor files ... after packing`.

The demo really does run our model, checked against the live endpoint rather than assumed: the
4x output carries **16.20** of high-frequency energy where a plain bicubic enlargement of the
same crop carries **4.14**, and the prompts come back specific ("red panda, face, whiskers").
Worth knowing because `space/dev_ui.py` stubs the model out with bicubic for layout work, and
its blurry output reads exactly like a broken model. It now says so in a red banner.

The merged transformer loads with **0 missing and 0 unexpected keys**, which is the proof that
the demo runs our model and not the base one.

Scope cut, agreed Sept 10: **show our zoom only.** No baseline comparison against Chain-of-Zoom
anywhere in the demo. The before-and-after slider compares our result against a plain bicubic
enlargement, which is the honest floor, not against another method.

---

## 1. The three choices, decided

**SDK: Gradio. Hardware: ZeroGPU. Template: Blank.**

| Choice | Verdict | Why |
|---|---|---|
| Static | No | Cannot run Python or a GPU. Only good for a copy of the project page. |
| Docker | No | Docker Spaces **cannot use ZeroGPU**. A Docker Space needs paid GPU hardware. |
| **Gradio** | **Yes** | ZeroGPU is Gradio-only. Also gives us upload, gallery, video, streaming for free. |
| CPU Basic | No | A 2B diffusion model plus a 3B vision language model on 2 vCPU would take many minutes per zoom step, if it ran at all. |
| **ZeroGPU** | **Yes** | Free. Half an NVIDIA RTX Pro 6000 Blackwell, 48 GB VRAM. Our pipeline needs about 27 GB. |
| Custom Hardware | Fallback only | 1x L4 costs $0.80 per hour, about $19 per day. Only if ZeroGPU blocks us. |

Gradio template: pick **Blank**, not `text-to-image`. Our layout is custom.

### Eligibility, worth checking before anything else

Creating a Gradio Space normally needs a paid plan. There is one exception that covers us: a **free personal account in good standing can host 2 ZeroGPU Spaces**. Good standing means a verified email and an account older than 30 days. A PRO account raises the cap to 10 Spaces.

Source: https://huggingface.co/docs/hub/spaces-zerogpu

---

## 2. What the visitor sees

One page, two columns, plus a second tab.

**Left column, the input**
1. Drag-and-drop image box. Any photo works.
2. A live preview that draws the four nested zoom boxes on the photo, so the visitor sees exactly what will happen before spending anything.
3. Click the preview to move the zoom point. Defaults to the center, which is the paper's protocol.
4. A slider for zoom depth: 1 step (4x) up to 4 steps (256x). Default 4.
5. One big button: **Zoom in**.

**Right column, the payoff**
1. **A zoom video that plays and loops on its own.** It dollies smoothly from the full photo down into the 256x crop. This is the thing people screenshot and share. It is built on the CPU from the four output images after the GPU work is done, so it costs no GPU quota.
2. A 2x2 gallery of the four zoom levels, each labelled 4x, 16x, 64x, 256x.
3. **A before-and-after slider per level**, using `gr.ImageSlider`. Left side is the plain bicubic upscale of the crop, right side is our result. The pipeline already computes that bicubic image as the SR input, so this is free and it shows exactly what the model added. `gr.ImageSlider` is native since Gradio 5.27.0.
4. The prompt our model wrote at each step, shown as text. This is the paper's actual contribution, so it should be visible, not hidden.

**Results appear one level at a time.** The handler yields after each zoom step. The visitor sees the 4x result in about 5 seconds instead of staring at a spinner for 20.

**Second tab: Gallery.** Precomputed results, including the Chain-of-Zoom versus OracleZoom side-by-side figures we already have in `docs/static/images/`. Costs zero GPU. This matters more than it sounds, see the quota section.

**Header:** title, one-line pitch, and buttons to the paper (arXiv 2609.06490), the code, the project page, and the model.

---

## 3. The quota reality, and why it shapes the design

ZeroGPU daily quota per visitor:

| Visitor | Daily GPU seconds |
|---|---|
| Not logged in | 120 |
| Free HF account | 300 |
| PRO account | 2400 |

A full 4-level zoom will cost roughly 20 to 45 seconds. So **a visitor who is not logged in gets about 2 to 4 full runs per day.** Three consequences, all baked into the design above:

1. Precomputed examples and the Gallery tab must carry the first impression. They cost nothing.
2. `gr.Examples` with caching on: the very first visitor who clicks an example pays for it, and every visitor after that gets it instantly and free.
3. The declared `duration` must be tight. The platform checks `requested duration` against `remaining quota` **before** running. A run declared at 60 seconds is refused outright once the visitor has 59 seconds left, even if the real work needs 20. So we measure the real time and declare a little above it, scaled by the depth slider.

---

## 4. Risks, each with its fix

📌📌📌📌 **The biggest risk is the torch version. ZeroGPU only supports torch 2.8.0 and newer. This repo pins `torch>=2.4.1,<2.6` and `numpy<2`. The Space needs its own newer dependency set, and the vendored Chain-of-Zoom code must be tested against it before anything else.** 📌📌📌📌

| Risk | Fix |
|---|---|
| **torch 2.8+ required** (above). | Separate `requirements.txt` in the Space with torch 2.8.x, numpy 2, diffusers, transformers, peft. Then test the pipeline end to end. Step 1 of the build. |
| `torch.load` default changed. `ref/coz/osediff_sd3.py` calls `torch.load(self.vae_path, map_location="cpu")` at three places with no `weights_only` argument. torch 2.6 flipped that default to `True`. | If the SR_VAE `.pt` fails to load, add `weights_only=False`. One line, in our vendored copy. |
| **Stable Diffusion 3-medium is gated.** The Space cannot download it anonymously. | Add an `HF_TOKEN` secret in Space settings, from an account that has accepted the SD3 licence. `huggingface_hub` picks `HF_TOKEN` up on its own, with no `token=` argument anywhere in the code. Spaces do not inject it for you. This is exactly what the official gated `black-forest-labs/FLUX.1-dev` ZeroGPU Space does. **Do not use `preload_from_hub`**: gated support is undocumented and HF staff have an open PR to delete the field, replacing it with mounted volumes. |
| Disk. Settled, not a blocker. See section 4b. | Nothing to do. |
| **No git submodules on Spaces**, but the pipeline imports `osediff_sd3` from `ref/coz`. | Chain-of-Zoom is MIT licensed. Copy four files into the Space repo with attribution: `osediff_sd3.py`, `lora/lora_layers.py`, `utils/vaehook.py`, `utils/devices.py`. |
| `build_vlm` uses `device_map="auto"`, which asks accelerate to inspect real GPU memory. On ZeroGPU there is no real GPU at startup. | In the Space app, load the VLM with `device_map=None` and an explicit `.to("cuda")` at module scope. This is what ZeroGPU wants anyway. |
| flash-attention cannot be built on ZeroGPU. There is no `nvcc` at build time. | Nothing to do. `build_vlm` already falls back to sdpa. Single images do not need the varlen path. |
| Two visitors at once. ZeroGPU runs handlers in parallel by default. Our current code writes fixed paths like `{rec_dir}/{rec}.png`. | Rewrite the loop to pass PIL images in memory and use `tempfile` for anything that must hit disk. |
| The recursion is hardcoded to a center crop. | Add a `center` argument to the crop. Small change, and it unlocks the click-to-zoom feature. |

## 4b. Memory and disk, settled

Both looked like blockers. Neither is. The numbers below are measured or read from live Hugging Face APIs, not estimated.

**Host RAM does not bind us.** Hugging Face publishes no RAM figure for ZeroGPU. Their own hardware API returns `"cpu": "Dynamic resources"` and `"ram": "-"` for the `zero-a10g` tier, while every other tier has real numbers. The 16 GB figure in the docs belongs to CPU Basic, not ZeroGPU. It does not matter, because of how ZeroGPU works: weights you place on `cuda` at module scope are written to disk at a startup "pack" step and the RAM is freed, then streamed into VRAM per request. Host RAM never holds the whole model. Two public ZeroGPU Spaces prove the headroom: `black-forest-labs/FLUX.1-dev` loads 33.75 GB of weights, and `Qwen/Qwen-Image-2512` loads 57.70 GB. Both are running. Our roughly 27 GB is comfortable.

**Two separate disk budgets, and we fit in both.**

| Budget | What lands there | Our usage |
|---|---|---|
| Space ephemeral disk (the 50 GB figure) | Hugging Face download cache | SD3-medium **15.51 GB** + Qwen2.5-VL-3B about 7 GB + our merged transformer 8.35 GB + `ckpt/` 0.14 GB = **about 31 GB** |
| `/data-nvme/zerogpu-offload/` scratch | The ZeroGPU weight pack | about **27 GB** |

The 15.51 GB is exact: 15,505,436,951 bytes, computed by running diffusers' real file-selection code against the live repo listing. Two useful facts fell out of that. First, `stable-diffusion-3-medium-diffusers` is **already fp16 throughout**, and its `*.fp16.*` files are byte-identical duplicates of the default ones, so passing `variant="fp16"` saves nothing. Second, we do not need to publish an fp16 copy of `merged_transformer.safetensors`. The 8.35 GB fp32 file fits.

Reported ZeroGPU pack failures start at 35.1 GB, and that one turned out to be leftover packs from earlier boots rather than a real ceiling. Our 27 GB sits below it. If packing ever does fail with `OSError: [Errno 28] No space left on device` from `spaces/zero/torch/packing.py`, two things to try, in order: clear `/data-nvme/zerogpu-offload/`, then load the transformer and VAE in bf16 instead of fp32, which drops the pack to about 23 GB. Paid persistent storage does **not** enlarge that scratch area.

Keep the default `size="large"`. It gives 48 GB of VRAM, our pipeline needs about 27 GB, and `size="xlarge"` would double every visitor's quota cost for nothing.
| Cold start. The first visitor after a quiet spell waits while about 27 GB streams from disk into VRAM. | Accept it. Show a clear "warming up" message. Load models at module scope, which is what makes the streaming fast. |

---

## 5. What is built, and what is left

Everything lives in `space/`. It is a self-contained Space repo, not a Python package.

| File | What it is |
|---|---|
| `app.py` | The interface. Layout, streaming handler, preview overlay, examples. |
| `zoom.py` | Model loading and the recursion, in memory. Flattened from `src/opd_zoom/teacher/oracle_infer.py`. |
| `geometry.py` | Crop maths only. No heavy imports, so it stays cheap to test. |
| `video.py` | Renders the zoom clip. |
| `vendor/` | Four files copied from Chain-of-Zoom (MIT). Spaces do not support git submodules. |
| `samples/` | Five inputs for the examples, plus one shipped example result. |
| `test_space.py`, `test_app_smoke.py` | 16 tests, all passing, no GPU needed. |

### Done and verified here

- **The recursion is in memory and takes a zoom centre.** Was fixed to a centre crop and wrote every level to disk.
- **Results stream one level at a time**, so the first level shows up long before the last.
- **The zoom clip renders.** Each segment pushes in until the frame is exactly the blurry crop the next level came from, then holds and cross-fades blurry into sharp. That hold is where the viewer sees what the model added.
- **The page is never empty.** A real 256x run is shipped in `samples/` and loads as the starting value of the clip, the level strip, the compare slider, and the box overlay. Costs no GPU.
- **The whole Gradio graph builds and a full run completes in a browser**, checked with the model stubbed out.

Two bugs the tests caught, both of which would have shipped silently:

1. The clip's push loop used `f / PUSH_FRAMES`, so it stopped one frame short of the target crop. Every transition would have jumped.
2. `_build_sr` loaded the merged transformer **before** constructing `OSEDiff_SD3_TEST`. That construction swaps every targeted `Linear` for a `LoraInjectedLinear`, which renames the keys, so the checkpoint would have matched nothing and the demo would have quietly run the base model. Now it loads after, and raises if most keys go unmatched.

### Three bugs that only appear on ZeroGPU, found by deploying

ZeroGPU attaches no GPU while `app.py` is being imported. Anything that materialises weights
straight onto the model's own device therefore dies with `RuntimeError: No CUDA GPUs are
available`. Two places did.

1. **peft.** `PeftModel.from_pretrained` calls `infer_device()`, which reads the CUDA-available
   flag that ZeroGPU patches to `True`, then hands `"cuda"` to safetensors. Fixed with
   `torch_device="cpu"`. Checked separately that `PeftConfig` ignores the stray keyword.
2. **Chain-of-Zoom's `inject_lora`**, which does `torch.load(map_location=model.device)`. The
   whole super-resolution stack now assembles on CPU and moves to CUDA once at the end. One
   catch: `sr.device` has to be set by hand afterwards, because `encode_prompt` and
   `set_timesteps` read that attribute rather than the module devices. Leaving it `"cpu"` would
   have run text encoding on the CPU against GPU-resident encoders.
3. **`sentencepiece` and `protobuf` were missing.** SD3's third text encoder is T5, and its fast
   tokenizer is built from a sentencepiece model. Both arrive transitively in the research
   environment, so the slim Space requirements had dropped them.

### Layout, settled Sept 10

Inputs run down the left (upload, examples, click-to-place preview, the two point sliders, zoom
depth, the button, then the per-step prompts), results down the right (the zoom clip, the level
strip, the scale picker, the comparison). Both columns carry content the whole way, which is
what stops Gradio stretching one column's last component into an empty band.

Three layout traps worth remembering:

1. **`.gallery-container` carries the Gallery's height,** not the outer block. Styling only the
   block leaves the dead band behind. And any fixed height is wrong at some window width or
   depth: too short scrolls, too tall bands. CSS lets it size to its content instead.
2. **A Gallery handed to a handler arrives as `(path, caption)` tuples,** not the dicts the wire
   payload uses. Parsing it as dicts made the scale picker silently do nothing.
3. **Browsers restore slider values on reload,** which fires the preview handler before any
   upload. `preview` must therefore never return `None`, or the overlay blanks on a refresh.

### Left

- **Warm the example cache.** The examples currently only load a photo into the upload box, because caching needs `fn` and `outputs`, and those components are created further down the page than the examples now sit. Trade-off taken deliberately: placement over caching. The page still ships one real precomputed run, so a visitor with no quota sees real output on arrival.
- **Link it** from the README, the project page, and arXiv.

---

## 6. What we are deliberately not doing

- **No comparison against any other method.** Our zoom alone is the point. The before-and-after slider uses a plain bicubic enlargement, which costs nothing extra and is the honest floor.
- **No custom paid hardware,** unless ZeroGPU turns out to block us.
- **No image size limit beyond what the pipeline needs.** It centre-crops to 512x512 anyway, so a huge upload is just a slow upload.
- **No `preload_from_hub`.** Gated support is undocumented and HF staff have an open PR to delete the field. Weights download at startup instead, which is what the official gated FLUX.1-dev ZeroGPU Space does.
- **No `variant="fp16"` on SD3.** That repo is already fp16 throughout and its `*.fp16.*` files are byte-identical duplicates, so the flag saves nothing.
