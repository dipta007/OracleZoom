"""CoZ recursive_multiscale, forked with a privilege switch on the VLM's SECOND image.

mode="student": second image = bicubic upscale of the center-crop of the previous SR output
                (EXACT CoZ behavior, ref/coz/inference_coz.py).
mode="oracle":  second image = real GT-HR crop for this recursion (crop_geometry.gt_window).
mode="null":    no VLM, prompt = "" (control: does the prompt matter at all vs the backbone).

The prompt string, the VLM (Qwen2.5-VL-3B + GRPO LoRA), the SR backbone (SD3/OSEDiff), the
device split, and the output layout are identical to CoZ. Only the pixels of the second image
differ. One variable changed.

SR input is ALWAYS the student bicubic zoom (privilege is prompt-only; we do not feed GT pixels
into the SR model, only into the prompt-writer). Run on the box:

    python -m opd_zoom.teacher.oracle_infer --mode oracle --gt_dir <pilot_dir> --out <dir>
"""
import argparse
import glob
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from opd_zoom.teacher.crop_geometry import gt_window

# Reuse CoZ code from the submodule.
_COZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ref", "coz"))
if _COZ not in sys.path:
    sys.path.insert(0, _COZ)

COZ_PROMPT = (
    "The second image is a zoom-in of the first image. Based on this knowledge, "
    "what is in the second image? Give me a set of words."
)
_to_tensor = transforms.Compose([transforms.ToTensor()])


def _fix_triton_ptxas():
    # flash-attn's rotary kernel uses Triton, which shells out to ptxas to read the CUDA version.
    # The box exports TRITON_PTXAS_PATH=system ptxas (CUDA 13.1); Triton 3.0.0 can't parse 13.x
    # ("Triton only support CUDA 10.0 or higher"). Point it at Triton's OWN bundled ptxas (CUDA 12.4,
    # the version Triton is built for) so the rotary kernel compiles. sm_90 is supported by 12.4.
    import triton
    bundled = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    if os.path.exists(bundled):
        os.environ["TRITON_PTXAS_PATH"] = bundled


def build_vlm(vlm_lora_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    _fix_triton_ptxas()                                  # before any flash-attn Triton kernel runs
    name = "Qwen/Qwen2.5-VL-3B-Instruct"
    # flash_attention_2 REQUIRED (no fallback): vision tower uses varlen flash-attn (no dense [T,T]
    # mask) so batched eval generate over many images does not OOM. Fail loudly if flash-attn missing.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    proc = AutoProcessor.from_pretrained(name)
    if vlm_lora_path:  # None/"" -> base VLM, no GRPO LoRA
        model = PeftModel.from_pretrained(model, vlm_lora_path).merge_and_unload()
    model = model.eval()
    return model, proc, process_vision_info


def vlm_prompt(model, proc, process_vision_info, first_path, second_path, max_new_tokens=32):
    messages = [
        {"role": "system", "content": COZ_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": first_path},
            {"type": "image", "image": second_path},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors="pt").to("cuda")
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return proc.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]


def vlm_prompt_batch(model, proc, process_vision_info, pairs, max_new_tokens=32):
    """Batched vlm_prompt: `pairs` = list of (first, second), each a PIL or path -> list of prompts.
    ONE generate() over the whole batch (greedy = deterministic), so eval no longer loops the VLM.
    LEFT padding: batched causal generate needs pads on the left, else short seqs generate from a
    padded position and the len(i) trim is wrong. Restore the side after (single-image path unaffected)."""
    if not pairs:
        return []
    convs = [[{"role": "system", "content": COZ_PROMPT},
              {"role": "user", "content": [{"type": "image", "image": a},
                                           {"type": "image", "image": b}]}] for a, b in pairs]
    texts = [proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    image_inputs, video_inputs = process_vision_info(convs)      # flattened across the batch
    side = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"
    try:
        inputs = proc(text=texts, images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to("cuda")
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        proc.tokenizer.padding_side = side
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]  # left-pad -> len(i)==max_len for all
    return proc.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)


class _SRArgs:
    """Minimal args object OSEDiff_SD3_TEST reads (verified vs ref/coz/osediff_sd3.py)."""
    def __init__(self, coz_ckpt, sd3, process_size):
        self.lora_path = f"{coz_ckpt}/SR_LoRA/model_20001.pkl"
        self.vae_path = f"{coz_ckpt}/SR_VAE/vae_encoder_20001.pt"
        self.pretrained_model_name_or_path = sd3
        self.process_size = process_size
        self.lora_rank = 4
        self.merge_and_unload_lora = False
        self.mixed_precision = "fp16"


def build_sr(coz_ckpt, sd3, process_size):
    # Device layout auto-adapts to how many GPUs are VISIBLE (set via CUDA_VISIBLE_DEVICES):
    #   1 GPU  -> everything on cuda:0 (H100 80GB fits the whole ~27GB pipeline). Lets us run
    #            ONE worker per GPU -> 64 workers on an 8x8 fleet (2x the old 2-GPU-per-worker).
    #   2+ GPUs -> original split (text encoders on cuda:0, transformer+VAE on cuda:1), so every
    #            existing script that launches with CUDA_VISIBLE_DEVICES=a,b behaves unchanged.
    from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
    sr = SD3Euler()
    enc_dev = "cuda:0"
    heavy_dev = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    sr.text_enc_1.to(enc_dev); sr.text_enc_2.to(enc_dev); sr.text_enc_3.to(enc_dev)
    sr.transformer.to(heavy_dev, dtype=torch.float32)
    sr.vae.to(heavy_dev, dtype=torch.float32)
    for p in [sr.text_enc_1, sr.text_enc_2, sr.text_enc_3, sr.transformer, sr.vae]:
        p.requires_grad_(False)
    return OSEDiff_SD3_TEST(_SRArgs(coz_ckpt, sd3, process_size), sr)


def run(mode, gt_dir, out_dir, coz_ckpt, sd3, rec_num=4, upscale=4, process_size=512,
        max_new_tokens=32, base_vlm=False, pld_lora=None, lora_scale=1.0, lora_scale_sched=None,
        decoder_lora=None, full_transformer=None):
    from inference_coz import resize_and_center_crop
    os.makedirs(out_dir, exist_ok=True)
    sr_test = build_sr(coz_ckpt, sd3, process_size)
    # full_transformer (T8c #14): load a full fine-tuned transformer state_dict (replaces ALL weights,
    # not an adapter). Mutually exclusive with pld_lora (that path is for LoRA students). Without this a
    # full-param student is evaluated as the BASE transformer = wrong numbers.
    if full_transformer:
        from safetensors.torch import load_file
        sd = load_file(full_transformer)
        dev = next(sr_test.model.transformer.parameters()).device
        sd = {k: v.to(dev, dtype=torch.float32) for k, v in sd.items()}
        missing, unexpected = sr_test.model.transformer.load_state_dict(sd, strict=False)
        print(f"#### full transformer loaded: {full_transformer} (missing {len(missing)} unexpected {len(unexpected)})")
    # decoder_lora (T8c #27): load a distilled adapter onto the VAE DECODER so eval uses the trained
    # decoder LoRA. Independent of pld_lora (transformer): a --lora_on both run has both; decoder-only
    # has just this. Without this, a decoder-trained student is evaluated as the BASE decoder = wrong.
    if decoder_lora:
        from peft import PeftModel
        sr_test.model.vae = PeftModel.from_pretrained(sr_test.model.vae, decoder_lora).eval()
        print(f"#### decoder LoRA loaded on VAE: {decoder_lora}")
    # pld_lora: load the distilled PLD adapter onto the SR transformer so EVERY recursion's SR
    # step uses the distilled student (recursion-transfer test E5). Same PeftModel load as
    # infer_pld. Only meaningful with mode="student" (blind inputs, no privilege at test).
    if pld_lora:
        from peft import PeftModel
        sr_test.model.transformer = PeftModel.from_pretrained(
            sr_test.model.transformer, pld_lora).eval()
        # lora_scale != 1.0 -> DRaFT-style LoRA scaling (T8c option 3): interpolate base<->finetuned
        # by multiplying the adapter contribution. s=1 keeps the trained student (default, back-compat);
        # s<1 dials the reward-hacked directions back toward the base. Scale the PEFT `scaling` dict so
        # base weights are untouched. s=0 == base (no adapter).
        if lora_scale != 1.0:
            n = 0
            for m in sr_test.model.transformer.modules():
                if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
                    for k in m.scaling:
                        m.scaling[k] *= lora_scale
                    n += 1
            print(f"#### LoRA scaled by {lora_scale} on {n} adapter layers")
        # per-scale schedule (T8c honest-deep-win): remember each adapter's BASE scaling so we can
        # reset+rescale per recursion. sched[rec] multiplies the base. Idea: full reward where the
        # input supports detail (16x), dial out where it degenerates to crosshatch (64/256x).
        _lora_mods, _base_scaling = [], []
        if lora_scale_sched:
            for m in sr_test.model.transformer.modules():
                if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
                    _lora_mods.append(m); _base_scaling.append({k: v for k, v in m.scaling.items()})
            print(f"#### per-scale LoRA schedule {lora_scale_sched} on {len(_lora_mods)} layers")
        print(f"#### PLD LoRA loaded on SR transformer: {pld_lora} (scale={lora_scale})")
    if mode == "null":
        model = proc = pvi = None   # control: no VLM, empty prompt
    else:
        # base_vlm=True -> raw Qwen2.5-VL-3B (no GRPO LoRA), to test privilege on a non-expert.
        lora = None if base_vlm else f"{coz_ckpt}/VLM_LoRA/checkpoint-10000"
        model, proc, pvi = build_vlm(lora)

    for img_path in sorted(p for e in ("*.png", "*.jpg", "*.jpeg") for p in glob.glob(f"{gt_dir}/{e}")):
        bname = os.path.splitext(os.path.basename(img_path))[0]
        rec_dir = os.path.join(out_dir, "per-sample", bname)
        os.makedirs(os.path.join(rec_dir, "txt"), exist_ok=True)
        gt = Image.open(img_path).convert("RGB")
        cur = resize_and_center_crop(gt, process_size)
        cur.save(f"{rec_dir}/0.png")
        print(f"#### IMAGE: {bname}")
        for rec in range(rec_num):
            prev = Image.open(f"{rec_dir}/{rec}.png").convert("RGB")
            w, h = prev.size
            nw, nh = w // upscale, h // upscale
            crop = prev.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
            student_zoom = crop.resize((w, h), Image.BICUBIC)
            student_zoom_path = f"{rec_dir}/{rec + 1}_input.png"
            student_zoom.save(student_zoom_path)
            # pixel_teacher: SR input = REAL GT crop (upper-bound headroom test for latent
            # distillation). prompt held null so ONLY the pixels differ vs the blind student.
            sr_input = student_zoom
            if mode == "pixel_teacher":
                gt_crop, _, _ = gt_window(gt, rec + 1, process_size, upscale)
                sr_input = gt_crop
                gt_crop.save(f"{rec_dir}/{rec + 1}_pixteacher.png")
                prompt = ""
            elif mode == "null":
                prompt = ""
            else:
                if mode == "oracle":
                    gt_crop, _, _ = gt_window(gt, rec + 1, process_size, upscale)
                    second_path = f"{rec_dir}/{rec + 1}_oracle.png"
                    gt_crop.save(second_path)
                else:
                    second_path = student_zoom_path
                prompt = vlm_prompt(model, proc, pvi, f"{rec_dir}/{rec}.png", second_path, max_new_tokens)
            with open(f"{rec_dir}/txt/{rec}.txt", "w") as f:
                f.write(prompt)
            print(f"RECURSION {rec}: {prompt}")
            # per-scale LoRA scaling: reset each adapter to base*sched[rec] for THIS recursion's SR.
            if lora_scale_sched:
                sc = lora_scale_sched[rec] if rec < len(lora_scale_sched) else lora_scale_sched[-1]
                for m, base in zip(_lora_mods, _base_scaling):
                    for k in m.scaling:
                        m.scaling[k] = base[k] * sc
            lq = _to_tensor(sr_input).unsqueeze(0).to("cuda") * 2 - 1
            with torch.no_grad():
                out = sr_test(lq, prompt=prompt)
                out = torch.clamp(out[0].cpu(), -1.0, 1.0)
                out_pil = transforms.ToPILImage()(out * 0.5 + 0.5)
            out_pil.save(f"{rec_dir}/{rec + 1}.png")
        for s in range(rec_num + 1):
            d = os.path.join(out_dir, "per-scale", f"scale{s}")
            os.makedirs(d, exist_ok=True)
            Image.open(f"{rec_dir}/{s}.png").save(os.path.join(d, f"{bname}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["student", "oracle", "null", "pixel_teacher"], required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coz_ckpt", default="ref/coz/ckpt")
    ap.add_argument("--sd3", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--rec_num", type=int, default=4)
    ap.add_argument("--upscale", type=int, default=4,
                    help="zoom-out per recursion. 4 = CoZ default (4x/16x/..). 6/8 = single-step "
                         "multi-zoom for T13b training aug (rec_num=1, real GT at min_side>=512*upscale).")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--base_vlm", action="store_true", help="use raw Qwen2.5-VL-3B, no GRPO LoRA")
    ap.add_argument("--pld_lora", default=None,
                    help="path to distilled PLD adapter (student_lora); loads it on the SR "
                         "transformer for every recursion (E5 transfer test). Use with --mode student.")
    ap.add_argument("--lora_scale", type=float, default=1.0,
                    help="T8c opt3: scale the PLD LoRA contribution (DRaFT-style). 1.0=trained student "
                         "(default), <1 interpolates toward base to undo reward hacking, 0=base only.")
    ap.add_argument("--lora_scale_sched", default=None,
                    help="T8c per-scale: comma list of LoRA scales, one per recursion "
                         "(e.g. '1,1,0.5,0.3' = full at 4x/16x, dial out 64x/256x). Overrides --lora_scale.")
    ap.add_argument("--decoder_lora", default=None,
                    help="T8c #27: path to a distilled VAE-decoder adapter (out/decoder_lora); loads it on "
                         "the SR VAE for every recursion. Use for --lora_on decoder|both students.")
    ap.add_argument("--full_transformer", default=None,
                    help="T8c #14: path to a full fine-tuned transformer state_dict "
                         "(out/full_transformer.safetensors); replaces ALL transformer weights.")
    args = ap.parse_args()
    sched = [float(x) for x in args.lora_scale_sched.split(",")] if args.lora_scale_sched else None
    run(args.mode, args.gt_dir, args.out, args.coz_ckpt, args.sd3,
        args.rec_num, upscale=args.upscale, max_new_tokens=args.max_new_tokens,
        base_vlm=args.base_vlm, pld_lora=args.pld_lora, lora_scale=args.lora_scale,
        lora_scale_sched=sched, decoder_lora=args.decoder_lora, full_transformer=args.full_transformer)
