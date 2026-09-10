"""OracleZoom recursive zoom, one image at a time, all in memory.

Flattened from src/opd_zoom/teacher/oracle_infer.py for the Space. Three differences, none
of them to the model: no disk round-trip between recursions, the zoom window can sit
anywhere instead of only the centre, and the loop is a generator so the UI can show each
level the moment it lands. Everything on one device, which is what ZeroGPU gives us.
"""
import gc
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from geometry import PROCESS_SIZE, resize_and_center_crop, zoom_window

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

REPO = "dipta007/OracleZoom"
SD3 = "stabilityai/stable-diffusion-3-medium-diffusers"
VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
COZ_PROMPT = (
    "The second image is a zoom-in of the first image. Based on this knowledge, "
    "what is in the second image? Give me a set of words."
)
_to_tensor = transforms.Compose([transforms.ToTensor()])


class _SRArgs:
    def __init__(self, coz_ckpt):
        self.lora_path = f"{coz_ckpt}/SR_LoRA/model_20001.pkl"
        self.vae_path = f"{coz_ckpt}/SR_VAE/vae_encoder_20001.pt"
        self.pretrained_model_name_or_path = SD3
        self.process_size = PROCESS_SIZE
        self.lora_rank = 4
        self.merge_and_unload_lora = False
        self.mixed_precision = "fp16"


class Models:
    """Built once at import, read-only afterwards. Concurrent requests only read."""

    def __init__(self, weights=None):
        from huggingface_hub import snapshot_download

        w = weights or snapshot_download(repo_id=REPO)
        self.vlm, self.proc, self.process_vision_info = _build_vlm(f"{w}/ckpt/VLM_LoRA/checkpoint-10000")
        self.sr = _build_sr(f"{w}/ckpt", f"{w}/merged_transformer.safetensors")


def _build_vlm(lora_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    # device_map="auto" asks accelerate to read real GPU memory, which does not exist at
    # module scope on ZeroGPU. Place it ourselves. sdpa because flash-attn has no wheel here.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # torch_device="cpu" is load-bearing. Left to itself peft calls infer_device(), which reads
    # the cuda flag ZeroGPU patches to True, then hands "cuda" to safetensors. That builds
    # tensors straight on a device that is not attached yet: RuntimeError, no CUDA GPUs.
    model = PeftModel.from_pretrained(model, lora_path, torch_device="cpu").merge_and_unload()
    return model.eval().to("cuda"), AutoProcessor.from_pretrained(VLM), process_vision_info


def _build_sr(coz_ckpt, merged_transformer):
    from safetensors.torch import load_file
    from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler

    # Assemble entirely on CPU, then move once. Two places here materialise weights straight
    # onto the model's own device (inject_lora's torch.load map_location, and safetensors), and
    # on ZeroGPU that device reads as cuda while no GPU is attached yet.
    sr = SD3Euler(device="cpu")
    # Construct first, load second. OSEDiff_SD3_TEST swaps every targeted Linear for a
    # LoraInjectedLinear, which renames the keys. Our merged checkpoint was saved after that
    # swap, so loading it earlier matches nothing and silently leaves the base transformer.
    test = OSEDiff_SD3_TEST(_SRArgs(coz_ckpt), sr)
    sd = load_file(merged_transformer)
    missing, unexpected = test.model.transformer.load_state_dict(
        {k: v.to(torch.float32) for k, v in sd.items()}, strict=False)
    if len(unexpected) > len(sd) // 2:
        raise RuntimeError(f"merged transformer did not match: {len(unexpected)} of {len(sd)} "
                           f"keys unexpected. LoRA injection order or checkpoint is wrong.")
    print(f"#### merged transformer loaded (missing {len(missing)} unexpected {len(unexpected)})")
    del sd
    gc.collect()

    for m in (sr.text_enc_1, sr.text_enc_2, sr.text_enc_3):
        m.to("cuda")
    sr.transformer.to("cuda", dtype=torch.float32)
    sr.vae.to("cuda", dtype=torch.float32)
    for m in (sr.text_enc_1, sr.text_enc_2, sr.text_enc_3, sr.transformer, sr.vae):
        m.requires_grad_(False)
    sr.device = "cuda"   # encode_prompt and set_timesteps read this, not the module devices
    return test


def write_prompt(models, first, second, max_new_tokens=32):
    messages = [
        {"role": "system", "content": COZ_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": first},
                                     {"type": "image", "image": second}]},
    ]
    text = models.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = models.process_vision_info(messages)
    inputs = models.proc(text=[text], images=images, videos=videos,
                         padding=True, return_tensors="pt").to("cuda")
    gen = models.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return models.proc.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0].strip()


def super_resolve(models, img, prompt):
    lq = _to_tensor(img).unsqueeze(0).to("cuda") * 2 - 1
    with torch.no_grad():
        out = torch.clamp(models.sr(lq, prompt=prompt)[0].cpu(), -1.0, 1.0)
    return transforms.ToPILImage()(out * 0.5 + 0.5)


def zoom(models, image, levels=4, upscale=4, center=(0.5, 0.5)):
    """Yield one (level, factor, prompt, blurry_input, result) per recursion.

    `blurry_input` is the plain bicubic enlargement the model starts from. It is what the
    super-resolution has to improve on, so it doubles as the honest before-picture.
    """
    cur = resize_and_center_crop(image)
    yield 0, 1, "", cur, cur
    for i in range(levels):
        blurry = zoom_window(cur, upscale, center).resize(cur.size, Image.BICUBIC)
        prompt = write_prompt(models, cur, blurry)
        cur = super_resolve(models, blurry, prompt)
        yield i + 1, upscale ** (i + 1), prompt, blurry, cur
