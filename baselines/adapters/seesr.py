"""SeeSR adapter (task #37): wraps ref/SeeSR inference as a blind x4 SR op.

Reuses SeeSR's OWN functions (load_seesr_pipeline, load_tag_model, get_validation_prompt,
adain_color_fix) so behavior matches ref/SeeSR/test_seesr.py EXACTLY (defaults read from it:
steps 50, guidance 5.5, conditioning 1.0, tiled 96/4, align adain, start_point lr).

The 4KLSDB SeeSR ckpt (seesr/x4_from_x1) is the fine-tuned unet+controlnet (`seesr_model_path`);
SD2.1 base + RAM/DAPE come from ref/SeeSR/preset/models (present on box). SeeSR's native RAM
tagging IS the SeeSR method (not a privileged/oracle prompt) -> keeping it = fair SeeSR baseline.
Isolated: does not import into / modify src/opd_zoom.
"""
import os
import sys
import types

import torch

from adapters.base import SRAdapter

_SEESR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ref", "SeeSR"))


class Adapter(SRAdapter):
    name = "seesr"

    def __init__(self, ckpt_dir, device="cuda"):
        if _SEESR not in sys.path:
            sys.path.insert(0, _SEESR)
        os.chdir(_SEESR)  # SeeSR load_tag_model reads RAM via relative 'preset/models/ram_swin_large_14m.pth'
        from accelerate import Accelerator
        from test_seesr import load_seesr_pipeline, load_tag_model, get_validation_prompt
        from utils.wavelet_color_fix import adain_color_fix
        self._get_prompt = get_validation_prompt
        self._adain = adain_color_fix

        preset = os.path.join(_SEESR, "preset", "models")
        self.args = types.SimpleNamespace(
            pretrained_model_path=os.path.join(preset, "stable-diffusion-2-base"),
            seesr_model_path=ckpt_dir,                       # our 4KLSDB seesr/x4_from_x1
            ram_ft_path=os.path.join(preset, "DAPE.pth"),
            prompt="", added_prompt="clean, high-resolution, 8k",
            negative_prompt="dotted, noise, blur, lowres, smooth",
            mixed_precision="fp16", guidance_scale=5.5, conditioning_scale=1.0,
            num_inference_steps=50, process_size=512, upscale=4,
            vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
            latent_tiled_size=96, latent_tiled_overlap=4, sample_times=1,
            align_method="adain", start_point="lr", seed=None,
        )
        self.acc = Accelerator(mixed_precision=self.args.mixed_precision)
        try:
            self.pipe = load_seesr_pipeline(self.args, self.acc, True)
        except Exception:                                    # xformers unavailable -> off
            self.pipe = load_seesr_pipeline(self.args, self.acc, False)
        self.tag = load_tag_model(self.args, device)
        self.device = device
        self.gen = torch.Generator(device=device)

    @torch.no_grad()
    def sr_x4(self, img):
        a = self.args
        prompt, ram_hidden = self._get_prompt(a, img, self.tag, self.device)   # RAM tags (SeeSR native)
        prompt = prompt + a.added_prompt
        r = a.upscale
        vi = img.resize((img.size[0] * r, img.size[1] * r))  # SeeSR pre-upscales x4, then SD-refines
        vi = vi.resize((vi.size[0] // 8 * 8, vi.size[1] // 8 * 8))
        w, h = vi.size
        with torch.autocast("cuda"):
            out = self.pipe(
                prompt, vi, num_inference_steps=a.num_inference_steps, generator=self.gen,
                height=h, width=w, guidance_scale=a.guidance_scale,
                negative_prompt=a.negative_prompt, conditioning_scale=a.conditioning_scale,
                start_point=a.start_point, ram_encoder_hidden_states=ram_hidden,
                latent_tiled_size=a.latent_tiled_size, latent_tiled_overlap=a.latent_tiled_overlap,
                args=a,
            ).images[0]
        return self._adain(out, vi)                          # align_method='adain'; ~512 (recursive_render normalizes)
