"""OSEDiff adapter (task #37): 4KLSDB OSEDiff (one-step SD2.1 diffusion SR) as a blind x4 SR op.

The 4KLSDB osediff ckpt is an accelerate save_state; ONLY model.safetensors (unet+text_encoder+vae =
OSEDiff_gen) is needed for inference (model_1 = VSD regularizer, model_2 = LPIPS = training-only, ignored).

Stock OSEDiff_test.load_ckpt expects a .pkl {rank_unet/vae, lora module-lists, state_dict_unet/vae}. We
REBUILD that pkl from model.safetensors and let OSEDiff_test load + run UNCHANGED:
  - rank_unet/vae  = inferred from a `lora_A` key's shape[0] in the safetensors.
  - module lists   = OSEDiff's OWN initialize_unet/initialize_vae l_grep derivation (deterministic on a
                     fresh SD2.1 unet/vae) -> identical adapter target-modules to the trained model.
  - state_dict_*   = safetensors keys sliced by 'unet.'/'vae.' prefix.
load_ckpt copies weights by NAME (p.data.copy_(state_dict[n])) -> KeyErrors loudly if the mapping is wrong,
so a clean load is self-validating. text_encoder is frozen in OSEDiff (base CLIP) -> not loaded (correct).

Blind protocol: OSEDiff's native RAM/DAPE tag prompt (its own method, not a privileged prompt) -- same as
the SeeSR baseline. Runs in an isolated environment (SD2.1 + diffusers + peft + RAM), separate from the training env.
Isolated: no import into src/opd_zoom.
"""
import os
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from adapters.base import SRAdapter


def _repo(name):
    cands = [os.environ.get("OPDZ_REPOS", ""),
             os.path.join(os.path.dirname(__file__), "..", "repos", "upstream")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, name)):
            return os.path.abspath(os.path.join(c, name))
    raise FileNotFoundError(f"upstream repo '{name}' not found (tried {cands})")


_OSEDIFF = _repo("osediff")
# SD2.1 base + RAM + DAPE weights are present under ref/SeeSR/preset/models (reused, on box).
_PRESET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ref", "SeeSR", "preset", "models"))


def _build_pkl(sf, sd21, initialize_unet, initialize_vae):
    """Reconstruct the .pkl OSEDiff_test.load_ckpt expects, from model.safetensors (name-based)."""
    rank_u = next(v.shape[0] for k, v in sf.items() if k.startswith("unet.") and "lora_A" in k)
    rank_v = next(v.shape[0] for k, v in sf.items() if k.startswith("vae.") and "lora_A" in k)
    _, ue, ud, uo = initialize_unet(SimpleNamespace(pretrained_model_name_or_path=sd21, lora_rank=rank_u))
    _, ve = initialize_vae(SimpleNamespace(pretrained_model_name_or_path=sd21, lora_rank=rank_v))
    sdu = {k[len("unet."):]: v for k, v in sf.items() if k.startswith("unet.")}
    sdv = {k[len("vae."):]: v for k, v in sf.items() if k.startswith("vae.")}
    return dict(rank_unet=rank_u, unet_lora_encoder_modules=ue, unet_lora_decoder_modules=ud,
                unet_lora_others_modules=uo, state_dict_unet=sdu,
                rank_vae=rank_v, vae_lora_encoder_modules=ve, state_dict_vae=sdv)


class Adapter(SRAdapter):
    name = "osediff"

    def __init__(self, ckpt_dir, device="cuda"):
        if _OSEDIFF not in sys.path:
            sys.path.insert(0, _OSEDIFF)
        os.chdir(_OSEDIFF)                       # RAM/relative-import + preset paths resolve from repo root
        from osediff import OSEDiff_test, initialize_unet, initialize_vae
        from ram.models.ram_lora import ram
        from ram import inference_ram
        from my_utils.wavelet_color_fix import adain_color_fix
        self._inference_ram = inference_ram
        self._adain = adain_color_fix

        sd21 = os.path.join(_PRESET, "stable-diffusion-2-base")
        self.args = SimpleNamespace(
            pretrained_model_name_or_path=sd21, mixed_precision="fp16", prompt="",
            process_size=512, upscale=4, align_method="adain", merge_and_unload_lora=False,
            vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
            latent_tiled_size=96, latent_tiled_overlap=32, osediff_path=None,
        )
        cf = os.path.join(ckpt_dir, "model.safetensors") if os.path.isdir(ckpt_dir) else ckpt_dir
        pkl = _build_pkl(load_file(cf), sd21, initialize_unet, initialize_vae)
        pkl_path = f"/tmp/osediff_4klsdb_{os.getpid()}.pkl"
        torch.save(pkl, pkl_path)                # OSEDiff_test.__init__ does torch.load(osediff_path)
        self.args.osediff_path = pkl_path
        self.model = OSEDiff_test(self.args)     # load_ckpt copies by name -> self-validating

        self.ram = ram(pretrained=os.path.join(_PRESET, "ram_swin_large_14m.pth"),
                       pretrained_condition=os.path.join(_PRESET, "DAPE.pth"),
                       image_size=384, vit="swin_l").eval().to(device)
        self.ram_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.to_t = transforms.ToTensor()
        self.device = device

    @torch.no_grad()
    def sr_x4(self, img):
        a = self.args
        vi = img.convert("RGB")
        vi = vi.resize((vi.size[0] * a.upscale, vi.size[1] * a.upscale), Image.LANCZOS)   # pre-upscale x4
        vi = vi.resize((vi.size[0] // 8 * 8, vi.size[1] // 8 * 8), Image.LANCZOS)          # multiple of 8
        lq = self.to_t(vi).unsqueeze(0).to(self.device)                                    # [0,1]
        lq_ram = self.ram_tf(lq)                                                           # RAM/DAPE runs fp32
        caps = self._inference_ram(lq_ram, self.ram)
        prompt = f"{caps[0]}, {a.prompt},"                                                 # RAM/DAPE tags (native)
        out = self.model(lq * 2 - 1, prompt=prompt)                                        # [-1,1]
        pil = transforms.ToPILImage()(out[0].detach().cpu() * 0.5 + 0.5)
        return self._adain(target=pil, source=vi)                                          # align_method=adain
