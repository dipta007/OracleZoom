"""MambaIR adapter (task #37): 4KLSDB MambaIR as a blind x4 SR op.

The 4KLSDB MambaIR ckpt uses the STOCK config (from its train_config.yml: type MambaIR, upscale4,
embed_dim=180, depths=[6]x6, d_state=16, window_size=8, mlp_ratio=2, pixelshuffle, 1conv). Build
from upstream csguoh/MambaIR + load_state_dict(strict=True) => self-validating (load OK => config exact).

Runs in an isolated environment (torch 2.4.1/cu121 + mamba_ssm), separate from the training env.
The upstream repo VENDORS basicsr -> put it on sys.path so `from basicsr.archs.mambair_arch import MambaIR`
resolves. Isolated: no import into src/opd_zoom.
"""
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from adapters.base import SRAdapter


def _repo(name):
    cands = [os.environ.get("OPDZ_REPOS", ""),
             os.path.join(os.path.dirname(__file__), "..", "repos", "upstream")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, name)):
            return os.path.abspath(os.path.join(c, name))
    raise FileNotFoundError(f"upstream repo '{name}' not found (tried {cands})")


def _weights(sd):
    if isinstance(sd, dict):
        for k in ("params", "params_ema", "state_dict"):
            if k in sd:
                return sd[k]
    return sd


_MAMBAIR = _repo("mambair")
_WS = 8   # window_size -> pad input to a multiple (no-op for 128)


class Adapter(SRAdapter):
    name = "mambair"

    def __init__(self, ckpt_dir, device="cuda"):
        if _MAMBAIR not in sys.path:
            sys.path.insert(0, _MAMBAIR)
        from basicsr.archs.mambair_arch import MambaIR

        cf = os.path.join(ckpt_dir, "4KLSDB_x4.pth") if os.path.isdir(ckpt_dir) else ckpt_dir
        w = _weights(torch.load(cf, map_location="cpu"))
        self.scale = 4
        self.model = MambaIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                             d_state=16, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, mlp_ratio=2,
                             upsampler="pixelshuffle", resi_connection="1conv")
        self.model.load_state_dict(w, strict=True)          # strict -> config provably correct
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def sr_x4(self, img):
        x = to_tensor(img.convert("RGB")).unsqueeze(0).to(self.device)   # [0,1], 1x3xHxW
        _, _, h, w = x.shape
        hp = (_WS - h % _WS) % _WS
        wp = (_WS - w % _WS) % _WS
        if hp or wp:
            x = F.pad(x, (0, wp, 0, hp), mode="reflect")
        out = self.model(x)[..., : h * self.scale, : w * self.scale].clamp(0, 1)
        return to_pil_image(out.squeeze(0).cpu())
