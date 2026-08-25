"""HiT-SR (HiT_SRF) adapter (task #37): 4KLSDB HiT-SRF as a blind x4 SR op.

The 4KLSDB HiT-SR ckpt matches the STOCK HiT_SRF x4 config (options/Test/test_HiT_SRF_x4.yml):
embed_dim=60, depths=[6]x4, num_heads=[6]x4, base_win_size=[8,8], expansion_factor=2,
hier_win_ratios=[0.5,1,2,4,6,8], upsampler='pixelshuffledirect', resi_connection='1conv'.
Built with that config + load_state_dict(strict=True) => self-validating (load OK => config exact).

The upstream repo VENDORS its own basicsr (baselines/repos/upstream/hit_sr/basicsr) -> put that repo
on sys.path so `from basicsr.archs.hit_srf_arch import HiT_SRF` resolves to it. Regression SR
(no diffusion / no VLM) -> runs in coz env, fast. Isolated: no import into src/opd_zoom.
"""
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from adapters.base import SRAdapter


def _repo(name):
    """Locate an upstream repo (clones live on Lustre $CK/baselines/repos/upstream, not the /code tree)."""
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


_HITSR = _repo("hit_sr")
_WMAX = 8 * 8   # base_win_size(8) * max(hier_win_ratios)=8 -> pad input to a multiple of this


class Adapter(SRAdapter):
    name = "hit_sr"

    def __init__(self, ckpt_dir, device="cuda"):
        if _HITSR not in sys.path:
            sys.path.insert(0, _HITSR)
        from basicsr.archs.hit_srf_arch import HiT_SRF

        cf = os.path.join(ckpt_dir, "4KLSDB_x4.pth") if os.path.isdir(ckpt_dir) else ckpt_dir
        w = _weights(torch.load(cf, map_location="cpu"))
        self.scale = 4
        self.model = HiT_SRF(upscale=4, in_chans=3, img_size=64, base_win_size=[8, 8], img_range=1.,
                             depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                             expansion_factor=2, resi_connection="1conv",
                             hier_win_ratios=[0.5, 1, 2, 4, 6, 8], upsampler="pixelshuffledirect")
        self.model.load_state_dict(w, strict=True)          # strict -> config provably correct
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def sr_x4(self, img):
        x = to_tensor(img.convert("RGB")).unsqueeze(0).to(self.device)   # [0,1], 1x3xHxW
        _, _, h, w = x.shape
        hp = (_WMAX - h % _WMAX) % _WMAX                                  # pad to window multiple (no-op for 128)
        wp = (_WMAX - w % _WMAX) % _WMAX
        if hp or wp:
            x = F.pad(x, (0, wp, 0, hp), mode="reflect")
        out = self.model(x)[..., : h * self.scale, : w * self.scale].clamp(0, 1)
        return to_pil_image(out.squeeze(0).cpu())
