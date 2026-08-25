"""SwinIR adapter (task #37): 4KLSDB SwinIR as a blind x4 SR op.

The 4KLSDB SwinIR ckpt is a NON-STOCK config (embed_dim=240, 9 RSTB x depth6, num_heads=8,
nearest+conv upsampler, 3conv residual), so the upstream `--task classical_sr` defaults
(180 / [6]x6 / pixelshuffle / 1conv) do NOT match, and the 4klsdb patch config was never
vendored. So we INFER the arch from the ckpt's own key shapes and load `strict=True`:
self-validating -- if load_state_dict(strict=True) succeeds, the architecture is provably exact.

Regression SR (no diffusion / no VLM / no prompt) -> runs in coz env (torch 2.4.1 + timm), fast.
Isolated: does not import into / modify src/opd_zoom.
"""
import math
import os
import re
import sys

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from adapters.base import SRAdapter


def _repo(name):
    """Locate an upstream repo clone. Set OPDZ_REPOS to the directory holding them;
    otherwise falls back to baselines/repos/upstream next to this file."""
    cands = [os.environ.get("OPDZ_REPOS", ""),
             os.path.join(os.path.dirname(__file__), "..", "repos", "upstream")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, name)):
            return os.path.abspath(os.path.join(c, name))
    raise FileNotFoundError(f"upstream repo '{name}' not found (tried {cands})")


_SWINIR = _repo("swinir")


def _weights(sd):
    if isinstance(sd, dict):
        for k in ("params", "params_ema", "state_dict"):
            if k in sd:
                return sd[k]
    return sd


def _infer_arch(w):
    """Reconstruct SwinIR constructor args from state_dict key shapes (deterministic)."""
    embed = w["conv_first.weight"].shape[0]
    layers = {}
    for k in w:
        m = re.match(r"layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
        if m:
            layers.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    idx = sorted(layers)
    depths = [len(layers[i]) for i in idx]
    heads, ws = [], 8
    for i in idx:
        t = w[f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"]
        heads.append(t.shape[1])
        ws = int((math.sqrt(t.shape[0]) + 1) / 2)          # (2*ws-1)^2 = table.shape[0]
    mlp = w["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] // embed
    resi = "1conv" if "layers.0.conv.weight" in w else "3conv"
    if "conv_up1.weight" in w:
        ups = "nearest+conv"
    elif "upsample.0.weight" in w:
        ups = "pixelshuffle"
    else:
        ups = "pixelshuffledirect"
    return dict(embed_dim=embed, depths=depths, num_heads=heads, window_size=ws,
                mlp_ratio=float(mlp), resi_connection=resi, upsampler=ups)


class Adapter(SRAdapter):
    name = "swinir"

    def __init__(self, ckpt_dir, device="cuda"):
        if _SWINIR not in sys.path:
            sys.path.insert(0, _SWINIR)
        from models.network_swinir import SwinIR

        cf = os.path.join(ckpt_dir, "4KLSDB_x4.pth") if os.path.isdir(ckpt_dir) else ckpt_dir
        w = _weights(torch.load(cf, map_location="cpu"))
        a = _infer_arch(w)
        self.ws = a["window_size"]
        self.scale = 4
        # num_feat is hardcoded 64 inside SwinIR.__init__ (matches ckpt conv_before_upsample 64).
        self.model = SwinIR(upscale=self.scale, in_chans=3, img_size=64, img_range=1., **a)
        self.model.load_state_dict(w, strict=True)          # strict -> arch provably correct
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def sr_x4(self, img):
        x = to_tensor(img.convert("RGB")).unsqueeze(0).to(self.device)   # [0,1], 1x3xHxW
        _, _, h, w = x.shape
        ws = self.ws
        hp = (ws - h % ws) % ws                                          # reflect-pad to window multiple
        wp = (ws - w % ws) % ws
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : h + hp, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : w + wp]
        out = self.model(x)[..., : h * self.scale, : w * self.scale].clamp(0, 1)
        return to_pil_image(out.squeeze(0).cpu())
