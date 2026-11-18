"""Full-reference faithfulness: how close is a generated crop to the REAL GT crop.

LPIPS + DISTS via pyiqa (lower = closer). DINOv2 CLS cosine (higher = closer). Use only where
GT is real (4x; 16x near-real, caveated per the spec). Inputs are PIL RGB; sizes may differ
(we resize the GT to match the generated crop before scoring).
"""
import torch
import torch.nn.functional as F
import pyiqa
import torchvision.transforms.functional as TF

_cache = {}


def _get(name, device):
    key = (name, device)
    if key not in _cache:
        _cache[key] = pyiqa.create_metric(name, device=device)
    return _cache[key]


def _to_tensor(img, size, device):
    return TF.to_tensor(img.resize(size)).unsqueeze(0).to(device)


def faithfulness(gen_img, gt_img, device="cuda"):
    """Return dict: lpips, dists (lower better), dino_cos (higher better).

    gen_img, gt_img are PIL RGB. GT is resized to the generated crop size so both match.
    """
    size = gen_img.size
    g = _to_tensor(gen_img, size, device)
    r = _to_tensor(gt_img, size, device)
    lpips = float(_get("lpips", device)(g, r).item())
    dists = float(_get("dists", device)(g, r).item())
    dino_cos = _dino_cosine(g, r, device)   # float, or nan if DINO unavailable on this node
    return {"lpips": lpips, "dists": dists, "dino_cos": dino_cos}


def _load_dino(device):
    # multi-node: the torch.hub dinov2 repo import is fragile on child nodes
    # ("No module named 'dinov2.hub'"). Try local source + trust; if it still fails,
    # return None so the caller degrades to NaN (LPIPS/DISTS are the primary metrics).
    import os
    th = os.environ.get("TORCH_HOME")
    repo = os.path.join(th, "hub", "facebookresearch_dinov2_main") if th else None
    try:
        if repo and os.path.isdir(repo):
            return torch.hub.load(repo, "dinov2_vits14", source="local", trust_repo=True).to(device).eval()
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(device).eval()
    except Exception as e:
        print(f"[faithfulness] DINO unavailable ({type(e).__name__}: {e}); dino_cos -> NaN")
        return None


def _dino_cosine(g, r, device):
    if "dino" not in _cache:
        _cache["dino"] = _load_dino(device)
    model = _cache["dino"]
    if model is None:
        return float("nan")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def emb(x):
        # dinov2 vits14 wants side divisible by 14; 224 is standard.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return model((x - mean) / std)

    return float(F.cosine_similarity(emb(g), emb(r)).item())
