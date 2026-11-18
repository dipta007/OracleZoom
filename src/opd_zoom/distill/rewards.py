"""Differentiable image-quality rewards for B5 (sharpness-reward distillation).

B3/B4's deep loss is anchored cycle-consistency: it downsamples the 16x output by 4 before
matching GT, so it rewards COHERENCE, not detail (a blurry-but-consistent deep image satisfies it
as well as a sharp one). B5 adds ONE term that scores the full-resolution deep output directly, so
its gradient rises with real high-frequency detail: L_sharp = -reward(x16). Minimizing -reward =
maximizing sharpness.

Two reward families:
  1. learned NR-IQA nets via pyiqa (as_loss=True) -- topiq_nr / hyperiqa / nima / tres / dbcnn.
     ANTI-REWARD-HACKING: these MUST stay disjoint from the eval set {musiq, maniqa, clipiqa, niqe}.
  2. model-free laplacian high-frequency energy -- zero overlap with any learned metric.

Every reward returns a scalar to MAXIMIZE (higher = sharper/better). The trainer negates it.
build_reward() grad-checks the net at construction so a non-differentiable metric fails loudly
instead of silently zeroing the term.
"""
import torch
import torch.nn.functional as F

# pyiqa metrics where higher = better. For lower-is-better nets we negate to keep the
# "maximize" convention. (niqe/brisque are lower-better but they are EVAL metrics, excluded here.)
_PYIQA_HIGHER_BETTER = {"topiq_nr", "hyperiqa", "nima", "tres", "dbcnn", "musiq", "maniqa", "clipiqa"}
# eval-only set: never allowed as a reward (training on the test metric = invalid).
_EVAL_ONLY = {"musiq", "maniqa", "maniqa-pipal", "clipiqa", "clipiqa+", "niqe"}
_MODEL_FREE = {"laplacian"}


def laplacian_energy(x):
    """Model-free sharpness: mean squared response of a 3x3 Laplacian, per-channel, batch-mean.
    Fully differentiable, no learned model -> zero overlap with any eval metric. Higher = sharper."""
    k = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    c = x.shape[1]
    k = k.expand(c, 1, 3, 3)
    lap = F.conv2d(x, k, padding=1, groups=c)
    return lap.pow(2).mean()


def build_reward(name, device):
    """Return (reward_fn, meta). reward_fn(img_01) -> scalar tensor to MAXIMIZE (grad enabled).

    img_01: (B,3,H,W) in [0,1] on `device`. name: 'laplacian' or a pyiqa NR metric name.
    Raises ValueError if `name` is in the eval-only set, or if the built metric does not pass a
    gradient-flow check (grad is None or all-zero on a random input)."""
    key = name.lower()
    if key in _EVAL_ONLY:
        raise ValueError(f"reward '{name}' is in the EVAL set {_EVAL_ONLY}; training on it = "
                         f"reward-hacking. Pick a disjoint reward (e.g. topiq_nr, laplacian).")

    if key in _MODEL_FREE:
        fn = laplacian_energy
        meta = {"name": key, "kind": "model_free", "higher_better": True}
    else:
        import pyiqa
        metric = pyiqa.create_metric(key, device=device, as_loss=True)
        metric.eval()
        for p in metric.parameters():
            p.requires_grad_(False)          # freeze the critic; grad flows THROUGH it, not INTO it
        sign = 1.0 if key in _PYIQA_HIGHER_BETTER else -1.0   # normalize to "higher = better"

        def fn(x, _m=metric, _s=sign):
            return _s * _m(x).mean()

        meta = {"name": key, "kind": "pyiqa", "higher_better": True, "raw_higher_better": sign > 0}

    _assert_differentiable(fn, device, name)
    return fn, meta


def _assert_differentiable(fn, device, name):
    """Fail loudly if the reward has no usable gradient (some pyiqa metrics are not as_loss-safe)."""
    x = torch.rand(1, 3, 224, 224, device=device, requires_grad=True)
    r = fn(x)
    if not torch.is_tensor(r) or r.ndim != 0:
        raise ValueError(f"reward '{name}' did not return a scalar tensor (got {type(r)}).")
    g, = torch.autograd.grad(r, x, retain_graph=False, allow_unused=True)
    if g is None or float(g.abs().sum()) == 0.0:
        raise ValueError(f"reward '{name}' produced no gradient (not differentiable as_loss). "
                         f"Use a different reward net.")
