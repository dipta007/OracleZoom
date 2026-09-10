"""EMA self-teacher weights (SDPO teacher_regularization=ema, rate 0.05).

Track a slow copy of the trainable LoRA params. The trainer swaps these in to run the
privileged teacher forward, then restores the live student. Pure tensor math here; the
swap/forward is in train_pld_onpolicy.py.
"""
import torch


def ema_init(params):
    """params: dict[name -> Tensor]. Returns a detached, no-grad clone (the EMA buffer)."""
    return {k: v.detach().clone() for k, v in params.items()}


def ema_update(ema, params, decay):
    """In-place: ema = decay*ema + (1-decay)*params, for keys present in `ema`.
    decay near 1 = slow teacher. SDPO update_rate 0.05 => decay 0.95."""
    with torch.no_grad():
        for k in ema:
            ema[k].mul_(decay).add_(params[k].detach(), alpha=1.0 - decay)
