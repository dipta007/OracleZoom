# src/opd_zoom/distill/sr_latent.py
"""Expose OSEDiff's one-step latents. z = VAE.encode(x)*scale ; z' = z - transformer(z,t,c).

Reuses build_sr from oracle_infer (which vendors OSEDiff from ref/coz).
"""
import torch
from torchvision import transforms

_tt = transforms.Compose([transforms.ToTensor()])


def _to_lq(pil, device):
    return (_tt(pil).unsqueeze(0).to(device) * 2 - 1)


def encode_latent(sr, pil):
    """z = VAE.encode(input)*scaling (the input latent)."""
    m = sr.model
    x = _to_lq(pil, m.vae.device).to(torch.float32)
    z = m.vae.encode(x).latent_dist.sample() * m.vae.config.scaling_factor
    return z.to(m.transformer.device)


def encode_latent_tensor(sr, x01):
    """Like encode_latent but from a differentiable tensor x01: (1,3,H,W) in [0,1] (KEEPS grad).
    Used by B3 pass-2: the 16x SR input is the student's own 4x output (a tensor), not a PIL."""
    m = sr.model
    x = (x01.to(m.vae.device, torch.float32) * 2 - 1)
    z = m.vae.encode(x).latent_dist.sample() * m.vae.config.scaling_factor
    return z.to(m.transformer.device)


def final_latent_tensor(sr, x01, prompt):
    """z' from a differentiable [0,1] tensor input (B3 recursion pass-2). Grad flows to x01."""
    m = sr.model
    z = encode_latent_tensor(sr, x01)
    pe, poe = m.encode_prompt([prompt], 1)
    m.scheduler.set_timesteps(1, device=m.device)
    t = m.scheduler.timesteps[0].expand(z.shape[0]).to(m.transformer.device)
    pe = pe.to(m.transformer.device, dtype=torch.float32)
    poe = poe.to(m.transformer.device, dtype=torch.float32)
    v = sr.predict_vector(z, t, pe, poe)
    return z - v


def final_latent(sr, pil, prompt):
    """z' = z - transformer(z, t, prompt_emb) (the post-step latent that decode() consumes)."""
    m = sr.model
    z = encode_latent(sr, pil)
    pe, poe = m.encode_prompt([prompt], 1)
    m.scheduler.set_timesteps(1, device=m.device)
    t = m.scheduler.timesteps[0].expand(z.shape[0]).to(m.transformer.device)
    pe = pe.to(m.transformer.device, dtype=torch.float32)
    poe = poe.to(m.transformer.device, dtype=torch.float32)
    v = sr.predict_vector(z, t, pe, poe)
    return z - v


def _lq_batch(pils, device):
    return torch.cat([_tt(p).unsqueeze(0) for p in pils], 0).to(device) * 2 - 1


def final_latent_batch(sr, pils, prompts):
    """Batched final_latent: B PILs + B prompts -> z'[B]. One VAE.encode + one transformer forward
    over the batch (SD3 has no cross-image ops at inference, so per-image results match the single
    path up to the VAE sample() RNG). Used for FAST full-val eval; no grad expected (call under no_grad)."""
    m = sr.model
    x = _lq_batch(pils, m.vae.device).to(torch.float32)
    z = m.vae.encode(x).latent_dist.sample() * m.vae.config.scaling_factor
    z = z.to(m.transformer.device)
    pe, poe = m.encode_prompt(list(prompts), 1)
    m.scheduler.set_timesteps(1, device=m.device)
    t = m.scheduler.timesteps[0].expand(z.shape[0]).to(m.transformer.device)
    v = sr.predict_vector(z, t, pe.to(m.transformer.device, torch.float32), poe.to(m.transformer.device, torch.float32))
    return z - v


def final_latent_tensor_batch(sr, x01, prompts):
    """Batched final_latent_tensor: [B,3,H,W] in [0,1] + B prompts -> z'[B]."""
    m = sr.model
    x = (x01.to(m.vae.device, torch.float32) * 2 - 1)
    z = m.vae.encode(x).latent_dist.sample() * m.vae.config.scaling_factor
    z = z.to(m.transformer.device)
    pe, poe = m.encode_prompt(list(prompts), 1)
    m.scheduler.set_timesteps(1, device=m.device)
    t = m.scheduler.timesteps[0].expand(z.shape[0]).to(m.transformer.device)
    v = sr.predict_vector(z, t, pe.to(m.transformer.device, torch.float32), poe.to(m.transformer.device, torch.float32))
    return z - v


def decode_tensor(sr, zprime):
    """Differentiable decode -> image tensor in [0,1], shape (1,3,H,W). Keeps grad for training."""
    m = sr.model
    img = m.decode(zprime.to(dtype=torch.float32, device=m.vae.device))  # in [-1,1]
    return (img.clamp(-1, 1) + 1) / 2
