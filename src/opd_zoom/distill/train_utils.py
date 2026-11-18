# src/opd_zoom/distill/train_utils.py
"""Pure (torch-free) training helpers for PLD: deterministic split + early-stop rule.

Kept torch-free so they are importable + testable on the Mac (matching the repo pattern where
crop_geometry.py is torch-free too). The trainer imports these.
"""
import random
import re


def base_image_key(name):
    """Base image id for an augmented sample name: strip trailing zoom (_u<digits>) and/or
    privilege-source (_m) suffixes. So 0001, 0001_u6, 0001_u8, 0001_m all map to 0001 -> grouped
    on the SAME split side (no image content leaks train<->val across zoom/privilege variants)."""
    return re.sub(r"(_u\d+)?(_m)?$", "", name)


def split_train_val_grouped(names, val_frac, seed, key=base_image_key):
    """Group-aware split: all samples sharing a key (same base image) go to the SAME side, so a
    multi-zoom expansion can't leak an image's content from train into val. Splits on the groups
    (val_frac of the GROUPS), then expands back to sample names. Deterministic / seed-stable."""
    groups = {}
    for n in names:
        groups.setdefault(key(n), []).append(n)
    gkeys = sorted(groups)                         # stable order before shuffle
    random.Random(seed).shuffle(gkeys)
    n_val = max(1, int(len(gkeys) * val_frac))
    train = [n for k in gkeys[n_val:] for n in groups[k]]
    val = [n for k in gkeys[:n_val] for n in groups[k]]
    return train, val


def early_stop_update(val, best_val, patience_ctr, min_delta):
    """One early-stop step. Lower val = better. Returns (improved, new_best, new_ctr).
    'improved' iff val < best_val*(1-min_delta); resets ctr on improve, else increments."""
    if val < best_val * (1 - min_delta):
        return True, val, 0
    return False, best_val, patience_ctr + 1
