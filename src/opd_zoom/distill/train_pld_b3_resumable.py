"""OracleZoom training: recursion-in-the-loop, multi-scale, on-policy distillation.

Trains a rank-16 LoRA on the SD3 transformer. Everything else (SR backbone, VAE,
prompt extractor) stays frozen. One image per step, both scales:

  PASS 1 (4x) : z1 = SR(blind_input, cached 4x prompt); x1 = decode(z1).
                A  = LPIPS(x1, gt_4x)                     [real ground truth exists here]
  RECURSE     : x2_in = bicubic_up(center_quarter(x1), 512)          [differentiable]
                prompt_2 = VLM([x1, x2_in]), live, text detached      [matches eval]
  PASS 2 (16x): z2 = SR(x2_in, prompt_2); x2 = decode(z2).
                B  = LPIPS(downsample4(x2), same_region_of(gt_4x))   [cycle coherence]
                C  = -TOPIQ-NR(x2)                                    [sharpness reward]
                D  = ||z2 - z2_base||^2, base = adapter disabled      [KL leash]
  EMA         : teacher = EMA of the LoRA, run on the privileged GT 4x crop.
                E  = relu(mse(z1, z1_ema)). 4x only. The relu is inert on a
                     squared distance; it is kept from the recipe this follows.

  loss = A + w*B + beta_reward*C + beta_kl*D + lambda_ema*E

Memory: the two passes are backwarded on separate graphs (x2_in is detached from x1
unless --full_grad), so both full SR graphs are never held at once. This also matches
eval, where recursions are independent forwards.

Resumable: optimizer, scheduler, step, epoch, data order, RNG, and EMA buffer are all
checkpointed, so --resume continues an interrupted run exactly.
"""
import argparse, json, os, random, statistics as st, time
import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from peft import LoraConfig, get_peft_model
from opd_zoom.teacher.oracle_infer import build_sr, build_vlm, vlm_prompt, vlm_prompt_batch
from opd_zoom.distill.sr_latent import (final_latent, final_latent_tensor, decode_tensor,
                                        final_latent_batch, final_latent_tensor_batch)
from opd_zoom.teacher.crop_geometry import gt_window
from opd_zoom.distill.train_utils import split_train_val_grouped, early_stop_update
from opd_zoom.distill.ema import ema_init, ema_update
from opd_zoom.distill.recursion_loop import center_quarter, bicubic_up, recursive_deep_pair
from opd_zoom.distill.rewards import build_reward


def _to_pil(x01):
    """(1,3,H,W) in [0,1] -> PIL. Detached; used only to feed the VLM (no grad through text)."""
    return TF.to_pil_image(x01.detach().float().clamp(0, 1)[0].cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot_list", required=True)
    ap.add_argument("--val_cache", default="",
                    help="scaling: dir with a pairs.json used as a FIXED val set (same across data tiers; "
                         "full tier then trains, no internal split). Empty (default) = internal val_frac "
                         "split (the parity-locked path, unchanged).")
    ap.add_argument("--coz_ckpt", default="ref/coz/ckpt")
    ap.add_argument("--sd3", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--vlm_lora", default="ref/coz/ckpt/VLM_LoRA/checkpoint-10000")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=16, help="reported config: 16")
    ap.add_argument("--target_modules", default="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    # --- multi-scale knobs ---
    ap.add_argument("--w_deep", type=float, default=1.0, help="weight on the 16x cycle loss")
    ap.add_argument("--deep_factor", type=int, default=4, help="extra zoom over 4x (4 -> 16x)")
    ap.add_argument("--lambda_ema", type=float, default=0.1, help="EMA self-teacher weight (term E)")
    ap.add_argument("--ema_decay", type=float, default=0.95)
    ap.add_argument("--full_grad", action="store_true",
                    help="keep the graph across recursions, so the deep loss also reshapes the 4x "
                         "output. Default (off): detach x4, deep loss trains pass 2 only.")
    ap.add_argument("--max_new_tokens", type=int, default=32, help="VLM 16x prompt length")
    # --- sharpness reward + KL leash (on the 16x output) ---
    ap.add_argument("--beta_kl", type=float, default=8.0,
                    help="weight on the KL leash, ||z16 - z16_base||^2 (term D), base = LoRA "
                         "disabled. Bounds reward-hacking drift. Reported config: 8.0. 0 = off.")
    ap.add_argument("--beta_reward", type=float, default=0.4,
                    help="weight on -reward(x16) (term C). Reported config: 0.4. 0 = off.")
    ap.add_argument("--reward_metric", default="topiq_nr",
                    help="differentiable reward net, disjoint from the eval metrics musiq/maniqa/clipiqa/niqe: "
                         "topiq_nr | hyperiqa | nima | tres | dbcnn | laplacian.")
    # early stopping
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_n", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=1,
                    help="batch size for the val eval (>1 = batched full-val eval, ~Nx faster). 1 = "
                         "per-image (unchanged). Batched math checked vs per-image on first eval.")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--warmup_evals", type=int, default=2)
    ap.add_argument("--resume", default="", help="RESUMABLE: path to a full-state ckpt.pt; restores "
                    "params+optimizer+scheduler+EMA+RNG(torch/cuda/python)+data-order+progress. "
                    "Default off.")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="RESUMABLE: save full-state <out>/ckpt.pt every N steps (0 = at every eval). Atomic.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)                       # deterministic LoRA init (reproducible + parity)
    if torch.cuda.is_available():                      # resume later restores torch_rng, overriding this
        torch.cuda.manual_seed_all(args.seed)

    sr = build_sr(args.coz_ckpt, args.sd3, 512)
    dev = sr.model.transformer.device
    vlm, vproc, pvi = build_vlm(args.vlm_lora)          # live 16x prompter (Qwen2.5-VL-3B + GRPO LoRA)
    pairs = json.load(open(os.path.join(args.cache, "pairs.json")))
    gtmap = {os.path.basename(p)[:-4]: p for p in open(args.pilot_list).read().split()}

    def gt_path(n):
        return pairs[n].get("gt") or gtmap.get(n)

    def usable(n):
        return os.path.exists(pairs[n]["blind_input"]) and bool(gt_path(n))

    train_keys = set(pairs)                                 # train pool = the --cache (tier) pairs
    if args.val_cache:                                      # FIXED external val (scaling): full tier trains
        vpairs = json.load(open(os.path.join(args.val_cache, "pairs.json")))
        for k, v in vpairs.items():
            pairs.setdefault(k, v)                          # merge so blind/gt/prompt lookups work for val too
        val_names = sorted(n for n in vpairs if n not in train_keys and usable(n))
        train_names = sorted(n for n in train_keys if usable(n))
        print(f"FIXED val_cache: train {len(train_names)}  val {len(val_names)}")
    else:                                                   # PARITY path (unchanged): internal val_frac split
        names = sorted(n for n in pairs if usable(n))
        print(f"usable samples: {len(names)}")
        train_names, val_names = split_train_val_grouped(names, args.val_frac, args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"train": train_names, "val": val_names}, open(os.path.join(args.out, "split.json"), "w"))
    print(f"train {len(train_names)}  val {len(val_names)}")


    _blind, _gt, _crop = {}, {}, {}
    def blind_pil(n):
        if n not in _blind:
            _blind[n] = Image.open(pairs[n]["blind_input"]).convert("RGB")
        return _blind[n]
    def gt_crop(n):
        # perf: decode full GT + crop the 4x window ONCE, cache the PIL. priv_pil (EMA-teacher path)
        # previously re-decoded the 2-6K GT every step. Cache is numerics-preserving (same crop).
        if n not in _crop:
            up = pairs[n].get("upscale", 4)
            c, _, _ = gt_window(Image.open(gt_path(n)).convert("RGB"), args.scale, upscale=up)
            _crop[n] = c
        return _crop[n]
    def gt_tensor(n):
        if n not in _gt:
            _gt[n] = TF.to_tensor(gt_crop(n)).unsqueeze(0)     # CPU cache (unchanged values)
        return _gt[n].to(dev)
    def priv_pil(n):
        return gt_crop(n)                                       # was: re-decode full GT every call

    import pyiqa
    lpips_fn = pyiqa.create_metric("lpips", device=dev, as_loss=True)
    # Reference metrics at 4x (GT exists): PSNR/SSIM (higher better), DISTS (lower better). Eval-only,
    # feeds the paper's 4x reference table. Built once per process (workers eval too).
    ref_psnr = pyiqa.create_metric("psnr", device=dev)
    ref_ssim = pyiqa.create_metric("ssim", device=dev)
    ref_dists = pyiqa.create_metric("dists", device=dev)
    # differentiable sharpness reward on the 16x output (grad-checked at build).
    reward_fn = None
    if args.beta_reward > 0:
        reward_fn, rmeta = build_reward(args.reward_metric, dev)
        print(f"reward: {rmeta} beta={args.beta_reward}", flush=True)

    # LoRA on the SD3 transformer only. The KL base is read via disable_adapter(),
    # for the KL base forward (swap-in / restore, same pattern as the EMA teacher). #27 ablation otherwise:
    # LoRA on transformer (default) / VAE decoder / both.
    # LoRA on the SD3 transformer; everything else frozen. The KL leash reads the
    # base model via transformer.disable_adapter(), so no weight snapshot is needed.
    params = {}
    tcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                      target_modules=args.target_modules.split(","), lora_dropout=0.0)
    sr.model.transformer = get_peft_model(sr.model.transformer, tcfg)
    sr.model.transformer.train()
    for k, p in sr.model.transformer.named_parameters():
        if p.requires_grad:
            params["transformer." + k] = p
    print(f"trainable params: {sum(p.numel() for p in params.values())}")
    ema = ema_init(params)

    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))

    def teacher_latent4x(n):
        # swap EMA weights in, run PRIVILEGED 4x GT crop, get z'_teacher, restore live weights.
        backup = {k: params[k].detach().clone() for k in params}
        with torch.no_grad():
            for k in params:
                params[k].copy_(ema[k])
            zt = final_latent(sr, priv_pil(n), pairs[n]["prompt"]).detach()
            for k in params:
                params[k].copy_(backup[k])
        return zt

    def prompt16(x4_pil, x16_in_pil):
        # LIVE VLM 16x prompt from the student's OWN 4x output (matches eval). Detached: no grad.
        with torch.no_grad():
            return vlm_prompt(vlm, vproc, pvi, x4_pil, x16_in_pil, max_new_tokens=args.max_new_tokens)

    def prompt16_batch(x4_pils, x16_in_pils):
        # Batched prompt16 for EVAL: one VLM generate over the batch (greedy, deterministic).
        with torch.no_grad():
            return vlm_prompt_batch(vlm, vproc, pvi, list(zip(x4_pils, x16_in_pils)),
                                    max_new_tokens=args.max_new_tokens)

    def losses(n):
        """Return (loss_4x, loss_16x, loss_sharp, z4). Two SR passes; the LoRA (shared) gets
        gradient from BOTH via l4 (pass-1 graph) and l16 (pass-2 graph). x16_in is DETACHED from x4
        UNLESS --full_grad. z4 is returned so the EMA pull reuses it (no 3rd forward).
        l_sharp = -reward(x16), backpropped through the frozen decoder into the LoRA;
        0 when beta_reward==0. x16 is the SAME decoded 16x tensor used for the cycle
        loss, so the reward sees the full-res deep output (not the downsampled anchor)."""
        # PASS 1 (4x)
        z4 = final_latent(sr, blind_pil(n), pairs[n]["prompt"])
        x4 = decode_tensor(sr, z4)                                  # (1,3,512,512) [0,1], grad
        l4 = lpips_fn(x4, gt_tensor(n)).mean()
        # RECURSE -> 16x SR input (CoZ crop center 1/factor + bicubic to 512).
        # default: detach x4 -> deep loss trains pass 2 only, matching eval's independent
        # recursions. --full_grad: keep the graph -> deep loss ALSO reshapes the 4x output.
        x4_rec = x4 if args.full_grad else x4.detach()
        x16_in = bicubic_up(center_quarter(x4_rec, args.deep_factor), 512)
        p16 = prompt16(_to_pil(x4), _to_pil(x16_in))               # live VLM, detached text
        # KL base forward: adapter disabled -> the untouched pretrained model on the same input.
        # no-grad forward, restore trained weights -- BEFORE building the student z16 graph, so the in-place
        # swap never corrupts z16's leaves (same ordering rule as the EMA teacher). LoRA path defers to
        # disable_adapter() below (no swap needed).
        z16_base = None
        # PASS 2 (16x)
        z16 = final_latent_tensor(sr, x16_in, p16)
        x16 = decode_tensor(sr, z16)
        a_deep, a_gt = recursive_deep_pair(x16, gt_tensor(n), args.deep_factor)
        l16 = lpips_fn(a_deep, a_gt).mean()
        # sharpness reward on the FULL-RES 16x output (not the downsampled anchor).
        l_sharp = -reward_fn(x16) if reward_fn is not None else x16.new_zeros(())
        # KL leash (DPOK/DRaFT style). Penalize the student's 16x latent for
        # drifting from what the UN-rewarded base produces on the SAME input -> bounds how far the reward
        # can drag the model. Latent-space (no extra decode); base forward is no-grad. 0 when beta_kl==0.
        if args.beta_kl > 0:
            if z16_base is None:  # LoRA path: base = adapter disabled
                with torch.no_grad(), sr.model.transformer.disable_adapter():
                    z16_base = final_latent_tensor(sr, x16_in.detach(), p16)
            l_kl = F.mse_loss(z16, z16_base)
        else:
            l_kl = x16.new_zeros(())
        return l4, l16, l_sharp, z4, l_kl

    # the module carrying the trainable weights -> toggle train/eval on it.
    toggle_mods = [sr.model.transformer]

    eval_val = random.Random(args.seed + 1).sample(val_names, min(args.eval_n, len(val_names)))
    def val_metric():
        for m in toggle_mods:
            m.eval()
        agg = {"l4": [], "l16": [], "l_sharp": [], "l_kl": []}
        with torch.no_grad():
            for n in eval_val:
                l4, l16, l_sharp, _, l_kl = losses(n)
                agg["l4"].append(float(l4)); agg["l16"].append(float(l16))
                agg["l_sharp"].append(float(l_sharp)); agg["l_kl"].append(float(l_kl))
        for m in toggle_mods:
            m.train()
        r = {k: st.mean(v) for k, v in agg.items()}
        # SAME objective as training; mean-of-linear-combo == per-image-mean, so `loss` is unchanged.
        r["loss"] = r["l4"] + args.w_deep * r["l16"] + args.beta_reward * r["l_sharp"] + args.beta_kl * r["l_kl"]
        r["topiq_nr"] = -r["l_sharp"]                       # reward score on the val 16x outputs
        return r

    def batched_val_metric(vnames, eb):
        """Same objective as val_metric, but SR forwards run in batches of `eb` (VLM still per-image).
        Not bit-exact vs per-image (VAE sample() RNG), but mean matches within sampling noise. LoRA-only
        KL path (disable_adapter); our scaling cells are LoRA."""
        for m in toggle_mods:
            m.eval()
        agg = {"l4": [], "l16": [], "l_sharp": [], "l_kl": [], "prompt_len": [],
               "psnr": [], "ssim": [], "dists": []}
        with torch.no_grad():
            for i in range(0, len(vnames), eb):
                ch = vnames[i:i + eb]
                z4 = final_latent_batch(sr, [blind_pil(n) for n in ch], [pairs[n]["prompt"] for n in ch])
                x4 = decode_tensor(sr, z4)                                      # [B,3,512,512]
                gt = torch.cat([gt_tensor(n) for n in ch], 0)                   # [B,3,512,512]
                agg["l4"] += lpips_fn(x4, gt).flatten().tolist()
                agg["psnr"] += ref_psnr(x4, gt).flatten().tolist()             # 4x reference metrics vs GT
                agg["ssim"] += ref_ssim(x4, gt).flatten().tolist()
                agg["dists"] += ref_dists(x4, gt).flatten().tolist()
                x16_in = bicubic_up(center_quarter(x4, args.deep_factor), 512)
                x4_pils = [_to_pil(x4[j:j + 1]) for j in range(len(ch))]
                x16_pils = [_to_pil(x16_in[j:j + 1]) for j in range(len(ch))]
                p16 = prompt16_batch(x4_pils, x16_pils)                        # ONE VLM generate over batch
                agg["prompt_len"] += [len([t for t in p.split(",") if t.strip()]) for p in p16]  # VLM tag count ("response length")
                z16 = final_latent_tensor_batch(sr, x16_in, p16)
                x16 = decode_tensor(sr, z16)
                a_deep, a_gt = recursive_deep_pair(x16, gt, args.deep_factor)
                agg["l16"] += lpips_fn(a_deep, a_gt).flatten().tolist()
                agg["l_sharp"] += ((-reward_fn(x16)).flatten().tolist() if reward_fn is not None else [0.0] * len(ch))
                if args.beta_kl > 0:
                    with sr.model.transformer.disable_adapter():
                        z16_base = final_latent_tensor_batch(sr, x16_in, p16)
                    agg["l_kl"] += F.mse_loss(z16, z16_base, reduction="none").mean(dim=(1, 2, 3)).flatten().tolist()
                else:
                    agg["l_kl"] += [0.0] * len(ch)
        for m in toggle_mods:
            m.train()
        r = {k: st.mean(v) for k, v in agg.items()}
        r["loss"] = r["l4"] + args.w_deep * r["l16"] + args.beta_reward * r["l_sharp"] + args.beta_kl * r["l_kl"]
        r["topiq_nr"] = -r["l_sharp"]
        return r

    def save_best():
        sr.model.transformer.save_pretrained(os.path.join(args.out, "student_lora"))


    rng = random.Random(args.seed)
    steps_per_epoch = max(1, len(train_names) // args.batch_size)
    best_val, best_step, best_epoch, patience_ctr, n_eval, step, stop = float("inf"), -1, -1, 0, 0, 0, False

    # --- RESUMABLE (native full-state; no framework: this loop has NO DataLoader, data pos = epoch/bi/order/rng) ---
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    def save_ckpt(epoch, bi, order):
        ck = {"params": {k: v.detach().cpu() for k, v in params.items()},
              "opt": opt.state_dict(), "sched": sched.state_dict(),
              "ema": ({k: v.detach().cpu() for k, v in ema.items()} if ema is not None else None),
              "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
              "epoch": epoch, "bi": bi, "order": order, "step": step, "best_val": best_val,
              "best_step": best_step, "best_epoch": best_epoch, "patience_ctr": patience_ctr,
              "n_eval": n_eval, "args": vars(args)}
        tmp = ckpt_path + ".tmp"; torch.save(ck, tmp); os.replace(tmp, ckpt_path)  # atomic overwrite
    start_epoch, start_bi, saved_order = 0, 0, None
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        for k in params:
            params[k].data.copy_(ck["params"][k].to(params[k].device, params[k].dtype))
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ema is not None and ck["ema"] is not None:
            for k in ema:
                ema[k].copy_(ck["ema"][k].to(ema[k].device, ema[k].dtype))
        rng.setstate(ck["rng"]); torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        step, best_val, best_step, best_epoch = ck["step"], ck["best_val"], ck["best_step"], ck["best_epoch"]
        patience_ctr, n_eval = ck["patience_ctr"], ck["n_eval"]
        start_epoch, start_bi, saved_order = ck["epoch"], ck["bi"] + 1, ck["order"]  # continue AFTER saved batch
        print(f"RESUMED {args.resume}: epoch {start_epoch} bi {start_bi} step {step} best {best_val:.4f}", flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        if saved_order is not None and epoch == start_epoch:
            order = saved_order                       # reuse the checkpointed epoch's shuffle
        else:
            order = train_names[:]; rng.shuffle(order)
        bi0 = start_bi if (saved_order is not None and epoch == start_epoch) else 0
        for bi in range(bi0, steps_per_epoch):
            batch = order[bi * args.batch_size:(bi + 1) * args.batch_size]
            if not batch:
                continue
            opt.zero_grad(); tr4 = tr16 = trsh = trkl = trloss = trema = 0.0
            for n in batch:
                # EMA teacher first (swaps LoRA weights in place under no_grad + restores),
                # so the student graph built after restore keeps its leaves unmutated.
                zt = teacher_latent4x(n)
                l4, l16, l_sharp, z4, l_kl = losses(n)
                loss = l4 + args.w_deep * l16
                if args.beta_reward > 0:
                    loss = loss + args.beta_reward * l_sharp          # term C: -reward(x16)
                if args.beta_kl > 0:
                    loss = loss + args.beta_kl * l_kl                 # term D: distance-to-base
                ema_term = args.lambda_ema * torch.relu(F.mse_loss(z4, zt))  # one-sided 4x self-distill (reuses z4)
                loss = loss + ema_term
                trema += float(ema_term.detach())
                loss.backward()
                tr4 += float(l4.detach()); tr16 += float(l16.detach()); trsh += float(l_sharp.detach()); trkl += float(l_kl.detach())
                trloss += float(loss.detach())              # the ACTUAL optimized objective (all terms)
            opt.step(); sched.step(); step += 1
            ema_update(ema, params, args.ema_decay)
            if step % args.eval_every == 0:
                _ev0 = time.time()
                vd = (batched_val_metric(eval_val, args.eval_batch) if args.eval_batch > 1
                      else val_metric())
                vd["eval_secs"] = time.time() - _ev0            # validation wall-time
                v = vd["loss"]; n_eval += 1
                improved, best_val, patience_ctr = early_stop_update(v, best_val, patience_ctr, args.min_delta)
                if improved:
                    best_step, best_epoch = step, epoch; save_best(); tag = " *best (saved)"
                else:
                    tag = f"  no-improve {patience_ctr}/{args.patience}"
                shtxt = f"lsh {trsh/len(batch):.4f} " if args.beta_reward > 0 else ""
                print(f"epoch {epoch} step {step} l4 {tr4/len(batch):.4f} l16 {tr16/len(batch):.4f} "
                      f"{shtxt}val {v:.4f} best {best_val:.4f}@e{best_epoch}s{best_step}{tag}", flush=True)
                if n_eval > args.warmup_evals and patience_ctr >= args.patience:
                    print(f"EARLY_STOP at epoch {epoch} step {step}"); stop = True
            if (args.ckpt_every and step % args.ckpt_every == 0) or \
               (not args.ckpt_every and step % args.eval_every == 0):
                save_ckpt(epoch, bi, order)          # RESUMABLE: latest full state (atomic overwrite)
            if stop:
                break
        if stop:
            break

    if best_step < 0:
        save_best(); best_step, best_epoch = 0, 0
    json.dump({"best_val": best_val, "best_step": best_step, "best_epoch": best_epoch,
               "stopped_early": stop, "w_deep": args.w_deep, "deep_factor": args.deep_factor,
               "lambda_ema": args.lambda_ema, "ema_decay": args.ema_decay,
               "full_grad": args.full_grad,
               "beta_reward": args.beta_reward, "reward_metric": args.reward_metric, "beta_kl": args.beta_kl,
               "lora_rank": args.lora_rank, "n_train": len(train_names), "n_val": len(val_names)},
              open(os.path.join(args.out, "train_meta.json"), "w"))
    print(f"BEST val {best_val:.4f} @ epoch {best_epoch} step {best_step}")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
