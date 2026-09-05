# src/opd_zoom/eval/full_eval.py
"""Full-panel scorer for ONE rendered arm. Writes one JSON with means + per-image values.

Input = a per-scale render dir (scale1..scale4/*.png) produced by oracle_infer (blind / pld_lora
/ pixel_teacher, rec_num 4). Scores:
  - @scale1 (4x, real GT): FR panel + NR panel + FID.
  - @scale2/3/4 (16x/64x/256x): NR panel only (no real GT past 4x -> no FR, per claim discipline).
Uses pinned pyiqa metric ids, plus faithfulness() for lpips/dists/dino.

GT for FR: the real HR window gt_window(gt, i) at the matching recursion i (i=1..4). Only scored
where the window is real (is_real). For DIV2K most scales have no real GT -> FR skipped there.

Usage (box, CoZ env):
  python -m opd_zoom.eval.full_eval --render_dir <per-scale parent> --names_json <split.json:key>
    --gt_list <paths.txt> --arm opt2_r64 --method opt2 --rank 64 --evalset div8k_test
    --out <results.json> [--full_fr] [--fid] [--device cuda]
"""
import argparse, json, os, statistics as st
import torch
import pyiqa
from importlib.metadata import version as _pkg_version
from PIL import Image
import torchvision.transforms.functional as TF
from opd_zoom.teacher.crop_geometry import gt_window
from opd_zoom.eval.faithfulness import faithfulness

# pinned pyiqa ids. FR uses pyiqa for the non-faithfulness metrics.
NR_METRICS = ["niqe", "musiq", "maniqa-pipal", "clipiqa", "brisque", "pi", "topiq_nr",
              "hyperiqa", "tres", "dbcnn", "clipiqa+", "nima"]
FR_PYIQA = ["psnr", "ssim", "ms_ssim", "topiq_fr", "fsim", "gmsd", "vif"]  # + faithfulness(lpips/dists/dino)
_M = {}


def _metric(name, device):
    if name not in _M:
        _M[name] = pyiqa.create_metric(name, device=device)
    return _M[name]


def _load(path, device):
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def score_scale_nr(img_dir, names, device, skip=()):
    """NR panel over a scale dir. Returns (means, per_image). skip = metric ids to drop."""
    nr_list = [m for m in NR_METRICS if m not in skip]
    per = {}
    for n in names:
        fp = f"{img_dir}/{n}.png"
        if not os.path.exists(fp):
            continue
        x = _load(fp, device)
        row = {}
        for m in nr_list:
            try:
                with torch.no_grad():
                    row[m] = float(_metric(m, device)(x).item())
            except Exception as e:
                row[m] = float("nan")
                print(f"[full_eval] NR {m} failed on {n}: {type(e).__name__}")
        per[n] = row
    means = _means(per, nr_list)
    return means, per


def score_scale_fr(img_dir, scale_i, names, gtmap, device, full_fr, skip=()):
    """FR panel @ recursion i (only where GT window is real). Returns (means, per_image).
    skip = metric ids to drop (e.g. topiq_fr, which intermittently hangs on load for large sets)."""
    fr_list = [m for m in (FR_PYIQA if full_fr else ["psnr", "ssim"]) if m not in skip]
    per = {}
    for n in names:
        fp = f"{img_dir}/{n}.png"
        if not os.path.exists(fp) or n not in gtmap:
            continue
        crop, real, _ = gt_window(Image.open(gtmap[n]).convert("RGB"), scale_i)
        if not real:
            continue
        gen = Image.open(fp).convert("RGB")
        row = faithfulness(gen, crop, device)            # lpips, dists, dino_cos
        g = TF.to_tensor(gen).unsqueeze(0).to(device)
        r = TF.to_tensor(crop.resize(gen.size)).unsqueeze(0).to(device)
        for m in fr_list:
            try:
                row[m] = float(_metric(m, device)(g, r).item())
            except Exception as e:
                row[m] = float("nan")
                print(f"[full_eval] FR {m} failed on {n}: {type(e).__name__}")
        per[n] = row
    keys = ["lpips", "dists", "dino_cos"] + fr_list
    return _means(per, keys), per


def _means(per, keys):
    out = {}
    for k in keys:
        vals = [v[k] for v in per.values() if k in v and v[k] == v[k]]  # drop NaN
        if vals:
            out[k] = st.mean(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)   # parent of per-scale/scaleN
    ap.add_argument("--names_json", required=True)   # "<split.json>:<key>" e.g. opt1_split.json:val
    ap.add_argument("--gt_list", default="")         # paths.txt for FR (empty -> NR only, e.g. DIV2K)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full_fr", action="store_true", help="full FR panel (else psnr/ssim only)")
    ap.add_argument("--skip_metrics", default="", help="comma-list of metric ids to drop (e.g. "
                    "topiq_fr, which intermittently hangs on model-load for large sets)")
    ap.add_argument("--fid", action="store_true", help="also compute set-level FID @4x")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only_scale", type=int, default=0,
                    help="if >0, score ONLY this scale (1..4) + write to --out; lets a deep-eval be "
                         "chunked one-scale-per-process across nodes so the metric zoo (238 imgs x scale) "
                         "stays under the RAM ceiling that OOMs the all-4-scales run. 0 (default) = all 4.")
    a = ap.parse_args()

    sj, key = a.names_json.rsplit(":", 1)
    names = json.load(open(sj))[key]
    gtmap = {}
    if a.gt_list:
        gtmap = {os.path.basename(p)[:-4]: p for p in open(a.gt_list).read().split() if p}
    dev = a.device
    ps = f"{a.render_dir}/per-scale"

    # record the metric-library version: pyiqa internals have changed across releases,
    # so a scores file is not interpretable without it
    result = {"arm": a.arm, "method": a.method, "rank": a.rank, "evalset": a.evalset,
              "pyiqa_version": _pkg_version("pyiqa"),
              "scales": {}, "per_image": {}}
    skip = set(x for x in a.skip_metrics.split(",") if x)   # applies to BOTH NR + FR panels
    scales_to_do = (a.only_scale,) if a.only_scale else (1, 2, 3, 4)
    for scale_i in scales_to_do:
        d = f"{ps}/scale{scale_i}"
        if not os.path.isdir(d):
            continue
        nr_means, nr_per = score_scale_nr(d, names, dev, skip=skip)
        means = dict(nr_means)
        per = {n: dict(nr_per.get(n, {})) for n in nr_per}
        if gtmap:  # FR only where real GT exists (mainly scale1 for DIV8K)
            fr_means, fr_per = score_scale_fr(d, scale_i, names, gtmap, dev, a.full_fr, skip=skip)
            means.update(fr_means)
            for n, row in fr_per.items():
                per.setdefault(n, {}).update(row)
        result["scales"][str(scale_i)] = means
        result["per_image"][str(scale_i)] = per
        print(f"[full_eval] {a.arm} scale{scale_i}: {len(per)} imgs, {len(means)} metrics")

    # FID @4x (set-level) if requested + GT exists
    if a.fid and gtmap:
        try:
            import tempfile
            gt_dir = materialize_gt_from_map(gtmap, names, f"{a.render_dir}/_gtcrops", dev)
            fid = _metric("fid", dev)
            with tempfile.TemporaryDirectory() as tmp:
                valset = set(os.path.splitext(f)[0] for f in os.listdir(gt_dir))
                for f in os.listdir(f"{ps}/scale1"):
                    if os.path.splitext(f)[0] in valset:
                        os.symlink(os.path.join(f"{ps}/scale1", f), os.path.join(tmp, f))
                score = float(fid(tmp, gt_dir, mode="clean"))   # pyiqa FID = __call__(dir1,dir2)
            result["scales"].setdefault("1", {})["fid"] = score
            print(f"[full_eval] {a.arm} FID={score:.4f}")
        except Exception as e:
            print(f"[full_eval] FID failed: {type(e).__name__}: {e}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"))
    print(f"FULL_EVAL_DONE {a.arm} -> {a.out}")


def materialize_gt_from_map(gtmap, names, gt_out, device):
    """Real 4x GT crops (gt_window i=1) for FID. Reused across arms if already there."""
    os.makedirs(gt_out, exist_ok=True)
    for n in names:
        if n in gtmap and not os.path.exists(f"{gt_out}/{n}.png"):
            crop, real, _ = gt_window(Image.open(gtmap[n]).convert("RGB"), 1)
            if real:
                crop.save(f"{gt_out}/{n}.png")
    return gt_out


if __name__ == "__main__":
    main()
