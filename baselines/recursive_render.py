"""Model-agnostic recursive-zoom renderer for baseline SR models (task #37).

Replicates oracle_infer's crop-recursion EXACTLY so the output drops into our UNMODIFIED
downstream eval (full_eval):
  - scale0 = resize_and_center_crop(gt, process_size)      (x1 base, same as oracle_infer)
  - per recursion: crop the center 1/upscale of the current image -> model native x4 SR -> save
  - layout: <out>/per-scale/scale{0..4}/<name>.png                  (identical to oracle_infer)

The ONLY difference vs oracle_infer: the per-recursion SR step is a baseline model's native x4
SR (option A), with NO VLM / NO privileged prompt (blind) -- the fair no-OPD baseline at extreme
zoom. FR@scale-i in full_eval uses gt_window(gt, i) on the SAME crop geometry, so it aligns.

ISOLATION: does not import/modify src/opd_zoom eval code. Reuses ref/coz resize_and_center_crop
read-only for identical crop geometry (same as oracle_infer does).

Usage (box):
  python baselines/recursive_render.py --model seesr --ckpt $CK/baselines/ckpts/seesr \
    --gt_dir $CK/data/eval/4klsdb/gt --out $E/gen_4klsdb_seesr_deep --rec_num 4
"""
import argparse
import glob
import importlib
import os
import sys

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)         # so `import adapters.<model>` resolves


def resize_and_center_crop(img, size):
    """Inlined verbatim from ref/coz/inference_coz.py (identical crop geometry to oracle_infer).
    Copied (not imported) so baselines don't pull CoZ's heavy module deps (peft/diffusers) into
    a baseline's own environment (some lack peft). Isolation-safe."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _save(out_dir, scale, bname, pil):
    d = os.path.join(out_dir, "per-scale", f"scale{scale}")
    os.makedirs(d, exist_ok=True)
    pil.save(os.path.join(d, f"{bname}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter name in baselines/adapters/")
    ap.add_argument("--ckpt", required=True, help="ckpt dir ($CK/baselines/ckpts/<model>)")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True, help="render dir (E/gen_<ds>_<model>_deep)")
    ap.add_argument("--rec_num", type=int, default=4)
    ap.add_argument("--process_size", type=int, default=512)   # match oracle_infer default
    ap.add_argument("--upscale", type=int, default=4)          # 4 = CoZ default
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="j/N: this proc renders images[j::N] (data-parallel "
                    "over GPUs; each shard writes disjoint per-name files into the SAME out dir -> safe merge)")
    a = ap.parse_args()

    j, N = (int(x) for x in a.shard.split("/"))

    Adapter = importlib.import_module(f"adapters.{a.model}").Adapter
    sr = Adapter(a.ckpt, a.device)

    imgs = sorted(p for e in ("*.png", "*.jpg", "*.jpeg") for p in glob.glob(f"{a.gt_dir}/{e}"))
    imgs = imgs[j::N]                                          # round-robin shard (no overlap, no gaps)
    print(f"[{a.model}] shard {j}/{N}: {len(imgs)} images, rec_num={a.rec_num}, native x4 blind", flush=True)
    for k, img_path in enumerate(imgs):
        bname = os.path.splitext(os.path.basename(img_path))[0]
        gt = Image.open(img_path).convert("RGB")
        cur = resize_and_center_crop(gt, a.process_size)        # scale0 (x1 base)
        _save(a.out, 0, bname, cur)
        for rec in range(a.rec_num):
            w, h = cur.size
            nw, nh = w // a.upscale, h // a.upscale
            crop = cur.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
            out = sr.sr_x4(crop)                                # native x4 SR, blind
            if out.size != (w, h):                              # normalize to process_size
                out = out.resize((w, h), Image.BICUBIC)
            _save(a.out, rec + 1, bname, out)
            cur = out
        print(f"[{a.model}] {k + 1}/{len(imgs)} {bname} done", flush=True)


if __name__ == "__main__":
    main()
