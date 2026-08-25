"""Prep the 6 generalization eval datasets into ONE uniform layout the existing
CoZ synthetic-deep eval consumes unchanged (oracle_infer --gt_dir + full_eval).

Per dataset writes under $CK/data/eval/<ds>/:
  gt/*.png     symlinks (or decoded, for ffhq parquet) to HR images, min-side >= 512
  names.json   {"all": [basename, ...]}          -> full_eval --names_json <..>:all
  gt.txt       one full HR path per line          -> full_eval --gt_list (FR@4x)

Protocol = same as DIV8K/DIV2K: oracle_infer center-crops each HR to 512, recurses x4
(4x/16x/64x/256x). FR@4x compares scale1 vs the 512 center-crop of HR; NR@all scales.
This is the SYNTHETIC-DEEP path (uniform, zero renderer change). RealSR/DRealSR native-x4
(real LR->HR) is a separate mode (needs an oracle_infer real-LR flag) -- NOT built here.

Registry kinds:
  flat   : glob of HR .png, sample N even-spaced (flickr2k, div8k)
  pairs  : RealSR-style */Test/<s>/*_HR.png (use HR only; synthetic deep on real photos)
  hrdir  : DRealSR Test_x4/test_HR/*.png
  parquet: HF image parquet shards (ffhq) -> decode N to real .png files

Usage:
  python scripts/prep_eval_datasets.py [ds ...] [--n 200]

Set OPDZ_ROOT to wherever you keep the datasets (defaults to ./ ).
Each dataset must sit under $OPDZ_ROOT/data/ as shown in REG below.
"""
import argparse, glob, io, json, os

CK = os.environ.get("OPDZ_ROOT", ".")
D = f"{CK}/data"
MIN_SIDE = 512  # 512 center-crop must be real detail, not an upscale

# name -> (kind, glob-or-dir, default_n). default_n=0 => use all.
REG = {
    "div8k_238":  ("flat",    f"{D}/div8ktest_dir/*.png",                       0),
    "div8k_1238": ("flat",    f"{D}/div8k_full_dir/*.png",                      0),
    "flickr2k":   ("flat",    f"{D}/flickr2k/Flickr2K/*.png",                 200),
    "drealsr":    ("hrdir",   f"{D}/drealsr/Test_x4/test_HR/*.png",             0),
    "realsr":     ("pairs",   f"{D}/realsr/RealSR(V3)/*/Test/4/*_HR.png",       0),
    "ffhq":       ("parquet", f"{D}/FFHQ512_hf/data/*.parquet",               200),
    # 4KLSDB test (Phase 8): HR pulled by scripts/pull_4klsdb_test.py (photo-filtered, deduped).
    # Native 4K (min_side>=3840) -> real GT at 4x AND 6x. IN-DISTRIBUTION once we train on 4KLSDB.
    "4klsdb":     ("flat",    f"{D}/4klsdb/hr_test/*.png",                      0),
}


def even_sample(items, n):
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def min_side_ok(path):
    from PIL import Image
    try:
        w, h = Image.open(path).size
        return min(w, h) >= MIN_SIDE
    except Exception:
        return False


def prep_flat_like(g, n):
    """flat / pairs / hrdir all reduce to: sorted png glob, filter min-side, even-sample."""
    files = sorted(glob.glob(g))
    ok = [f for f in files if min_side_ok(f)]
    dropped = len(files) - len(ok)
    return even_sample(ok, n), len(files), dropped


def prep_parquet(g, n, out_gt):
    """Decode up to n HR images from HF image parquet (>=512) into out_gt/*.png."""
    import pyarrow.parquet as pq
    from PIL import Image
    shards = sorted(glob.glob(g))
    written = []
    idx = 0
    for sh in shards:
        if len(written) >= n:
            break
        t = pq.read_table(sh)
        col = "image" if "image" in t.column_names else t.column_names[0]
        for cell in t.column(col):
            if len(written) >= n:
                break
            v = cell.as_py()
            b = v["bytes"] if isinstance(v, dict) else v
            im = Image.open(io.BytesIO(b)).convert("RGB")
            if min(im.size) < MIN_SIDE:
                continue
            name = f"ffhq_{idx:05d}.png"
            p = os.path.join(out_gt, name)
            im.save(p)
            written.append(p)
            idx += 1
    return written


def build(ds, n_override):
    kind, g, dflt = REG[ds]
    root = f"{D}/eval/{ds}"
    gt_dir = f"{root}/gt"
    os.makedirs(gt_dir, exist_ok=True)
    # clear stale symlinks/files from a prior prep
    for f in os.listdir(gt_dir):
        p = os.path.join(gt_dir, f)
        if os.path.islink(p) or os.path.isfile(p):
            os.remove(p)
    n = n_override if n_override is not None else dflt

    if kind == "parquet":
        picked = prep_parquet(g, n if n else 200, gt_dir)
        total, dropped = len(picked), 0
        gt_paths = picked  # real files, already in gt_dir
    else:
        picked, total, dropped = prep_flat_like(g, n)
        gt_paths = []
        seen = set()
        for src in picked:
            base = os.path.basename(src)
            # RealSR HR names collide across Canon/Nikon+scale dirs -> disambiguate
            if base in seen:
                stem = os.path.dirname(src).replace("/", "_").split("RealSR")[-1].strip("_")
                base = f"{stem}_{base}"
            seen.add(base)
            link = os.path.join(gt_dir, base)
            if not os.path.exists(link):
                os.symlink(src, link)
            gt_paths.append(src)  # gt.txt points at the REAL source (full HR)

    names = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".png"))
    with open(f"{root}/names.json", "w") as fh:
        json.dump({"all": names}, fh)
    with open(f"{root}/gt.txt", "w") as fh:
        fh.write("\n".join(gt_paths) + "\n")
    print(f"{ds:12s} kind={kind:8s} picked={len(names):4d}  (src_total={total}, dropped_small={dropped})  -> {gt_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(REG), help="subset; default all 6")
    ap.add_argument("--n", type=int, default=None, help="override sample count (0=all)")
    a = ap.parse_args()
    for ds in (a.datasets or list(REG)):
        if ds not in REG:
            print(f"!! unknown dataset {ds}; known: {list(REG)}"); continue
        build(ds, a.n)
