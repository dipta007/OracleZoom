"""Download the training data from HuggingFace and build what the trainer reads.

  uv run scripts/prepare_data.py --tier 1k --out data/train_1k

The dataset ships the high-resolution image and its cached 4x prompt. It does NOT ship the
low-quality input, because that is derived. This script rebuilds it with exactly the geometry the
zoom loop uses at test time, so training and evaluation see the same kind of input:

    hr  --resize_and_center_crop(512)-->  canvas
    canvas --centre 1/4 crop--> 128  --bicubic--> 512   = the blind input
    hr  --centre min(W,H)/4 square--> --lanczos--> 512  = the 4x target

Output layout (what --cache and --val_cache expect):
    <out>/hr/<name>.png       the high-resolution image
    <out>/blind/<name>.png    the low-quality 512 input
    <out>/pairs.json          {name: {blind_input, gt, prompt, source, upscale}}
    <out>/train_list.txt      one hr path per line

The dataset is public; an HF token is optional.
"""
import argparse, json, os, sys

from PIL import Image

# resize_and_center_crop lives in the Chain-of-Zoom submodule. Reuse it rather than
# reimplementing, so the crop geometry cannot drift from the zoom loop.
_COZ = os.path.join(os.path.dirname(__file__), "..", "ref", "coz")
if os.path.isdir(_COZ) and _COZ not in sys.path:
    sys.path.insert(0, os.path.abspath(_COZ))

PROCESS_SIZE = 512
UPSCALE = 4


def _resize_and_center_crop(img, size):
    try:
        from inference_coz import resize_and_center_crop
        return resize_and_center_crop(img, size)
    except ImportError:
        # same maths, used when the submodule is not checked out
        w, h = img.size
        s = size / min(w, h)
        nw, nh = int(w * s), int(h * s)
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - size) // 2, (nh - size) // 2
        return img.crop((left, top, left + size, top + size))


def blind_input(hr):
    """The input the model actually gets at 4x: no ground truth in it."""
    canvas = _resize_and_center_crop(hr, PROCESS_SIZE)
    w, h = canvas.size
    nw, nh = w // UPSCALE, h // UPSCALE
    crop = canvas.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
    return crop.resize((w, h), Image.BICUBIC)


def build(split_rows, out):
    os.makedirs(f"{out}/hr", exist_ok=True)
    os.makedirs(f"{out}/blind", exist_ok=True)
    pairs, listed, skipped = {}, [], 0
    for i, row in enumerate(split_rows):
        name = row["file_name"]
        prompt = (row.get("prompt") or "").strip()
        if not prompt:                       # the trainer needs the cached 4x prompt
            skipped += 1
            continue
        hr = row["image"].convert("RGB")
        hp, bp = f"{out}/hr/{name}.png", f"{out}/blind/{name}.png"
        hr.save(hp)
        blind_input(hr).save(bp)
        pairs[name] = {"blind_input": os.path.abspath(bp), "gt": os.path.abspath(hp),
                       "prompt": prompt, "source": "gt", "upscale": UPSCALE}
        listed.append(os.path.abspath(hp))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1} images", flush=True)
    json.dump(pairs, open(f"{out}/pairs.json", "w"))
    open(f"{out}/train_list.txt", "w").write("\n".join(listed) + "\n")
    print(f"  wrote {len(pairs)} pairs -> {out}" + (f"  ({skipped} skipped: no prompt)" if skipped else ""))
    return len(pairs)


def build_eval(split_rows, out, limit=0):
    """Write the layout scripts/eval_final.sh needs: gt/, names.json, gt.txt.

    This lets you score on the dataset's own validation split with no external downloads.
    The seven test sets in the paper (DIV2K, DIV8K, DRealSR, RealSR, Flickr2K, FFHQ, 4KLSDB) are
    third-party data you must obtain yourself; use scripts/prep_eval_datasets.py for those.
    """
    os.makedirs(f"{out}/gt", exist_ok=True)
    names, paths = [], []
    rows = split_rows.select(range(min(limit, len(split_rows)))) if limit else split_rows
    for row in rows:
        name = row["file_name"]
        p = f"{out}/gt/{name}.png"
        row["image"].convert("RGB").save(p)
        names.append(name)
        paths.append(os.path.abspath(p))
    json.dump({"all": names}, open(f"{out}/names.json", "w"))
    open(f"{out}/gt.txt", "w").write("\n".join(paths) + "\n")
    print(f"  wrote {len(names)} images -> {out}  (gt/, names.json, gt.txt)")
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dipta007/OracleZoom-4KLSDB-train")
    ap.add_argument("--tier", default="1k", choices=["1k", "4k", "16k", "all"],
                    help="how many training images. The paper uses 1k.")
    ap.add_argument("--out", default="data/train", help="written as <out> and <out>_val")
    ap.add_argument("--val_limit", type=int, default=0, help="0 = the whole validation split")
    ap.add_argument("--eval_layout", default="",
                    help="also write an eval-shaped copy of the validation split here, so "
                         "scripts/eval_final.sh runs with no external downloads.")
    ap.add_argument("--eval_limit", type=int, default=200,
                    help="how many validation images to write for --eval_layout. 0 = all.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets is missing; run uv sync from the project root")

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"downloading {args.repo} [{args.tier}] ...", flush=True)
    ds = load_dataset(args.repo, args.tier, token=tok)

    print("train split:")
    n_tr = build(ds["train"], args.out)
    val = ds["validation"]
    if args.val_limit:
        val = val.select(range(min(args.val_limit, len(val))))
    print("validation split:")
    n_va = build(val, f"{args.out}_val")

    n_ev = 0
    if args.eval_layout:
        print("eval layout:")
        n_ev = build_eval(ds["validation"], args.eval_layout, args.eval_limit)

    print(f"\nDONE  train={n_tr}  val={n_va}" + (f"  eval={n_ev}" if n_ev else ""))
    print("\ntrain with:")
    print(f"  scripts/train_final.sh {args.out} {args.out}/train_list.txt {args.out}_val out/run1")
    if args.eval_layout:
        print("score with:")
        print(f"  scripts/eval_final.sh {args.eval_layout} renders/ours results/ours")


if __name__ == "__main__":
    main()
