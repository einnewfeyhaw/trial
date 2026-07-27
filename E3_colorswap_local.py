"""
E3_colorswap_local.py — Run E3's exact protocol on ColorSwap (local files).

This wrapper imports E3_second_2x2's core functions without modification.
The scorer, conditions, bootstrap CI, and JSON output format are identical.
The only difference is how the dataset is loaded (from local JSON+images
instead of HuggingFace Hub), because stanfordnlp/colorswap is gated and
the token does not have approved access.

ColorSwap (Burapacheep et al. 2024) is a Winoground-format 2x2 benchmark:
  - 1,000 examples (train=700, test=300)
  - Two captions per group with the SAME words, colour words swapped
  - Two AI-generated images that match each caption
  - Exactly the 2x2 structure mean-erasure addresses

Data downloaded from:
  https://drive.google.com/file/d/1xdG94DQdz_eQVH1lrEeaHVz_BNkrVgb5
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# ── Import E3's canonical scoring/analysis functions (UNMODIFIED) ───────────
sys.path.insert(0, os.path.dirname(__file__))
from E3_second_2x2 import (
    winoground_scores,
    erase,
    run_conditions,
    bootstrap_delta,
    MODELS,
)

# ── Dataset loading ──────────────────────────────────────────────────────────

def load_colorswap(data_dir: str, split: str):
    """
    Load ColorSwap from the local extracted directory.
    Returns a list of dicts with keys image_1, image_2, caption_1, caption_2.
    image_* values are PIL.Image objects (RGB).
    """
    json_path = os.path.join(data_dir, f"{split}.json")
    img_dir   = os.path.join(data_dir, "images")
    with open(json_path) as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        fname1 = os.path.basename(item["image_1"])
        fname2 = os.path.basename(item["image_2"])
        p1 = os.path.join(img_dir, fname1)
        p2 = os.path.join(img_dir, fname2)
        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"  MISSING: {p1} or {p2} — skipping")
            continue
        rows.append({
            "image_1": Image.open(p1).convert("RGB"),
            "image_2": Image.open(p2).convert("RGB"),
            "caption_1": item["caption_1"],
            "caption_2": item["caption_2"],
        })
    print(f"  Loaded {len(rows)} pairs from ColorSwap ({split})")
    return rows


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_all_local(model_name, rows, device, max_pairs):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    is_siglip = "siglip" in model_name.lower()

    V0, V1, T0, T1 = [], [], [], []
    with torch.no_grad():
        for i, ex in enumerate(tqdm(rows, desc="  encoding")):
            if i >= max_pairs:
                break
            try:
                i0 = ex["image_1"]
                i1 = ex["image_2"]
                c0 = str(ex["caption_1"])
                c1 = str(ex["caption_2"])
            except Exception as e:
                print(f"  skip {i}: {e}")
                continue

            kwargs = {"padding": "max_length"} if is_siglip else {"padding": True}
            inputs = processor(
                text=[c0, c1], images=[i0, i1],
                return_tensors="pt", truncation=True, **kwargs,
            ).to(device)
            out = model(**inputs)

            t = out.text_embeds
            v = out.image_embeds
            t = t / t.norm(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True)

            V0.append(v[0].cpu())
            V1.append(v[1].cpu())
            T0.append(t[0].cpu())
            T1.append(t[1].cpu())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(V0), torch.stack(V1), torch.stack(T0), torch.stack(T1)


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyze(model_name, rows, device, args):
    print(f"\n=== {model_name} ===")
    V0, V1, T0, T1 = encode_all_local(model_name, rows, device, args.max_pairs)
    n = len(V0)
    print(f"  encoded {n} pairs")

    conds = run_conditions(V0, V1, T0, T1)
    base  = conds["baseline"]

    res = {"model": model_name, "n_pairs": int(n), "conditions": {}}
    for name, arr in conds.items():
        entry = {
            "text_score":  float(arr[:, 0].mean()),
            "image_score": float(arr[:, 1].mean()),
            "group_score": float(arr[:, 2].mean()),
        }
        if name != "baseline":
            entry["group_vs_baseline"] = bootstrap_delta(
                base, arr, col=2, n_boot=args.n_boot)
            g0 = base[:, 2].mean()
            entry["relative_gain"] = (
                float(arr[:, 2].mean() / g0) if g0 > 0 else None)
        res["conditions"][name] = entry

    print(json.dumps(res, indent=2))
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/colorswap/data",
                    help="Path to the extracted ColorSwap data/ directory")
    ap.add_argument("--split",    default="test")
    ap.add_argument("--models",   nargs="*", default=MODELS)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--n-boot",   type=int, default=10000)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"dataset: ColorSwap (local, split={args.split})")
    print(f"data_dir: {args.data_dir}")

    rows = load_colorswap(args.data_dir, args.split)

    out_path = args.out or "E3_colorswap_local.json"
    results = {
        "dataset": "stanfordnlp/colorswap (local)",
        "split": args.split,
        "source": "https://drive.google.com/file/d/1xdG94DQdz_eQVH1lrEeaHVz_BNkrVgb5",
        "note": "stanfordnlp/colorswap is gated on HF Hub; data downloaded from the paper's public Google Drive link. Scoring functions imported unmodified from E3_second_2x2.py.",
        "models": {}
    }

    for name in args.models:
        try:
            r = analyze(name, rows, device, args)
            results["models"][name] = r
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  checkpoint saved -> {out_path}")
        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print(f"FAILED {name}: {e}\n{msg}")
            results["models"][name] = {"error": str(e), "traceback": msg}
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
