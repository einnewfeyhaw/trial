"""
E6: Spectral bridge between the global SVD analysis and instance-specific erasure.

The paper's two halves are currently disconnected: Section 3-4 characterizes a
GLOBAL spectral property (compositional importance vs singular value), while
mean-erasure removes a PAIR-SPECIFIC direction. Reviewer v8Kz and the AC both
flagged that nothing links them.

This script closes that gap. For every Winoground pair it computes:

  (a) spectral placement of C_mean and Delta -- the fraction of each vector's
      energy sitting in the top-20% vs bottom-20% singular dimensions;
  (b) the per-pair erasure OUTCOME, both as a binary flip (fail -> pass) and as
      a continuous margin change, which is far more statistically powerful than
      the binary score;
  (c) the correlation between (a) and (b).

The paper's hypothesis predicts: C_mean concentrates in HIGH-sigma dims, Delta
concentrates in LOW-sigma dims, and pairs where Delta sits lowest should gain
MOST from erasure. If (c) comes back null, the spectral story and the erasure
story are independent mechanisms and the paper must say so explicitly.

Outputs spectral_placement_results.json.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pointbiserialr, spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def get_projection_matrix(model, model_name):
    if getattr(model, "visual_projection", None) is not None:
        return model.visual_projection.weight.detach().cpu().numpy()
    if "siglip" in model_name.lower():
        return model.vision_model.head.attention.out_proj.weight.detach().cpu().numpy()
    return None


def scores(v0, v1, t0, t1):
    s00 = (v0 @ t0).item()
    s01 = (v0 @ t1).item()
    s10 = (v1 @ t0).item()
    s11 = (v1 @ t1).item()
    return s00, s01, s10, s11


def eval_pair(v0, v1, t0, t1):
    """Winoground text/image/group pass flags plus continuous margins."""
    s00, s01, s10, s11 = scores(v0, v1, t0, t1)
    text = (s00 > s01) and (s11 > s10)
    image = (s00 > s10) and (s11 > s01)
    # margin > 0 exactly when the corresponding score passes
    text_margin = min(s00 - s01, s11 - s10)
    image_margin = min(s00 - s10, s11 - s01)
    return {
        "text": int(text),
        "image": int(image),
        "group": int(text and image),
        "text_margin": text_margin,
        "image_margin": image_margin,
    }


def energy_fractions(vec_svd, top_cut, bot_cut):
    """Fraction of L1 energy in the top-20% and bottom-20% singular dims."""
    a = vec_svd.abs()
    total = a.sum().item()
    if total == 0:
        return 0.0, 0.0
    return (
        a[:top_cut].sum().item() / total,
        a[bot_cut:].sum().item() / total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default="spectral_placement_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Consistent with the rest of the repo; the ungated mirror of Winoground.
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")

    W = get_projection_matrix(model, args.model)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    U_t = torch.tensor(U.T).float().to(device)

    d = len(S)
    top_cut = int(0.2 * d)
    bot_cut = int(0.8 * d)
    is_siglip = "siglip" in args.model.lower()

    rows = []
    with torch.no_grad():
        for ex in tqdm(dataset, desc="pairs"):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            kwargs = {"padding": "max_length"} if is_siglip else {"padding": True}
            inputs = processor(
                text=[cap0, cap1],
                images=[img0, img1],
                return_tensors="pt",
                truncation=True,
                **kwargs,
            ).to(device)
            out = model(**inputs)

            t = out.text_embeds
            v = out.image_embeds
            t = t / t.norm(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True)
            t0, t1 = t[0], t[1]
            v0, v1 = v[0], v[1]

            c_mean = 0.5 * (t0 + t1)
            delta = 0.5 * (t0 - t1)

            # spectral placement
            cm_top, cm_bot = energy_fractions(c_mean @ U_t.T, top_cut, bot_cut)
            dl_top, dl_bot = energy_fractions(delta @ U_t.T, top_cut, bot_cut)

            base = eval_pair(v0, v1, t0, t1)

            # mean-erasure (text-side), matching 04_concept_erasure/concept_erasure_eval.py
            cm_hat = c_mean / c_mean.norm()
            v0e = v0 - (v0 @ cm_hat) * cm_hat
            v1e = v1 - (v1 @ cm_hat) * cm_hat
            v0e = v0e / v0e.norm()
            v1e = v1e / v1e.norm()
            erased = eval_pair(v0e, v1e, t0, t1)

            rows.append(
                {
                    "cmean_top_frac": cm_top,
                    "cmean_bot_frac": cm_bot,
                    "delta_top_frac": dl_top,
                    "delta_bot_frac": dl_bot,
                    "base_group": base["group"],
                    "erased_group": erased["group"],
                    "base_image_margin": base["image_margin"],
                    "erased_image_margin": erased["image_margin"],
                    "image_margin_gain": erased["image_margin"] - base["image_margin"],
                    "flipped_to_pass": int(
                        erased["group"] == 1 and base["group"] == 0
                    ),
                }
            )

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    # --- descriptive: where do C_mean and Delta live? ---
    placement = {
        "cmean_energy_in_top20_sigma": float(col("cmean_top_frac").mean()),
        "cmean_energy_in_bottom20_sigma": float(col("cmean_bot_frac").mean()),
        "delta_energy_in_top20_sigma": float(col("delta_top_frac").mean()),
        "delta_energy_in_bottom20_sigma": float(col("delta_bot_frac").mean()),
        "uniform_baseline": 0.2,
    }

    # --- the bridge: does placement predict erasure benefit? ---
    gain = col("image_margin_gain")
    flip = col("flipped_to_pass")

    def sp(x, y):
        r, p = spearmanr(x, y)
        return {"rho": float(r), "p_value": float(p)}

    def pb(binary, x):
        # point-biserial needs both classes present
        if len(np.unique(binary)) < 2:
            return {"r": None, "p_value": None}
        r, p = pointbiserialr(binary, x)
        return {"r": float(r), "p_value": float(p)}

    bridge = {
        "margin_gain_vs_delta_bottom_frac": sp(col("delta_bot_frac"), gain),
        "margin_gain_vs_delta_top_frac": sp(col("delta_top_frac"), gain),
        "margin_gain_vs_cmean_top_frac": sp(col("cmean_top_frac"), gain),
        "flip_vs_delta_bottom_frac": pb(flip, col("delta_bot_frac")),
        "flip_vs_cmean_top_frac": pb(flip, col("cmean_top_frac")),
    }

    results = {
        "model": args.model,
        "n_pairs": len(rows),
        "baseline_group_score": float(col("base_group").mean()),
        "erased_group_score": float(col("erased_group").mean()),
        "n_flipped_to_pass": int(flip.sum()),
        "spectral_placement": placement,
        "placement_vs_gain": bridge,
    }

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": results, "per_pair": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
