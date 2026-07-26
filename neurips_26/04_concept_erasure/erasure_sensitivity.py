"""
Mean-erasure sensitivity analysis + bootstrap statistical test.

Two new experiments:
  1. SENSITIVITY: Sweep partial erasure strength alpha from 0.0 (baseline) to 1.0 (full
     erasure), to confirm the effect is graded and not an all-or-nothing artifact:
         I' = I - alpha * (I . hat{C_mean}) hat{C_mean}
  2. STATISTICAL TEST: Bootstrap p-value (N=10,000) for the headline mean-erasure
     improvement (Group Score 9.0% -> 31.0%) on CLIP-ViT-B/32.
"""
import os
import json
import warnings

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 10_000


def evaluate(v0, v1, t0, t1):
    s00 = (v0 @ t0).item()
    s01 = (v0 @ t1).item()
    s10 = (v1 @ t0).item()
    s11 = (v1 @ t1).item()
    t_ok = (s00 > s01) and (s11 > s10)
    i_ok = (s00 > s10) and (s11 > s01)
    return [int(t_ok), int(i_ok), int(t_ok and i_ok)]


def normalize(x, eps=1e-12):
    return x / (x.norm() + eps)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading Winoground ...")
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    n = len(dataset)

    # Pre-extract post-projection L2-normalized embeddings
    v0_list, v1_list, t0_list, t1_list = [], [], [], []
    print("Extracting embeddings ...")
    with torch.no_grad():
        for ex in tqdm(dataset):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            t_inputs = processor(text=[cap0, cap1], return_tensors="pt",
                                 padding=True, truncation=True).to(DEVICE)
            t_out = model.text_model(**t_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)

            i_inputs = processor(images=[img0, img1], return_tensors="pt").to(DEVICE)
            v_out = model.vision_model(**i_inputs)
            v_embeds = model.visual_projection(v_out.pooler_output)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)

            v0_list.append(v_embeds[0].cpu())
            v1_list.append(v_embeds[1].cpu())
            t0_list.append(t_embeds[0].cpu())
            t1_list.append(t_embeds[1].cpu())

    v0_all = torch.stack(v0_list).to(DEVICE)
    v1_all = torch.stack(v1_list).to(DEVICE)
    t0_all = torch.stack(t0_list).to(DEVICE)
    t1_all = torch.stack(t1_list).to(DEVICE)

    # ---- 1. SENSITIVITY SWEEP ----
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sensitivity = {}

    # Per-pair scores at full erasure (alpha=1) needed for the bootstrap below
    full_erasure_scores = None
    baseline_scores = None

    print("Sensitivity sweep over partial-erasure alphas ...")
    for alpha in alphas:
        per_pair_scores = []
        with torch.no_grad():
            for i in range(n):
                v0 = v0_all[i]; v1 = v1_all[i]
                t0 = t0_all[i]; t1 = t1_all[i]
                c_mean = (t0 + t1) / 2.0
                c_mean_hat = c_mean / (c_mean.norm() + 1e-12)

                # Partial erasure: remove alpha-fraction of the projection
                v0_e = v0 - alpha * (v0 @ c_mean_hat) * c_mean_hat
                v1_e = v1 - alpha * (v1 @ c_mean_hat) * c_mean_hat
                v0_e = normalize(v0_e)
                v1_e = normalize(v1_e)
                per_pair_scores.append(evaluate(v0_e, v1_e, t0, t1))
        arr = np.array(per_pair_scores)
        sensitivity[f"alpha_{alpha:.1f}"] = {
            "alpha": alpha,
            "text_score": float(arr[:, 0].mean()),
            "image_score": float(arr[:, 1].mean()),
            "group_score": float(arr[:, 2].mean()),
        }
        print(f"  alpha={alpha:.1f}: T={arr[:,0].mean():.4f} "
              f"I={arr[:,1].mean():.4f} G={arr[:,2].mean():.4f}")
        if alpha == 0.0:
            baseline_scores = arr
        if alpha == 1.0:
            full_erasure_scores = arr

    # ---- 2. BOOTSTRAP STATISTICAL TEST ----
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations vs baseline ...")
    base_g = baseline_scores[:, 2].astype(float)
    full_g = full_erasure_scores[:, 2].astype(float)
    base_t = baseline_scores[:, 0].astype(float)
    full_t = full_erasure_scores[:, 0].astype(float)
    base_i = baseline_scores[:, 1].astype(float)
    full_i = full_erasure_scores[:, 1].astype(float)

    diffs_g = full_g - base_g
    diffs_t = full_t - base_t
    diffs_i = full_i - base_i

    n_pairs = len(base_g)
    boots_g = np.empty(N_BOOTSTRAP)
    boots_t = np.empty(N_BOOTSTRAP)
    boots_i = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_pairs, n_pairs)
        boots_g[b] = diffs_g[idx].mean()
        boots_t[b] = diffs_t[idx].mean()
        boots_i[b] = diffs_i[idx].mean()

    # 95% percentile CI on the difference
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    bootstrap_results = {
        "n_bootstrap": N_BOOTSTRAP,
        "n_pairs": int(n_pairs),
        "group_score": {
            "baseline": float(base_g.mean()),
            "erasure": float(full_g.mean()),
            "mean_diff": float(diffs_g.mean()),
            "ci_95_diff": ci(boots_g),
            "p_value_one_sided": float(np.mean(boots_g <= 0.0)),
        },
        "text_score": {
            "baseline": float(base_t.mean()),
            "erasure": float(full_t.mean()),
            "mean_diff": float(diffs_t.mean()),
            "ci_95_diff": ci(boots_t),
            "p_value_one_sided": float(np.mean(boots_t <= 0.0)),
        },
        "image_score": {
            "baseline": float(base_i.mean()),
            "erasure": float(full_i.mean()),
            "mean_diff": float(diffs_i.mean()),
            "ci_95_diff": ci(boots_i),
            "p_value_one_sided": float(np.mean(boots_i <= 0.0)),
        },
    }

    out = {
        "model": "openai/clip-vit-base-patch32",
        "sensitivity_sweep": sensitivity,
        "bootstrap_test": bootstrap_results,
    }

    print("\n" + json.dumps(out, indent=2))
    out_path = os.path.join(os.path.dirname(__file__), "..", "05_results",
                            "erasure_sensitivity.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
