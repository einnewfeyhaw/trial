"""
E15: null control for E7/E2's swap_obj access-ratio measure (0.484 / 0.47).

Motivation: the reply currently argues that 0.484 being measurable "this
precisely" implies real discriminative signal. That inference is invalid --
precision does not imply non-null. There is also a concrete mechanism by
which a ratio below 1.0 could arise with NO real signal: we select "the
most discriminative 5% of dimensions" by AUC computed on the same n=245
held-out data we then measure cosine mass on. Low-true-signal dimensions
produce noisier AUC estimates, so they can enter the top-5% set by chance
more often, and if those noise-selected dims also happen to carry less
cosine mass on average, the selection procedure alone -- with zero real
match/mismatch signal -- could produce a ratio below 1.0.

This tests that directly. Uses the IDENTICAL feature extraction and
access-ratio computation as E7_structural_variance.py (same model, same
subset, same top-5%-by-AUC selection rule, same |contribution| convention),
so the null is apples-to-apples with the reported 0.484.

Procedure, repeated N_NULL trials:
  1. Randomly shuffle the match/mismatch labels across the pooled
     2*n_pairs samples (breaks any true relationship between a dimension's
     value and whether it belongs to the true or foil caption, while
     preserving the exact same finite-sample size/noise structure).
  2. Recompute per-dimension AUC discriminability under the shuffled labels.
  3. Re-select the top-5% most "discriminative" dims under THIS shuffle's
     own AUC ranking (not the real ranking -- the selection step itself is
     part of what's being tested for bias).
  4. Compute cosine mass using the shuffled "matched" class assignment,
     and the access ratio for the shuffle's own top-5% selection.
Report the distribution of null ratios (mean, sd, 95% range) and compare
directly to the real point estimate (0.484) and CI ([0.448, 0.530]).

Output: E15_results.json
"""

import json
import os
import sys
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(__file__))
from E2_direct_spectral import subset_of

SUBSET_FILTER = "swap_obj"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_NULL = 1000          # null trials (cheap: relabel + recompute, no re-extraction)
TOP_FRAC_ACCESS = 0.05  # matches E7's "top-5% most discriminative"


def normalize(x, eps=1e-12):
    n = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / n


def compute_access_ratio_abs(mc, top_idx, d_out, k_access):
    """Same convention as E7: |contribution| before summing, divided by
    the uniform-baseline fraction k_access/d_out."""
    mc_abs = np.abs(mc)
    mc_total = mc_abs.sum()
    if mc_total <= 0:
        return np.nan
    mc_topdisc = mc_abs[top_idx].sum()
    frac_topdisc = mc_topdisc / mc_total
    return frac_topdisc / (k_access / d_out)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    W = model.visual_projection.weight.detach().cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    d_out = len(S)
    U_t = torch.tensor(U.T, dtype=torch.float32).to(DEVICE)

    print("Loading haideraltahan/wds_sugarcrepe ...")
    full_dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")
    LIMIT = 500
    filtered_indices = []
    n_scanned = 0
    for i, ex in enumerate(tqdm(full_dataset, desc="filtering to swap_obj")):
        n_scanned = i + 1
        s = subset_of(ex)
        if s is not None and SUBSET_FILTER in s:
            filtered_indices.append(i)
            if len(filtered_indices) >= LIMIT:
                break
    dataset = full_dataset.select(filtered_indices)
    print(f"matched {len(filtered_indices)} swap_obj examples out of {n_scanned} scanned")

    match_contrib, mismatch_contrib = [], []
    n_pairs = 0
    print("Extracting per-dimension contributions ...")
    with torch.no_grad():
        for ex in tqdm(dataset):
            img = ex["0.webp"].convert("RGB")
            true_cap, false_cap = ex["npy"][0], ex["npy"][1]

            t_in = processor(text=[true_cap, false_cap], return_tensors="pt",
                              padding=True, truncation=True).to(DEVICE)
            t_out = model.text_model(**t_in)
            t_emb = normalize(model.text_projection(t_out.pooler_output))

            v_in = processor(images=[img], return_tensors="pt").to(DEVICE)
            v_out = model.vision_model(**v_in)
            v_emb = normalize(model.visual_projection(v_out.pooler_output))

            v_svd = (v_emb @ U_t.T)[0]
            t_match_svd = (t_emb[0] @ U_t.T)
            t_mismatch_svd = (t_emb[1] @ U_t.T)

            match_contrib.append((v_svd * t_match_svd).cpu().numpy())
            mismatch_contrib.append((v_svd * t_mismatch_svd).cpu().numpy())
            n_pairs += 1

    match_contrib = np.array(match_contrib)
    mismatch_contrib = np.array(mismatch_contrib)
    pool = np.concatenate([match_contrib, mismatch_contrib], axis=0)  # (2n, d_out)
    print(f"Extracted {n_pairs} pairs.")

    k_access = max(1, int(np.round(TOP_FRAC_ACCESS * d_out)))
    true_labels = np.ones(2 * n_pairs)
    true_labels[n_pairs:] = 0

    # ── Real (non-null) point estimate, exactly matching E7's procedure ──
    auc_per_dim_real = np.zeros(d_out)
    for dim in range(d_out):
        auc_per_dim_real[dim] = abs(roc_auc_score(true_labels, pool[:, dim]) - 0.5)
    real_top_idx = np.argsort(auc_per_dim_real)[::-1][:k_access]
    cosine_mass_real = match_contrib.mean(axis=0)
    real_access_ratio = compute_access_ratio_abs(cosine_mass_real, real_top_idx, d_out, k_access)
    print(f"\nReal access ratio (reference, should match E7's 0.484): {real_access_ratio:.4f}")

    # ── Null trials: shuffle labels, re-select top-5% under the shuffle's
    #    own AUC ranking, compute access ratio using the shuffle's own
    #    "matched" class assignment ──
    null_ratios = np.empty(N_NULL)
    print(f"\nRunning {N_NULL} null trials (label shuffle) ...")
    for trial in tqdm(range(N_NULL)):
        shuffled = rng.permutation(true_labels)
        auc_null = np.zeros(d_out)
        for dim in range(d_out):
            auc_null[dim] = abs(roc_auc_score(shuffled, pool[:, dim]) - 0.5)
        null_top_idx = np.argsort(auc_null)[::-1][:k_access]

        mc_null = pool[shuffled == 1].mean(axis=0)
        null_ratios[trial] = compute_access_ratio_abs(mc_null, null_top_idx, d_out, k_access)

    null_mean = float(np.mean(null_ratios))
    null_sd = float(np.std(null_ratios, ddof=1))
    null_ci_lo = float(np.percentile(null_ratios, 2.5))
    null_ci_hi = float(np.percentile(null_ratios, 97.5))
    # one-sided: fraction of null trials at or below the real point estimate
    p_le_real = float(np.mean(null_ratios <= real_access_ratio))

    print(f"\n=== NULL DISTRIBUTION (N={N_NULL}) ===")
    print(f"  mean = {null_mean:.4f}, sd = {null_sd:.4f}")
    print(f"  95% range = [{null_ci_lo:.4f}, {null_ci_hi:.4f}]")
    print(f"  real point estimate = {real_access_ratio:.4f} (E7 reference: 0.484)")
    print(f"  P(null ratio <= real estimate) = {p_le_real:.4f}")

    verdict = ("CLEAN -- null centers near 1.0, real estimate sits outside null range"
               if null_mean > 0.9 and real_access_ratio < null_ci_lo
               else "ARTIFACT RISK -- null distribution overlaps or is itself below 1.0; "
                    "do not report 0.484 as evidence without addressing this")
    print(f"\nVERDICT: {verdict}")

    out = {
        "n_pairs": int(n_pairs),
        "d_out": int(d_out),
        "k_access": int(k_access),
        "real_access_ratio": float(real_access_ratio),
        "null_trials": int(N_NULL),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_95_range": [null_ci_lo, null_ci_hi],
        "p_null_le_real": p_le_real,
        "verdict": verdict,
    }
    with open("E15_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote E15_results.json")


if __name__ == "__main__":
    main()
