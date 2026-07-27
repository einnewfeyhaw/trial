"""
E7b: Per-subset access ratio across all SugarCrepe categories.

Closes the scope gap flagged in review of the g6iB rebuttal draft: E7's
access_ratio=0.484 was verified only on swap_obj (n=245). The reviewer's
original Q2 explicitly referenced attribute/relation swaps, where the
paper's own probe drops to near chance (Swap Attribute 51.8%, Replace
Relation 50.0%). Reporting swap_obj alone risks an AC noticing the same
scope mismatch already flagged elsewhere for this paper (BTbD, G1/G3).

This script reuses E7_structural_variance.py's exact analysis (same
abs-value-corrected compute_access_ratio, same bootstrap procedure) and
E2_direct_spectral.py's subset_of() filter, looping over every subset found
in the dataset rather than hardcoding swap_obj. One single pass over the
full test split buckets examples by subset (avoids 7x redundant scans),
then each subset is analyzed independently up to --limit-per-subset.

Honest expectation, stated up front: if the probe itself is near-chance on
attribute/relation swaps, access_ratio there may be close to 1.0 (no
reliable masking OR alignment) rather than showing the same clean effect as
swap_obj. That is itself a valid, reportable finding -- do not tune anything
to force a particular direction. Report whatever comes out.

Output: E7b_per_subset_results.json
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 10_000
TOP_FRAC = 0.20


def normalize(x, eps=1e-12):
    n = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / n


def bucket_by_subset(dataset, limit_per_subset, scan_cap):
    """Single pass: bucket example indices by subset label, up to
    limit_per_subset each. Returns {subset_label: [indices]}."""
    buckets = defaultdict(list)
    n_scanned = 0
    for i, ex in enumerate(tqdm(dataset, desc="  bucketing by subset")):
        n_scanned = i + 1
        s = subset_of(ex)
        if s is not None and len(buckets[s]) < limit_per_subset:
            buckets[s].append(i)
        if scan_cap and n_scanned >= scan_cap:
            break
    print(f"  scanned {n_scanned} examples, found {len(buckets)} subsets:")
    for s, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"    {s!r}: {len(idxs)}")
    return buckets, n_scanned


def analyze_subset(model, processor, U_t, S, d_out, dataset, indices, rng):
    """Same computation as E7_structural_variance.py's main(), factored out
    to run per-subset. Returns the result dict for this subset, or None if
    too few examples to be meaningful."""
    if len(indices) < 20:
        return {"n_pairs": len(indices), "skipped": "fewer than 20 examples"}

    sub = dataset.select(indices)
    match_contrib, mismatch_contrib = [], []
    with torch.no_grad():
        for ex in sub:
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

    match_contrib = np.array(match_contrib)
    mismatch_contrib = np.array(mismatch_contrib)
    n_pairs = len(match_contrib)
    mean_cosine = float(match_contrib.sum(axis=1).mean())

    cosine_mass = match_contrib.mean(axis=0)

    labels = np.ones(2 * n_pairs)
    labels[n_pairs:] = 0
    scores_concat = np.concatenate([match_contrib, mismatch_contrib], axis=0)
    auc_per_dim = np.zeros(d_out)
    for dim in range(d_out):
        try:
            auc = roc_auc_score(labels, scores_concat[:, dim])
        except ValueError:
            auc = 0.5  # degenerate column (e.g. all-zero), no discriminability
        auc_per_dim[dim] = abs(auc - 0.5)

    k_top = int(np.round(TOP_FRAC * d_out))
    top_sv_idx = np.arange(k_top)
    frac_cosine_top = float(np.abs(cosine_mass[top_sv_idx]).sum() / np.abs(cosine_mass).sum())
    frac_auc_top = float(auc_per_dim[top_sv_idx].sum() / auc_per_dim.sum())
    amplification = frac_cosine_top / frac_auc_top if frac_auc_top > 0 else float("nan")

    k_access = max(1, int(np.round(0.05 * d_out)))
    disc_rank = np.argsort(auc_per_dim)[::-1]
    top_disc_idx = disc_rank[:k_access]

    def compute_access_ratio(mc):
        mc_abs = np.abs(mc)
        mc_total = mc_abs.sum()
        if mc_total <= 0:
            return np.nan
        return (mc_abs[top_disc_idx].sum() / mc_total) / (k_access / d_out)

    point = compute_access_ratio(cosine_mass)
    boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_pairs, n_pairs)
        boot[b] = compute_access_ratio(match_contrib[idx].mean(axis=0))
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p_ge_1 = float(np.mean(boot >= 1.0))
    direction = "masking" if point < 1.0 else "alignment"
    reliability = (1.0 - p_ge_1) if direction == "masking" else p_ge_1

    return {
        "n_pairs": int(n_pairs),
        "mean_cosine_similarity": mean_cosine,
        "top20pct_sv_dims": {
            "cosine_mass_fraction": frac_cosine_top,
            "auc_discriminability_fraction": frac_auc_top,
            "amplification_ratio_cosine_over_auc": float(amplification),
        },
        "access_ratio": {
            "point_estimate": float(point),
            "ci_95": [ci_lo, ci_hi],
            "p_ratio_ge_1": p_ge_1,
            "direction": direction,
            "reliability": reliability,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-subset", type=int, default=300)
    ap.add_argument("--scan-cap", type=int, default=0,
                    help="0 = scan the full test split once; set a cap for a quick check")
    ap.add_argument("--out", default="E7b_per_subset_results.json")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    W = model.visual_projection.weight.detach().cpu().numpy()
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    d_out = len(S)
    U_t = torch.tensor(U.T, dtype=torch.float32).to(DEVICE)

    print("Loading haideraltahan/wds_sugarcrepe ...")
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    buckets, n_scanned = bucket_by_subset(dataset, args.limit_per_subset, args.scan_cap)

    results = {
        "model": "openai/clip-vit-base-patch32",
        "n_scanned_total": n_scanned,
        "limit_per_subset": args.limit_per_subset,
        "subsets": {},
    }
    for subset_label, indices in buckets.items():
        print(f"\n=== subset: {subset_label!r} (n={len(indices)}) ===")
        r = analyze_subset(model, processor, U_t, S, d_out, dataset, indices, rng)
        results["subsets"][subset_label] = r
        print(json.dumps(r, indent=2))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")
    print("\nSummary (access_ratio point estimate, direction, n_pairs):")
    for s, r in results["subsets"].items():
        if "access_ratio" in r:
            ar = r["access_ratio"]
            print(f"  {s:25s} n={r['n_pairs']:4d}  ratio={ar['point_estimate']:.3f}  "
                  f"[{ar['ci_95'][0]:.3f},{ar['ci_95'][1]:.3f}]  {ar['direction']}")
        else:
            print(f"  {s:25s} {r.get('skipped', 'no result')}")


if __name__ == "__main__":
    main()
