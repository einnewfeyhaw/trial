"""
E2: Direct per-dimension spectral analysis. No probe.

Replaces the l1-weight-norm-vs-sigma correlation, which is probe-dependent,
seed-dependent, scale-confounded, and sensitive to early stopping. Every
criticism raised against Table 1 attaches to that proxy rather than to the
underlying hypothesis.

This script measures the hypothesis directly. For each basis dimension i:

  cosine contribution  mean over matched pairs of (v'_i * t'_i). Because the
                       basis is orthonormal, these sum EXACTLY to cos(v, t),
                       so each value is that dimension's literal share of the
                       similarity score.

  discriminability     AUC of (v'_i * t'_i) alone as a univariate match/
                       mismatch classifier, reported as |AUC - 0.5|.
                       Rank-based, hence scale-invariant by construction and
                       immune to the magnitude confound.

Headline statistic:

  What fraction of total cosine mass is carried by the most DISCRIMINATIVE
  dimensions, versus by the highest-MAGNITUDE dimensions?

If compositionally informative dimensions carry a small share of cosine mass
while high-sigma dimensions carry the bulk, that is magnitude masking stated
quantitatively -- with no MLP, no seeds, no standardization debate. If the two
sets coincide, the hypothesis is not supported. Either outcome is defensible.

Also fixes two bugs present in both the original and the first correction:

  --subset      SugarCrepe subset filter. The wds mirror may mix all seven
                subsets; the paper reports "Swap Object". Training on a blend
                that includes near-chance subsets dilutes any effect.
                Use --inspect first to see what the mirror actually contains.

  --basis pca   The SigLIP "projection matrix" used previously was an attention
                POOLING head's out_proj, not an analogue of CLIP's
                visual_projection, so its SVD basis is likely meaningless.
                The pca basis is the uncentered SVD of the empirical image
                embedding matrix: uniform across architectures, and arguably a
                more faithful notion of "magnitude" for cosine similarity.

Pre-register your confirmation criterion BEFORE running.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

MODELS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
]


def inspect_dataset(dataset):
    """Print structure so we can tell whether subsets are mixed."""
    print("\n--- dataset inspection ---")
    print("columns:", dataset.column_names)
    ex = dataset[0]
    for k, v in ex.items():
        if k.endswith(".webp"):
            print(f"  {k}: <image>")
        else:
            s = str(v)
            print(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

    # Look for a field that encodes the subset.
    for key in ("__key__", "json", "cls", "txt", "subset", "type"):
        if key in dataset.column_names:
            vals = [str(dataset[i][key])[:80] for i in range(min(20, len(dataset)))]
            print(f"\n  sample '{key}' values:")
            for v in vals[:20]:
                print("   ", v)
    print(f"\n  total examples: {len(dataset)}")
    print("--- end inspection ---\n")


def subset_of(example):
    """Best-effort extraction of the SugarCrepe subset label."""
    for key in ("subset", "type", "__key__", "cls"):
        if key in example and example[key] is not None:
            return str(example[key])
    return None


def get_projection_matrix(model, model_name):
    if getattr(model, "visual_projection", None) is not None:
        return model.visual_projection.weight.detach().cpu().numpy()
    if "siglip" in model_name.lower():
        head = getattr(model.vision_model, "head", None)
        if head is not None and hasattr(head, "attention"):
            print("  WARNING: using SigLIP attention-pooling out_proj as the "
                  "projection matrix. This is probably NOT an analogue of "
                  "CLIP's visual_projection -- prefer --basis pca for SigLIP.")
            return head.attention.out_proj.weight.detach().cpu().numpy()
    return None


def encode(model_name, dataset, device, limit, subset_filter):
    """Return V (n,d) image embeds, T_true, T_false, and W (or None)."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    is_siglip = "siglip" in model_name.lower()

    W = get_projection_matrix(model, model_name)

    V, T_true, T_false = [], [], []
    kept = 0
    with torch.no_grad():
        for example in tqdm(dataset, desc="  encoding"):
            if kept >= limit:
                break
            if subset_filter:
                s = subset_of(example)
                if s is None or subset_filter not in s:
                    continue

            img = example["0.webp"].convert("RGB")
            true_cap, false_cap = example["npy"][0], example["npy"][1]

            kwargs = {"padding": "max_length"} if is_siglip else {"padding": True}
            inputs = processor(
                text=[true_cap, false_cap],
                images=img,
                return_tensors="pt",
                truncation=True,
                **kwargs,
            ).to(device)
            out = model(**inputs)

            t = out.text_embeds
            v = out.image_embeds
            t = t / t.norm(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True)

            V.append(v[0].cpu().numpy())
            T_true.append(t[0].cpu().numpy())
            T_false.append(t[1].cpu().numpy())
            kept += 1

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if kept == 0:
        return None, None, None, None
    return np.array(V), np.array(T_true), np.array(T_false), W


def build_basis(basis, V, W):
    """Return (B, sigma) where rows of B are orthonormal basis vectors."""
    if basis == "proj":
        if W is None:
            return None, None
        U, S, _ = np.linalg.svd(W, full_matrices=False)
        return U.T, S  # v' = v @ U  ->  rows of U.T
    # uncentered SVD of the image embedding matrix; preserves dot products
    _, S, Vh = np.linalg.svd(V, full_matrices=False)
    return Vh, S


def analyze(model_name, dataset, device, args):
    print(f"\n=== {model_name}  (basis={args.basis}, subset={args.subset}) ===")
    V, T_true, T_false, W = encode(
        model_name, dataset, device, args.limit, args.subset
    )
    if V is None:
        print("  no examples after filtering; skipping")
        return None

    B, sigma = build_basis(args.basis, V, W)
    if B is None:
        print("  no basis available; skipping")
        return None

    # rotate into the basis (orthonormal => cosine is preserved)
    Vb = V @ B.T
    Tt = T_true @ B.T
    Tf = T_false @ B.T

    P_match = Vb * Tt      # (n, d) sums along axis 1 to cos(v, t_true)
    P_mismatch = Vb * Tf

    d = P_match.shape[1]
    n = P_match.shape[0]

    # --- per-dimension discriminability: scale-invariant ---
    y = np.concatenate([np.ones(n), np.zeros(n)])
    stacked = np.vstack([P_match, P_mismatch])
    auc = np.array([roc_auc_score(y, stacked[:, i]) for i in range(d)])
    discrim = np.abs(auc - 0.5)

    # --- per-dimension cosine contribution ---
    contrib = P_match.mean(axis=0)          # signed; sums to mean cos
    mass = np.abs(contrib)
    mass_share = mass / mass.sum()

    # --- correlations with magnitude ---
    rho_disc, p_disc = spearmanr(sigma, discrim)
    rho_mass, p_mass = spearmanr(sigma, mass)

    # --- headline: cosine mass held by the most discriminative dims ---
    def top_share(rank_by, frac):
        k = max(1, int(frac * d))
        idx = np.argsort(rank_by)[::-1][:k]
        return float(mass_share[idx].sum()), idx

    out = {
        "model": model_name,
        "basis": args.basis,
        "subset": args.subset,
        "n_examples": int(n),
        "n_dims": int(d),
        "mean_cosine_match": float(P_match.sum(axis=1).mean()),
        "mean_cosine_mismatch": float(P_mismatch.sum(axis=1).mean()),
        "sigma_vs_discriminability": {"rho": float(rho_disc), "p": float(p_disc)},
        "sigma_vs_cosine_mass": {"rho": float(rho_mass), "p": float(p_mass)},
        "cosine_mass_share": {},
        "discriminability_of_top_sigma": {},
    }

    for frac in (0.05, 0.10, 0.20):
        share_disc, idx_disc = top_share(discrim, frac)
        share_sigma, idx_sigma = top_share(sigma, frac)
        pct = f"top_{int(frac*100)}pct"
        out["cosine_mass_share"][pct] = {
            "by_discriminability": share_disc,
            "by_sigma": share_sigma,
            "uniform_baseline": frac,
            # >1 means informative dims get MORE cosine mass than chance;
            # <1 is the magnitude-masking prediction.
            "access_ratio": share_disc / frac,
            "overlap_with_top_sigma": float(
                len(np.intersect1d(idx_disc, idx_sigma)) / len(idx_disc)
            ),
        }
        out["discriminability_of_top_sigma"][pct] = {
            "mean_discrim_top_sigma": float(discrim[idx_sigma].mean()),
            "mean_discrim_all": float(discrim.mean()),
        }

    print(json.dumps(out, indent=2))

    if args.dump_dims:
        np.savez_compressed(
            f"dims_{model_name.replace('/', '_')}_{args.basis}.npz",
            sigma=sigma, discrim=discrim, contrib=contrib, auc=auc,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--basis", choices=["proj", "pca"], default="proj")
    ap.add_argument("--subset", default=None,
                    help="substring filter for the SugarCrepe subset, e.g. swap_obj")
    ap.add_argument("--inspect", action="store_true",
                    help="print dataset structure and exit")
    ap.add_argument("--dump-dims", action="store_true")
    ap.add_argument("--out", default="E2_direct_spectral.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    if args.inspect:
        inspect_dataset(dataset)
        return

    results = {}
    for name in args.models:
        try:
            r = analyze(name, dataset, device, args)
            if r:
                results[name] = r
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            print(f"FAILED {name}: {e}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
