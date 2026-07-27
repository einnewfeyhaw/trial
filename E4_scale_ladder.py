"""
E4: Scale Ladder — SVD Spearman rho across a model size ladder.

Addresses Reviewer iv2d Q1:
  "The magnitude–compositionality reversal is shown only for SigLIP-SO400M.
   Can the authors provide evidence across a model series to disentangle which
   factor drives the reversal?"

Design: reuse E1's exact extract_features() pipeline (same SVD basis, same
pairwise split, same probes, same seeds) across a ladder of models that vary
scale while holding objective/data approximately constant:

  LAION ladder (same CLIP objective + LAION-2B data, varying scale):
    CLIP-B/32-openai     ~63M  vision params
    CLIP-ViT-H-14-laion  ~302M
    CLIP-ViT-g-14-laion  ~354M (bigG uses different data; skip for now)

  SigLIP ladder (same sigmoid objective, varying scale):
    SigLIP-base-patch16  ~86M
    SigLIP-SO400M        ~400M

This lets us ask: within the LAION CLIP family, does rho become more positive
as scale increases? And within SigLIP, same question?

If rho trends from negative → zero → positive with scale (within either
family), scale is the prime suspect over objective or architecture.

Outputs: E4_results.json  (mirrors E1 output format per model)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Import E1's helpers (they live in the same directory)
sys.path.insert(0, os.path.dirname(__file__))
# CRITICAL FIX: this script originally loaded haideraltahan/wds_sugarcrepe with
# no subset filter, the same bug found in the ORIGINAL submitted-paper scripts
# and in E1_robust_probing.py -- we have direct proof (g6iB per-subset
# investigation) that the first 500 raw examples of this dataset are 100%
# "add_obj", not the "Swap Object" the paper's Table 1 claims. This explains
# why raw_rho here didn't match Table 1: different (mislabeled) data, not a
# different protocol. Reuse the already-fixed filter from E2/E7/E7b.
from E2_direct_spectral import subset_of

SCALE_LADDER = [
    # (model_id, family, approx_vision_params_M, note)
    ("openai/clip-vit-base-patch32",        "CLIP-OpenAI",  63,   "Base – OpenAI CLIP"),
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "CLIP-LAION", 302,  "Huge – LAION CLIP"),
    ("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "CLIP-LAION", 354,  "Giant – LAION CLIP"),
    ("google/siglip-base-patch16-224",      "SigLIP",       86,   "SigLIP Base"),
    ("google/siglip-so400m-patch14-384",    "SigLIP",       400,  "SigLIP SO400M"),
]

SEEDS  = [42, 43, 44]   # fewer seeds than E1 for speed; still 3-seed average
CACHE  = "embedding_cache"


def get_projection_matrix(model, model_name):
    if getattr(model, "visual_projection", None) is not None:
        return model.visual_projection.weight.detach().cpu().numpy()
    if "siglip" in model_name.lower():
        # Same approach as E1 — use attention out_proj weight as proxy basis
        return model.vision_model.head.attention.out_proj.weight.detach().cpu().numpy()
    return None


def extract_features(model_name, dataset, device, limit, subset_filter="swap_obj"):
    os.makedirs(CACHE, exist_ok=True)
    tag   = model_name.replace("/", "_")
    stag  = subset_filter or "UNFILTERED"
    cache = os.path.join(CACHE, f"{tag}_sugarcrepe_{stag}_{limit}.npz")
    if os.path.exists(cache):
        print(f"  cached: {cache}")
        d = np.load(cache)
        return d["X"], d["y"], d["S"]

    proc  = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    W = get_projection_matrix(model, model_name)
    if W is None:
        print(f"  no visual projection for {model_name} — skipping")
        return None, None, None

    U, S, _ = np.linalg.svd(W, full_matrices=False)
    U_t     = torch.tensor(U.T).float().to(device)
    is_sig  = "siglip" in model_name.lower()

    if subset_filter:
        filtered_indices = []
        n_scanned = 0
        for i, ex in enumerate(tqdm(dataset, desc=f"  filtering to {subset_filter}")):
            n_scanned = i + 1
            s = subset_of(ex)
            if s is not None and subset_filter in s:
                filtered_indices.append(i)
                if len(filtered_indices) >= limit:
                    break
        if not filtered_indices:
            print(f"  no examples matched '{subset_filter}'; skipping")
            return None, None, None
        print(f"  matched {len(filtered_indices)} '{subset_filter}' examples "
              f"out of {n_scanned} scanned")
        scoped_dataset = dataset.select(filtered_indices)
    else:
        print("  WARNING: no subset filter -- reproduces the original bug")
        scoped_dataset = dataset.select(range(min(limit, len(dataset))))

    samples, labels = [], []
    with torch.no_grad():
        for ex in tqdm(scoped_dataset, desc=f"  {model_name.split('/')[-1]}"):
            img     = ex["0.webp"].convert("RGB")
            tc, fc  = ex["npy"][0], ex["npy"][1]
            kw      = {"padding": "max_length"} if is_sig else {"padding": True}
            inp     = proc(text=[tc, fc], images=img, return_tensors="pt",
                           truncation=True, **kw).to(device)
            out     = model(**inp)
            te      = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            ve      = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            if ve.shape[-1] != U_t.shape[0]:
                continue
            z       = ve[0] @ U_t.T
            samples.append((z * (te[0] @ U_t.T)).cpu().numpy())
            labels.append(1)
            samples.append((z * (te[1] @ U_t.T)).cpu().numpy())
            labels.append(0)

    del model
    if device == "cuda": torch.cuda.empty_cache()
    X = np.asarray(samples); y = np.asarray(labels)
    np.savez_compressed(cache, X=X, y=y, S=S)
    return X, y, S


def analyze(model_name, family, params_M, note, dataset, device, limit):
    print(f"\n{'='*60}")
    print(f"  {model_name}  ({family}, ~{params_M}M params)")
    X, y, S = extract_features(model_name, dataset, device, limit)
    if X is None:
        return None

    n_pairs = len(X) // 2
    # pairwise split (same as E1)
    n_p  = n_pairs
    tr_p, te_p = train_test_split(np.arange(n_p), test_size=0.2, random_state=42)
    tr_i = np.concatenate([tr_p*2, tr_p*2+1])
    te_i = np.concatenate([te_p*2, te_p*2+1])
    X_tr, X_te = X[tr_i], X[te_i]
    y_tr, y_te = y[tr_i], y[te_i]

    # scale confound
    feat_std = X_tr.std(axis=0)
    rho_sc, p_sc = spearmanr(S, feat_std)

    scaler        = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    raw_rhos, raw_accs = [], []
    std_rhos, null_rhos = [], []
    rng = np.random.RandomState(0)

    for seed in SEEDS:
        clf = MLPClassifier((256,), max_iter=500, random_state=seed,
                            early_stopping=True).fit(X_tr, y_tr)
        raw_accs.append(clf.score(X_te, y_te))
        imp = np.linalg.norm(clf.coefs_[0], axis=1)
        raw_rhos.append(spearmanr(S, imp)[0])

        clfs = MLPClassifier((256,), max_iter=500, random_state=seed,
                             early_stopping=True).fit(X_tr_s, y_tr)
        imp_s = np.linalg.norm(clfs.coefs_[0], axis=1)
        std_rhos.append(spearmanr(S, imp_s)[0])

        clfn = MLPClassifier((256,), max_iter=500, random_state=seed,
                             early_stopping=True).fit(X_tr, rng.permutation(y_tr))
        imp_n = np.linalg.norm(clfn.coefs_[0], axis=1)
        null_rhos.append(spearmanr(S, imp_n)[0])

    # permutation importance (scale-free gold standard)
    clf_p = MLPClassifier((256,), max_iter=500, random_state=SEEDS[0],
                          early_stopping=True).fit(X_tr, y_tr)
    perm  = permutation_importance(clf_p, X_te, y_te, n_repeats=5,
                                   random_state=42, n_jobs=-1)
    rho_perm, p_perm = spearmanr(S, perm.importances_mean)

    # cosine contribution per dim
    cos_contrib = np.abs(X[y==1]).mean(axis=0)
    rho_cos, p_cos = spearmanr(S, cos_contrib)

    res = {
        "model": model_name, "family": family,
        "approx_params_M": params_M, "note": note,
        "n_pairs": n_pairs,
        "scale_confound": {
            "rho_sigma_vs_feature_std": float(rho_sc), "p": float(p_sc)},
        "raw_rho_mean": float(np.mean(raw_rhos)),
        "raw_rho_std":  float(np.std(raw_rhos)),
        "raw_test_acc": float(np.mean(raw_accs)),
        "std_rho_mean": float(np.mean(std_rhos)),
        "null_rho_mean": float(np.mean(null_rhos)),
        "permutation_rho": float(rho_perm),
        "permutation_p":   float(p_perm),
        "cosine_rho":  float(rho_cos),
        "cosine_p":    float(p_cos),
    }

    print(f"  raw_rho={res['raw_rho_mean']:+.3f}  "
          f"std_rho={res['std_rho_mean']:+.3f}  "
          f"perm_rho={res['permutation_rho']:+.3f} (p={res['permutation_p']:.2e})  "
          f"null_rho={res['null_rho_mean']:+.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out",   default="E4_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"limit: {args.limit} examples from haideraltahan/wds_sugarcrepe")

    ds = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    results = {}
    for model_id, family, params, note in SCALE_LADDER:
        try:
            r = analyze(model_id, family, params, note, ds, device, args.limit)
            if r:
                results[model_id] = r
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → checkpoint saved to {args.out}")
        except Exception as e:
            import traceback
            print(f"FAILED {model_id}: {e}")
            print(traceback.format_exc())

    print(f"\nwrote {args.out}")
    print("\n=== SCALE LADDER SUMMARY ===")
    print(f"  {'Model':<45} {'params':>8} {'raw_rho':>8} {'perm_rho':>9} {'null_rho':>9}")
    print(f"  {'-'*82}")
    for mid, r in results.items():
        print(f"  {mid:<45} {r['approx_params_M']:>6}M "
              f"{r['raw_rho_mean']:>+8.3f} {r['permutation_rho']:>+9.3f} "
              f"{r['null_rho_mean']:>+9.3f}")


if __name__ == "__main__":
    main()
