"""
E1: Robust re-run of the SVD feature-importance correlation.

Fixes three problems in the original Table 1 pipeline
(feature_importance.py, multi_model_correlation.py, multi_model_aro_correlation.py,
strong_model_correlation.py, aro_svd_correlation.py):

  1. No train/test split. Those scripts call clf.score(X, y) on training data,
     so the reported "Probe Acc." is train accuracy and the importance weights
     come from an unregularized fit with no held-out check.

  2. Pair leakage. Each image contributes two rows (true caption, false caption)
     that share the SAME image embedding. A flat random split puts siblings on
     opposite sides, leaking every test image into training. We split on PAIR
     indices, matching 01_probing_tests/corrected_probe_eval.py.

  3. Scale confound (the important one). Features are (v.U)_i * (t.U)_i. Because
     v' = vU = xV(Sigma), coordinate i is scaled by sigma_i, so low-sigma dims
     have intrinsically small variance. A probe needs LARGER weights on
     small-variance inputs, so an l1-weight-norm importance is mechanically
     anti-correlated with sigma even under zero compositional signal.

     We therefore report rho under four conditions:
       raw            - the paper's original measure (held-out, multi-seed)
       standardized   - inputs z-scored on train stats; kills the scale artifact
       permutation    - held-out permutation importance; scale-free, gold standard
       shuffled-label - null control; rho here should be ~0

     If rho survives standardization AND permutation importance AND the null is
     flat, magnitude masking is a real effect. If rho collapses to ~0 once
     standardized while the shuffled-label null reproduces the original negative
     value, the original finding was a scaling artifact.

Outputs E1_robust_results.json. Embeddings are cached to .npz for fast reruns.
"""

import argparse
import json
import os
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

MODELS_TO_TEST = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
]

SEEDS = [42, 43, 44, 45, 46]
CACHE_DIR = "embedding_cache"


def get_projection_matrix(model, model_name):
    if getattr(model, "visual_projection", None) is not None:
        return model.visual_projection.weight.detach().cpu().numpy()
    if "siglip" in model_name.lower():
        return model.vision_model.head.attention.out_proj.weight.detach().cpu().numpy()
    return None


def extract_features(model_name, dataset, device, limit):
    """Return X (n_samples, d) in the SVD basis, y, and the singular values S."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(
        CACHE_DIR, f"{model_name.replace('/', '_')}_sugarcrepe_{limit}.npz"
    )
    if os.path.exists(cache):
        print(f"  loading cached embeddings: {cache}")
        d = np.load(cache)
        return d["X"], d["y"], d["S"]

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    W = get_projection_matrix(model, model_name)
    if W is None:
        print(f"  no visual projection matrix for {model_name}; skipping")
        return None, None, None

    U, S, _ = np.linalg.svd(W, full_matrices=False)
    U_t = torch.tensor(U.T).float().to(device)

    samples, labels = [], []
    is_siglip = "siglip" in model_name.lower()

    with torch.no_grad():
        for example in tqdm(
            dataset.select(range(min(limit, len(dataset)))), desc="  encoding"
        ):
            img = example["0.webp"].convert("RGB")
            true_cap, false_cap = example["npy"][0], example["npy"][1]

            kwargs = (
                {"padding": "max_length"} if is_siglip else {"padding": True}
            )
            inputs = processor(
                text=[true_cap, false_cap],
                images=img,
                return_tensors="pt",
                truncation=True,
                **kwargs,
            ).to(device)
            out = model(**inputs)
            t_embeds = out.text_embeds
            v_embeds = out.image_embeds

            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)

            if v_embeds.shape[-1] != U_t.shape[0]:
                print(f"  dim mismatch for {model_name}; skipping")
                return None, None, None

            z = v_embeds[0] @ U_t.T
            t_true = t_embeds[0] @ U_t.T
            t_false = t_embeds[1] @ U_t.T

            # Order matters: row 2i and row 2i+1 belong to the same pair.
            samples.append((z * t_true).cpu().numpy())
            labels.append(1)
            samples.append((z * t_false).cpu().numpy())
            labels.append(0)

    X = np.asarray(samples)
    y = np.asarray(labels)
    np.savez_compressed(cache, X=X, y=y, S=S)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return X, y, S


def pairwise_split(n_samples, test_size=0.2, seed=42):
    """Split on pair indices so an image never spans train and test."""
    n_pairs = n_samples // 2
    train_p, test_p = train_test_split(
        np.arange(n_pairs), test_size=test_size, random_state=seed
    )
    train_idx = np.concatenate([train_p * 2, train_p * 2 + 1])
    test_idx = np.concatenate([test_p * 2, test_p * 2 + 1])
    return np.sort(train_idx), np.sort(test_idx)


def fit_mlp(X_tr, y_tr, seed):
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
    )
    clf.fit(X_tr, y_tr)
    return clf


def analyze(model_name, dataset, device, limit):
    print(f"\n=== {model_name} ===")
    X, y, S = extract_features(model_name, dataset, device, limit)
    if X is None:
        return None

    train_idx, test_idx = pairwise_split(len(X))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # --- diagnostic: does feature scale itself track sigma? ---
    feat_std = X_tr.std(axis=0)
    rho_scale, p_scale = spearmanr(S, feat_std)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    raw_rhos, raw_accs = [], []
    std_rhos, std_accs = [], []
    null_rhos = []

    rng = np.random.RandomState(0)
    for seed in SEEDS:
        # raw features (original paper measure, now held-out)
        clf = fit_mlp(X_tr, y_tr, seed)
        raw_accs.append(clf.score(X_te, y_te))
        imp = np.linalg.norm(clf.coefs_[0], axis=1)
        raw_rhos.append(spearmanr(S, imp)[0])

        # standardized features (scale artifact removed)
        clf_s = fit_mlp(X_tr_s, y_tr, seed)
        std_accs.append(clf_s.score(X_te_s, y_te))
        imp_s = np.linalg.norm(clf_s.coefs_[0], axis=1)
        std_rhos.append(spearmanr(S, imp_s)[0])

        # null control: shuffled labels on raw features
        clf_n = fit_mlp(X_tr, rng.permutation(y_tr), seed)
        imp_n = np.linalg.norm(clf_n.coefs_[0], axis=1)
        null_rhos.append(spearmanr(S, imp_n)[0])

    # --- permutation importance: scale-free, computed on held-out data ---
    clf_perm = fit_mlp(X_tr, y_tr, SEEDS[0])
    perm = permutation_importance(
        clf_perm, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1
    )
    rho_perm, p_perm = spearmanr(S, perm.importances_mean)

    # --- linear probe on standardized features ---
    lr = LogisticRegression(max_iter=2000, random_state=42).fit(X_tr_s, y_tr)
    rho_lr, p_lr = spearmanr(S, np.abs(lr.coef_[0]))

    # --- actual contribution of each dim to the cosine similarity ---
    # mean |v_i * t_i| over matched pairs = per-dim share of cos(v,t)
    cos_contrib = np.abs(X[y == 1]).mean(axis=0)
    rho_cos, p_cos = spearmanr(S, cos_contrib)

    res = {
        "model": model_name,
        "n_pairs": len(X) // 2,
        "scale_confound": {
            "rho_sigma_vs_feature_std": float(rho_scale),
            "p_value": float(p_scale),
        },
        "raw_features": {
            "test_acc_mean": float(np.mean(raw_accs)),
            "test_acc_std": float(np.std(raw_accs)),
            "rho_mean": float(np.mean(raw_rhos)),
            "rho_std": float(np.std(raw_rhos)),
        },
        "standardized_features": {
            "test_acc_mean": float(np.mean(std_accs)),
            "test_acc_std": float(np.std(std_accs)),
            "rho_mean": float(np.mean(std_rhos)),
            "rho_std": float(np.std(std_rhos)),
        },
        "shuffled_label_null": {
            "rho_mean": float(np.mean(null_rhos)),
            "rho_std": float(np.std(null_rhos)),
        },
        "permutation_importance": {
            "rho": float(rho_perm),
            "p_value": float(p_perm),
        },
        "linear_probe_standardized": {
            "test_acc": float(lr.score(X_te_s, y_te)),
            "rho": float(rho_lr),
            "p_value": float(p_lr),
        },
        "cosine_contribution": {
            "rho_sigma_vs_contribution": float(rho_cos),
            "p_value": float(p_cos),
        },
    }
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--models", nargs="*", default=MODELS_TO_TEST)
    ap.add_argument("--out", default="E1_robust_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    results = {}
    for name in args.models:
        try:
            r = analyze(name, dataset, device, args.limit)
            if r:
                results[name] = r
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            print(f"FAILED {name}: {e}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
