"""
E13: cross-validated re-estimate of swap_obj probe accuracy.

Motivation: E9's swap_obj number (50.0%) comes from a single seed-42
80/20 pair split -- with n=245 pairs, the held-out test fold is only 49
pairs (98 samples), so the standard error on that one point estimate is
roughly +-7pp. That's not a stable enough estimate to report as "the"
swap_obj number. This reruns the SAME feature extraction and MLP
architecture as E9 under 5-fold pair-level CV and reports mean +- 95% CI
across folds, instead of one split.

Also includes a lower-capacity backup: logistic regression and a linear
SVM, same CV folds, same features. This isn't method-shopping -- the
choice to drop capacity is justified independently of any result, by the
n=245 sample size alone (a 512-256 MLP has orders of magnitude more
parameters than training examples). Both variants are reported regardless
of which looks better.

Uses the SAME subset_of() filter as every other per-category script, so
n_pairs should match E9 (245) up to the swap_obj off-by-one noted when
cross-checking against HuggingFaceM4/SugarCrepe (246 there vs 245 here) --
worth a quick look if the counts don't match exactly.

Output: E13_results.json
"""

import json
import os
import sys
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(__file__))
from E2_direct_spectral import subset_of

SUBSET = "swap_obj"
N_FOLDS = 5
SEED = 42


def extract_features(dataset, subset_filter, model, processor, device):
    filtered_indices = []
    for i, ex in enumerate(tqdm(dataset, desc=f"filtering to {subset_filter}")):
        s = subset_of(ex)
        if s is not None and subset_filter in s:
            filtered_indices.append(i)
    scoped = dataset.select(filtered_indices)
    n = len(scoped)
    print(f"n_pairs = {n}")

    post_samples, labels = [], []
    with torch.no_grad():
        for ex in tqdm(scoped, desc="extracting features"):
            img = ex["0.webp"].convert("RGB")
            true_cap, false_cap = str(ex["npy"][0]), str(ex["npy"][1])

            text_inputs = processor(text=[true_cap, false_cap], return_tensors="pt",
                                     padding=True, truncation=True).to(device)
            text_outputs = model.text_model(**text_inputs)
            text_embeds = model.text_projection(text_outputs.pooler_output)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            t_true, t_false = text_embeds[0].cpu().numpy(), text_embeds[1].cpu().numpy()

            img_inputs = processor(images=[img], return_tensors="pt").to(device)
            vision_outputs = model.vision_model(**img_inputs)
            post_proj = model.visual_projection(vision_outputs.pooler_output)[0]
            post_proj = post_proj / post_proj.norm(dim=-1, keepdim=True)
            post_proj = post_proj.cpu().numpy()

            post_samples.append(np.concatenate([post_proj, t_true]))
            labels.append(1)
            post_samples.append(np.concatenate([post_proj, t_false]))
            labels.append(0)

    return np.array(post_samples), np.array(labels), n


def pair_kfold_eval(X, y, n_pairs, make_clf, seed=SEED, n_folds=N_FOLDS):
    """K-fold CV at the PAIR level (both samples of a pair always in the
    same fold), matching the pair-level train/test split convention used
    throughout this project (E1/E9/corrected_probe_eval.py)."""
    pair_idx = np.arange(n_pairs)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    for train_pairs, test_pairs in kf.split(pair_idx):
        train_idx = np.concatenate([train_pairs * 2, train_pairs * 2 + 1])
        test_idx = np.concatenate([test_pairs * 2, test_pairs * 2 + 1])
        clf = make_clf()
        clf.fit(X[train_idx], y[train_idx])
        fold_accs.append(clf.score(X[test_idx], y[test_idx]))
    fold_accs = np.array(fold_accs)
    mean = fold_accs.mean()
    se = fold_accs.std(ddof=1) / np.sqrt(n_folds)
    ci95 = 1.96 * se
    return {
        "fold_accuracies": fold_accs.tolist(),
        "mean_accuracy": float(mean),
        "std_accuracy": float(fold_accs.std(ddof=1)),
        "se_of_mean": float(se),
        "ci95_low": float(mean - ci95),
        "ci95_high": float(mean + ci95),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading haideraltahan/wds_sugarcrepe ...")
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    X, y, n_pairs = extract_features(dataset, SUBSET, model, processor, device)

    results = {"subset": SUBSET, "n_pairs": n_pairs, "n_folds": N_FOLDS, "seed": SEED}

    print("\n=== 5-fold CV: MLP (512,256), same architecture as E9 ===")
    results["mlp_512_256"] = pair_kfold_eval(
        X, y, n_pairs,
        lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000,
                               random_state=SEED, early_stopping=True))
    print(json.dumps(results["mlp_512_256"], indent=2))

    print("\n=== 5-fold CV: logistic regression (low-capacity backup) ===")
    results["logistic_regression"] = pair_kfold_eval(
        X, y, n_pairs,
        lambda: LogisticRegression(max_iter=2000, random_state=SEED))
    print(json.dumps(results["logistic_regression"], indent=2))

    print("\n=== 5-fold CV: linear SVM (low-capacity backup) ===")
    results["linear_svm"] = pair_kfold_eval(
        X, y, n_pairs,
        lambda: LinearSVC(max_iter=5000, random_state=SEED))
    print(json.dumps(results["linear_svm"], indent=2))

    with open("E13_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (swap_obj, n={}, {}-fold CV) ===".format(n_pairs, N_FOLDS))
    print(f"  single-split E9 reference: 50.0% (49-pair test fold, seed 42 only)")
    for name in ["mlp_512_256", "logistic_regression", "linear_svm"]:
        r = results[name]
        print(f"  {name:22s} mean={r['mean_accuracy']:.3f}  "
              f"95% CI=[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")

    print("\nwrote E13_results.json")


if __name__ == "__main__":
    main()
