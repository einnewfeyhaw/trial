"""
E9: Held-out probe accuracy across all 7 SugarCrepe categories.

Addresses Reviewer BTbD directly: "this 80.3% is only on 1 split out of 7
(swap object)... I would suggest that the authors check their claims on
more datasets and adjust the text accordingly."

Reuses corrected_probe_eval.py's exact protocol unmodified (concatenated
[embedding, caption] features, MLPClassifier(512,256), pair-wise 80/20
held-out split, seed 42) -- only the dataset loading changes, adding the
subset filter already verified in E2/E7/E7b so each split is genuinely
what it claims to be.

Output: E9_results.json with post-projection accuracy for all 7 categories.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(__file__))
from E2_direct_spectral import subset_of

SUBSETS = ["swap_obj", "swap_att", "replace_obj", "replace_att",
           "replace_rel", "add_obj", "add_att"]


def extract_for_subset(model, processor, device, dataset, subset_filter, limit):
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
        return None, None, 0
    print(f"  matched {len(filtered_indices)} examples out of {n_scanned} scanned")
    scoped = dataset.select(filtered_indices)

    post_samples, labels = [], []
    with torch.no_grad():
        for example in tqdm(scoped, desc="  encoding"):
            img = example["0.webp"].convert("RGB")
            true_cap, false_cap = example["npy"][0], example["npy"][1]

            text_inputs = processor(text=[true_cap, false_cap], return_tensors="pt",
                                     padding=True, truncation=True).to(device)
            text_outputs = model.text_model(**text_inputs)
            text_embeds = model.text_projection(text_outputs.pooler_output)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            t_true = text_embeds[0].cpu().numpy()
            t_false = text_embeds[1].cpu().numpy()

            img_inputs = processor(images=[img], return_tensors="pt").to(device)
            vision_outputs = model.vision_model(**img_inputs)
            post_proj = model.visual_projection(vision_outputs.pooler_output)[0]
            post_proj = post_proj / post_proj.norm(dim=-1, keepdim=True)
            post_proj = post_proj.cpu().numpy()

            post_samples.append(np.concatenate([post_proj, t_true]))
            labels.append(1)
            post_samples.append(np.concatenate([post_proj, t_false]))
            labels.append(0)

    return np.array(post_samples), np.array(labels), len(filtered_indices)


def analyze_subset(model, processor, device, dataset, subset_filter, limit):
    print(f"\n=== {subset_filter} ===")
    X, y, n_pairs = extract_for_subset(model, processor, device, dataset, subset_filter, limit)
    if X is None or n_pairs < 20:
        return {"n_pairs": n_pairs, "skipped": "too few examples"}

    train_idx_pairs, test_idx_pairs = train_test_split(
        np.arange(n_pairs), test_size=0.2, random_state=42)
    train_idx = np.concatenate([train_idx_pairs * 2, train_idx_pairs * 2 + 1])
    test_idx = np.concatenate([test_idx_pairs * 2, test_idx_pairs * 2 + 1])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000,
                        random_state=42, early_stopping=True)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return {
        "n_pairs": int(n_pairs),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "post_projection_accuracy": float(acc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-subset", type=int, default=1000)
    ap.add_argument("--out", default="E9_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    results = {}
    for subset in SUBSETS:
        r = analyze_subset(model, processor, device, dataset, subset, args.limit_per_subset)
        results[subset] = r
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")
    print("\n=== SUMMARY ===")
    for s, r in results.items():
        if "post_projection_accuracy" in r:
            print(f"  {s:15s} n={r['n_pairs']:4d}  acc={r['post_projection_accuracy']:.1%}")
        else:
            print(f"  {s:15s} {r.get('skipped', 'no result')}")


if __name__ == "__main__":
    main()
