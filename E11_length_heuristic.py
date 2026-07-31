"""
E11: Length-heuristic control across all 7 SugarCrepe categories.

Addresses Reviewer BTbD's specific hypothesis directly: "Can't it be that
... your probes detect the correct caption in these cases, and not because
there is a survival of compositional information?" -- referring to
add_obj/add_att, where negative captions are constructed by ADDING extra
content, making them longer than the true caption (consistent with
Udandarao et al. 2025's finding that blind length/likelihood heuristics
match CLIP's performance across several compositional benchmarks).

This tests the exact mechanism directly: for each pair, predict the TRUE
caption is whichever one has FEWER words -- no image, no model, no
embeddings, nothing but a word count. If this trivial heuristic matches or
nearly matches the MLP probe's accuracy on a given category (E9_results.json),
that's direct evidence the probe may be exploiting caption length rather
than genuine compositional understanding there. If the heuristic sits at
chance while the probe is meaningfully above chance, that category is not
explained by this bias.

Uses the SAME subset filter (subset_of, from E2_direct_spectral.py) as
every other per-category script in this repo, so n-per-category is directly
comparable to what's already reported.

Output: E11_results.json
"""

import argparse
import json
import os
import sys
import warnings

from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(__file__))
from E2_direct_spectral import subset_of

SUBSETS = ["swap_obj", "swap_att", "replace_obj", "replace_att",
           "replace_rel", "add_obj", "add_att"]

# Already-reported held-out MLP probe accuracy per category (E9_results.json),
# included here only for direct side-by-side reporting -- not used in any
# computation in this script.
PROBE_ACCURACY = {
    "add_obj": 0.805, "add_att": 0.705, "replace_rel": 0.610,
    "replace_obj": 0.582, "swap_att": 0.515, "replace_att": 0.500,
    "swap_obj": 0.500,
}


def word_count(text):
    return len(text.strip().split())


def evaluate_subset(dataset, subset_filter, limit):
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
        return {"n_pairs": 0, "skipped": "no examples matched"}

    n = len(filtered_indices)
    scoped = dataset.select(filtered_indices)

    correct_shorter = 0.0  # heuristic: TRUE caption is the SHORTER one
    correct_longer = 0.0   # heuristic: TRUE caption is the LONGER one
    ties = 0
    len_true_total = 0
    len_false_total = 0

    for ex in scoped:
        true_cap, false_cap = str(ex["npy"][0]), str(ex["npy"][1])
        lt, lf = word_count(true_cap), word_count(false_cap)
        len_true_total += lt
        len_false_total += lf

        if lt == lf:
            # A tie can't be resolved by either direction of the heuristic.
            # Score as a coin flip (0.5) for both rather than silently
            # dropping it, so ties don't inflate either heuristic's accuracy.
            ties += 1
            correct_shorter += 0.5
            correct_longer += 0.5
        elif lt < lf:
            correct_shorter += 1.0
        else:
            correct_longer += 1.0

    shorter_acc = correct_shorter / n
    longer_acc = correct_longer / n
    best_acc = max(shorter_acc, longer_acc)
    best_direction = "shorter_is_true" if shorter_acc >= longer_acc else "longer_is_true"

    return {
        "n_pairs": n,
        "n_scanned": n_scanned,
        "mean_len_true_caption_words": len_true_total / n,
        "mean_len_false_caption_words": len_false_total / n,
        "n_ties_same_length": ties,
        "heuristic_shorter_is_true_accuracy": shorter_acc,
        "heuristic_longer_is_true_accuracy": longer_acc,
        "best_length_heuristic_accuracy": best_acc,
        "best_heuristic_direction": best_direction,
        "mlp_probe_accuracy_for_reference": PROBE_ACCURACY.get(subset_filter),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-subset", type=int, default=1000)
    ap.add_argument("--out", default="E11_results.json")
    args = ap.parse_args()

    print("Loading haideraltahan/wds_sugarcrepe ...")
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")

    results = {}
    for subset in SUBSETS:
        print(f"\n=== {subset} ===")
        r = evaluate_subset(dataset, subset, args.limit_per_subset)
        results[subset] = r
        print(json.dumps(r, indent=2))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")
    print("\n=== SUMMARY: length heuristic vs. MLP probe accuracy ===")
    print(f"  {'Category':13s} {'n':>5s} {'LenHeur':>8s} {'MLPProbe':>9s} {'Gap':>7s}  Direction")
    for s, r in results.items():
        if "best_length_heuristic_accuracy" not in r:
            print(f"  {s:13s} skipped: {r.get('skipped')}")
            continue
        h = r["best_length_heuristic_accuracy"]
        p = r["mlp_probe_accuracy_for_reference"]
        gap = (p - h) if p is not None else None
        gap_s = f"{gap:+.3f}" if gap is not None else "n/a"
        print(f"  {s:13s} {r['n_pairs']:5d} {h:8.3f} "
              f"{(p if p is not None else float('nan')):9.3f} {gap_s:>7s}  {r['best_heuristic_direction']}")


if __name__ == "__main__":
    main()
