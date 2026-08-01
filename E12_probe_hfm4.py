"""
E12 step 2: independent re-run of the held-out MLP probe AND the length
heuristic (E9 / E11) against HuggingFaceM4/SugarCrepe instead of
haideraltahan/wds_sugarcrepe, as a cross-dataset check.

Schema (confirmed via E12_inspect_hfm4.py output, one repo per category):
    image         : PIL image, decode=True
    tested_labels : [true_caption, false_caption]   # index 0 = true, confirmed
                                                      # 100% of rows in all 7 configs
    true_label    : ClassLabel index into an 11,672-string global vocabulary
                    (NOT used here -- we rely on tested_labels[0] directly,
                    which was independently verified to match true_label's
                    resolved string in every row)

Protocol is intentionally identical to E9_probe_per_subset.py /
E11_length_heuristic.py so results are directly comparable:
  - CLIP-B/32, post-projection embeddings, concatenated [image_embed, text_embed]
    features, MLPClassifier(hidden_layer_sizes=(512,256), max_iter=1000,
    early_stopping=True), 80/20 pair-wise split, seed 42.
  - Length heuristic: predict TRUE caption is whichever of the two has fewer
    words (also tries "longer is true", takes the max), ties scored 0.5/0.5.

Output: E12_probe_results.json
"""

import json
import os
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

CONFIGS = {
    "swap_obj": "HuggingFaceM4/SugarCrepe_swap_obj",
    "swap_att": "HuggingFaceM4/SugarCrepe_swap_att",
    "replace_obj": "HuggingFaceM4/SugarCrepe_replace_obj",
    "replace_att": "HuggingFaceM4/SugarCrepe_replace_att",
    "replace_rel": "HuggingFaceM4/SugarCrepe_replace_rel",
    "add_obj": "HuggingFaceM4/SugarCrepe_add_obj",
    "add_att": "HuggingFaceM4/SugarCrepe_add_att",
}

# from E9_results.json / E11_results.json, for direct side-by-side reporting only
PRIOR_WDS_PROBE = {
    "add_obj": 0.805, "add_att": 0.705, "replace_rel": 0.610,
    "replace_obj": 0.582, "swap_att": 0.515, "replace_att": 0.500,
    "swap_obj": 0.500,
}
PRIOR_WDS_LENHEUR = {
    "add_obj": 0.988, "add_att": 0.991, "replace_rel": 0.541,
    "replace_obj": 0.549, "swap_att": 0.511, "replace_att": 0.510,
    "swap_obj": 0.524,
}


def word_count(text):
    return len(text.strip().split())


def length_heuristic(dataset):
    correct_shorter = correct_longer = ties = 0.0
    len_true_total = len_false_total = 0
    n = len(dataset)
    for ex in dataset:
        true_cap, false_cap = ex["tested_labels"][0], ex["tested_labels"][1]
        lt, lf = word_count(true_cap), word_count(false_cap)
        len_true_total += lt
        len_false_total += lf
        if lt == lf:
            ties += 1
            correct_shorter += 0.5
            correct_longer += 0.5
        elif lt < lf:
            correct_shorter += 1.0
        else:
            correct_longer += 1.0
    shorter_acc, longer_acc = correct_shorter / n, correct_longer / n
    return {
        "n_pairs": n,
        "mean_len_true_caption_words": len_true_total / n,
        "mean_len_false_caption_words": len_false_total / n,
        "n_ties_same_length": ties,
        "best_length_heuristic_accuracy": max(shorter_acc, longer_acc),
        "best_heuristic_direction": "shorter_is_true" if shorter_acc >= longer_acc else "longer_is_true",
    }


def mlp_probe(dataset, model, processor, device):
    n = len(dataset)
    post_samples, labels = [], []
    with torch.no_grad():
        for ex in dataset:
            img = ex["image"].convert("RGB")
            true_cap, false_cap = ex["tested_labels"][0], ex["tested_labels"][1]

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

    X = np.array(post_samples)
    y = np.array(labels)

    pair_idx = np.arange(n)
    train_pairs, test_pairs = train_test_split(pair_idx, test_size=0.2, random_state=42)
    train_idx = np.concatenate([train_pairs * 2, train_pairs * 2 + 1])
    test_idx = np.concatenate([test_pairs * 2, test_pairs * 2 + 1])

    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000,
                         random_state=42, early_stopping=True)
    clf.fit(X[train_idx], y[train_idx])
    acc = clf.score(X[test_idx], y[test_idx])
    return {"held_out_probe_accuracy": float(acc),
            "train_pairs": len(train_pairs), "test_pairs": len(test_pairs)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("Loading CLIP-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    torch.manual_seed(42)
    np.random.seed(42)

    results = {}
    for cat, repo in CONFIGS.items():
        print(f"\n=== {cat} ({repo}) ===")
        ds = load_dataset(repo, split="test")
        lh = length_heuristic(ds)
        print("length heuristic:", json.dumps(lh, indent=2))
        probe = mlp_probe(ds, model, processor, device)
        print("mlp probe:", json.dumps(probe, indent=2))

        results[cat] = {
            **lh,
            **probe,
            "prior_wds_sugarcrepe_probe_accuracy": PRIOR_WDS_PROBE.get(cat),
            "prior_wds_sugarcrepe_lenheur_accuracy": PRIOR_WDS_LENHEUR.get(cat),
        }
        with open("E12_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY: HFM4 vs. prior wds_sugarcrepe results ===")
    print(f"  {'Category':13s} {'n':>5s} {'HFM4 probe':>11s} {'wds probe':>10s} "
          f"{'HFM4 lenheur':>13s} {'wds lenheur':>12s}")
    for cat, r in results.items():
        print(f"  {cat:13s} {r['n_pairs']:5d} {r['held_out_probe_accuracy']:11.3f} "
              f"{(r['prior_wds_sugarcrepe_probe_accuracy'] or float('nan')):10.3f} "
              f"{r['best_length_heuristic_accuracy']:13.3f} "
              f"{(r['prior_wds_sugarcrepe_lenheur_accuracy'] or float('nan')):12.3f}")

    print("\nwrote E12_probe_results.json")


if __name__ == "__main__":
    main()
