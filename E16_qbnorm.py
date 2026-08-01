"""
E16: QB-Norm (Bogolin et al., CVPR 2022, arXiv:2112.12777) on Winoground.

Addresses Reviewer BTbD: "I am not convinced that instance-specific
intervention is necessary, i.e., that no other method can succeed on these
benchmarks." QB-Norm is a real, published, content-agnostic correction:
each gallery item (here, each Winoground image) gets a hubness penalty
D(I) = sum_b exp(beta * sim(bank_b, I)) computed against a fixed bank of
probe queries, INDEPENDENT of which caption it will later be compared
against. This is exactly the class of correction the paper's Section 5
claims cannot substitute for instance-specific (foil-dependent) mean
erasure -- but it has never been tested. If it fails, that is stronger
evidence than our four self-designed baselines (an established published
method for exactly this problem class also fails). If it succeeds, the
necessity claim must be walked back to a comparative one.

Query bank: true captions from haideraltahan/wds_sugarcrepe pooled across
all 7 categories, deduplicated (~5-8k natural captions derived from COCO,
the standard QB-Norm bank source, and already verified accessible in this
project -- avoids guessing at an unverified COCO-captions dataset name).

Scoring convention matches concept_erasure_eval.py exactly: s00,s01,s10,s11
via dot product of L2-normalized embeddings; text_match = (s00>s01) and
(s11>s10); image_match = (s00>s10) and (s11>s01); group = both.

Output: E16_results.json
"""

import json
import os
import sys
import warnings

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(__file__))
from E2_direct_spectral import subset_of

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BETAS = [1.0, 2.0, 5.0, 10.0, 20.0]
BANK_CATEGORIES = ["swap_obj", "swap_att", "replace_obj", "replace_att",
                    "replace_rel", "add_obj", "add_att"]
BANK_LIMIT_PER_CAT = 1500  # up to ~10.5k captions total, deduplicated
TEXT_BATCH = 256


def normalize(x, eps=1e-12):
    n = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / n


def build_query_bank(model, processor):
    print("Building query bank from haideraltahan/wds_sugarcrepe true captions ...")
    ds = load_dataset("haideraltahan/wds_sugarcrepe", split="test")
    captions = set()
    for cat in BANK_CATEGORIES:
        n_found = 0
        for ex in tqdm(ds, desc=f"  scanning for {cat}"):
            s = subset_of(ex)
            if s is not None and cat in s:
                captions.add(str(ex["npy"][0]))  # true caption only
                n_found += 1
                if n_found >= BANK_LIMIT_PER_CAT:
                    break
    captions = list(captions)
    print(f"  bank size: {len(captions)} unique captions")

    embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), TEXT_BATCH), desc="  embedding bank"):
            batch = captions[i:i + TEXT_BATCH]
            t_in = processor(text=batch, return_tensors="pt", padding=True,
                              truncation=True).to(DEVICE)
            t_out = model.text_model(**t_in)
            t_emb = normalize(model.text_projection(t_out.pooler_output))
            embeds.append(t_emb.cpu())
    bank = torch.cat(embeds, dim=0)  # (B, 512)
    print(f"  bank embedding shape: {tuple(bank.shape)}")
    return bank


def evaluate_pair(s00, s01, s10, s11):
    t_match = (s00 > s01) and (s11 > s10)
    i_match = (s00 > s10) and (s11 > s01)
    return int(t_match), int(i_match), int(t_match and i_match)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    bank = build_query_bank(model, processor).to(DEVICE)  # (B, 512)

    print("Loading haideraltahan/wds_winoground ...")
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")

    # ── Extract all embeddings + hubness scores D(I) for every beta ──
    all_v0, all_v1, all_t0, all_t1 = [], [], [], []
    print("Extracting Winoground embeddings ...")
    with torch.no_grad():
        for ex in tqdm(dataset):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            t_in = processor(text=[cap0, cap1], return_tensors="pt",
                              padding=True, truncation=True).to(DEVICE)
            t_out = model.text_model(**t_in)
            t_emb = normalize(model.text_projection(t_out.pooler_output))

            i_in = processor(images=[img0, img1], return_tensors="pt").to(DEVICE)
            v_out = model.vision_model(**i_in)
            v_emb = normalize(model.visual_projection(v_out.pooler_output))

            all_v0.append(v_emb[0].cpu())
            all_v1.append(v_emb[1].cpu())
            all_t0.append(t_emb[0].cpu())
            all_t1.append(t_emb[1].cpu())

    V0 = torch.stack(all_v0).to(DEVICE)  # (N, 512)
    V1 = torch.stack(all_v1).to(DEVICE)
    T0 = torch.stack(all_t0).to(DEVICE)
    T1 = torch.stack(all_t1).to(DEVICE)
    n = V0.shape[0]
    print(f"Extracted {n} pairs.")

    # sim(bank, image) for every image, once: (N, B)
    bank_sim_v0 = V0 @ bank.T
    bank_sim_v1 = V1 @ bank.T

    results = {"n_pairs": n, "bank_size": int(bank.shape[0])}

    # ── Baseline (beta irrelevant) ──
    base_counts = [0, 0, 0]
    for i in range(n):
        s00 = (V0[i] @ T0[i]).item()
        s01 = (V0[i] @ T1[i]).item()
        s10 = (V1[i] @ T0[i]).item()
        s11 = (V1[i] @ T1[i]).item()
        t, im, g = evaluate_pair(s00, s01, s10, s11)
        base_counts[0] += t; base_counts[1] += im; base_counts[2] += g
    results["baseline"] = {"text_score": base_counts[0] / n,
                            "image_score": base_counts[1] / n,
                            "group_score": base_counts[2] / n}
    print("Baseline:", results["baseline"])

    # ── QB-Norm sweep over beta ──
    results["qbnorm"] = {}
    for beta in BETAS:
        logD0 = torch.logsumexp(beta * bank_sim_v0, dim=1)  # (N,) = log D(I0)
        logD1 = torch.logsumexp(beta * bank_sim_v1, dim=1)  # (N,) = log D(I1)

        counts = [0, 0, 0]
        for i in range(n):
            s00 = beta * (V0[i] @ T0[i]).item() - logD0[i].item()
            s01 = beta * (V0[i] @ T1[i]).item() - logD0[i].item()
            s10 = beta * (V1[i] @ T0[i]).item() - logD1[i].item()
            s11 = beta * (V1[i] @ T1[i]).item() - logD1[i].item()
            t, im, g = evaluate_pair(s00, s01, s10, s11)
            counts[0] += t; counts[1] += im; counts[2] += g

        entry = {"text_score": counts[0] / n, "image_score": counts[1] / n,
                 "group_score": counts[2] / n}
        results["qbnorm"][f"beta_{beta}"] = entry
        print(f"QB-Norm beta={beta}:", entry)

    best_beta = max(results["qbnorm"], key=lambda k: results["qbnorm"][k]["group_score"])
    results["best_beta"] = best_beta
    results["best_group_score"] = results["qbnorm"][best_beta]["group_score"]
    results["mean_erasure_reference"] = {"text_score": 0.31, "image_score": 0.645, "group_score": 0.31}

    print(f"\nBest QB-Norm setting: {best_beta} -> Group Score {results['best_group_score']:.4f}")
    print(f"Baseline Group Score: {results['baseline']['group_score']:.4f}")
    print(f"Mean-erasure (foil-requiring) reference: 0.31")

    with open("E16_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote E16_results.json")


if __name__ == "__main__":
    main()
