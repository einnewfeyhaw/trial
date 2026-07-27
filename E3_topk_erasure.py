"""
E3: Top-K Dynamic C_mean Erasure — Zero-Shot Deployable Inference

Addresses Reviewer g6iB Q1:
  "How could the insights from instance-specific mean-erasure be adapted for
   true zero-shot inference without requiring access to the foil caption at
   query time?"

Approach:
  In practical retrieval, the query image is compared against a caption corpus
  of size N.  A global corpus mean is exactly the "generic" intervention that
  Section 5 already shows fails (Table 2, random-pair control, p > 0.2).  The
  correct approximation is a CANDIDATE-SET mean:

    1. Score image against all N captions with standard cosine similarity.
    2. Take the top-K highest-scoring captions.
    3. Compute  C̃_mean = mean of those K embeddings.
    4. Project image away from  C̃_mean  (same formula as mean-erasure).
    5. Re-score the top-K candidates with the projected image and re-rank.

  Intuition: top-K captions retrieved for a query image will almost always
  share the same salient objects (that is why they ranked highly), so their
  average approximates the shared-object direction that confuses 2×2 matching.
  No foil access required — the retrieval step itself identifies the shared
  content.

Evaluation:
  Winoground 400-pair corpus (800 captions, 800 images).
  We pool all 800 caption embeddings as the retrieval corpus.
  For each image:
    * Standard cosine → top-K from the 800-caption pool.
    * Project image away from mean(top-K).
    * Evaluate the standard 2×2 Winoground scores using the projected image.
  K sweep: [1(=baseline), 2, 4, 8, 16, 32].

K=2 special case: if both top-2 are the ground-truth paired captions,
this is IDENTICAL to the paper's mean-erasure.  We measure empirically how
often that happens (hit-rate@2).

Output: E3_results.json
"""

import json
import os
import warnings

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
K_SWEEP = [2, 4, 8, 16, 32]


def normalize(x, eps=1e-12):
    n = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / n


def evaluate_2x2(v0, v1, t0, t1):
    """Standard Winoground 2×2 evaluation. All inputs L2-normalized."""
    s00 = (v0 @ t0).item()
    s01 = (v0 @ t1).item()
    s10 = (v1 @ t0).item()
    s11 = (v1 @ t1).item()
    t_ok = (s00 > s01) and (s11 > s10)
    i_ok = (s00 > s10) and (s11 > s01)
    return int(t_ok), int(i_ok), int(t_ok and i_ok)


def project_away(v, direction):
    """Project v away from unit-norm direction and re-normalize."""
    v_proj = v - (v @ direction) * direction
    return normalize(v_proj)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading Winoground (haideraltahan/wds_winoground) ...")
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    n_pairs = len(dataset)  # 400
    print(f"  {n_pairs} pairs → {n_pairs * 2} images, {n_pairs * 2} captions")

    # ── 1. Embed everything ────────────────────────────────────────────────────
    all_v0, all_v1 = [], []   # per-pair image embeddings
    all_t0, all_t1 = [], []   # per-pair caption embeddings
    all_cap_ids = []          # which pair each caption belongs to

    print("Embedding all images and captions ...")
    with torch.no_grad():
        for pair_idx, ex in enumerate(tqdm(dataset)):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            # Texts
            t_in = processor(text=[cap0, cap1], return_tensors="pt",
                             padding=True, truncation=True).to(DEVICE)
            t_out = model.text_model(**t_in)
            t_emb = normalize(model.text_projection(t_out.pooler_output))

            # Images
            v_in = processor(images=[img0, img1], return_tensors="pt").to(DEVICE)
            v_out = model.vision_model(**v_in)
            v_emb = normalize(model.visual_projection(v_out.pooler_output))

            all_v0.append(v_emb[0].cpu())
            all_v1.append(v_emb[1].cpu())
            all_t0.append(t_emb[0].cpu())
            all_t1.append(t_emb[1].cpu())
            all_cap_ids.extend([pair_idx, pair_idx])  # cap0 and cap1 → same pair

    # Stack: (n_pairs, D)
    V0 = torch.stack(all_v0).to(DEVICE)   # (400, 512)
    V1 = torch.stack(all_v1).to(DEVICE)
    T0 = torch.stack(all_t0).to(DEVICE)
    T1 = torch.stack(all_t1).to(DEVICE)

    # Caption pool: (800, 512) — interleaved [cap0_pair0, cap1_pair0, cap0_pair1, ...]
    cap_pool = torch.zeros(n_pairs * 2, T0.shape[-1], device=DEVICE)
    pair_of_cap = torch.zeros(n_pairs * 2, dtype=torch.long, device=DEVICE)
    for i in range(n_pairs):
        cap_pool[2 * i]     = T0[i]
        cap_pool[2 * i + 1] = T1[i]
        pair_of_cap[2 * i]     = i
        pair_of_cap[2 * i + 1] = i

    # ── 2. Baseline 2×2 evaluation ─────────────────────────────────────────────
    baseline_T, baseline_I, baseline_G = [], [], []
    for i in range(n_pairs):
        t, im, g = evaluate_2x2(V0[i], V1[i], T0[i], T1[i])
        baseline_T.append(t); baseline_I.append(im); baseline_G.append(g)
    baseline_T = np.array(baseline_T)
    baseline_I = np.array(baseline_I)
    baseline_G = np.array(baseline_G)
    print(f"\nBaseline — T:{baseline_T.mean():.4f} I:{baseline_I.mean():.4f} "
          f"G:{baseline_G.mean():.4f}")

    # ── 3. Top-K dynamic erasure sweep ────────────────────────────────────────
    results_per_k = {}

    for K in K_SWEEP:
        text_scores, image_scores, group_scores = [], [], []
        # hit_rate@K: fraction of pairs where BOTH ground-truth captions appear in top-K
        both_in_topk = 0
        at_least_one_in_topk = 0

        with torch.no_grad():
            for i in range(n_pairs):
                t0_i = T0[i]; t1_i = T1[i]

                # ---- Process img0 ----
                sim0 = V0[i] @ cap_pool.T   # (800,)
                topk_idx0 = sim0.topk(K).indices
                top_caps0 = cap_pool[topk_idx0]               # (K, 512)
                c_mean0 = top_caps0.mean(dim=0)
                c_hat0 = c_mean0 / c_mean0.norm().clamp(min=1e-12)
                v0_proj = project_away(V0[i], c_hat0)

                # ---- Process img1 ----
                sim1 = V1[i] @ cap_pool.T
                topk_idx1 = sim1.topk(K).indices
                top_caps1 = cap_pool[topk_idx1]
                c_mean1 = top_caps1.mean(dim=0)
                c_hat1 = c_mean1 / c_mean1.norm().clamp(min=1e-12)
                v1_proj = project_away(V1[i], c_hat1)

                # ---- 2×2 evaluation with projected images ----
                t, im, g = evaluate_2x2(v0_proj, v1_proj, t0_i, t1_i)
                text_scores.append(t)
                image_scores.append(im)
                group_scores.append(g)

                # ---- Hit-rate: are cap0 AND cap1 of this pair in top-K(img0)? ----
                # Ground-truth caption indices for pair i: 2i and 2i+1
                gt_idx_0 = 2 * i
                gt_idx_1 = 2 * i + 1
                topk_set0 = set(topk_idx0.cpu().tolist())
                hit_both = (gt_idx_0 in topk_set0) and (gt_idx_1 in topk_set0)
                hit_one  = (gt_idx_0 in topk_set0) or  (gt_idx_1 in topk_set0)
                both_in_topk    += int(hit_both)
                at_least_one_in_topk += int(hit_one)

        ts = np.array(text_scores)
        is_ = np.array(image_scores)
        gs = np.array(group_scores)

        # Bootstrap CI on group score
        rng = np.random.default_rng(SEED)
        N_BOOT = 10_000
        boot_g = np.array([gs[rng.integers(0, n_pairs, n_pairs)].mean()
                           for _ in range(N_BOOT)])
        ci_lo, ci_hi = float(np.percentile(boot_g, 2.5)), float(np.percentile(boot_g, 97.5))

        # One-sided p-value vs baseline
        diffs = gs - baseline_G
        boot_diffs = np.array([diffs[rng.integers(0, n_pairs, n_pairs)].mean()
                                for _ in range(N_BOOT)])
        p_vs_baseline = float(np.mean(boot_diffs <= 0.0))

        results_per_k[K] = {
            "K": K,
            "text_score":  float(ts.mean()),
            "image_score": float(is_.mean()),
            "group_score": float(gs.mean()),
            "group_ci95":  [ci_lo, ci_hi],
            "group_delta_vs_baseline": float(diffs.mean()),
            "p_one_sided_vs_baseline": p_vs_baseline,
            "relative_gain": float(gs.mean() / baseline_G.mean()) if baseline_G.mean() > 0 else None,
            "hit_rate_both_captions_in_topK_img0": float(both_in_topk / n_pairs),
            "hit_rate_at_least_one_caption_in_topK_img0": float(at_least_one_in_topk / n_pairs),
        }

        print(f"K={K:3d}: T={ts.mean():.4f} I={is_.mean():.4f} G={gs.mean():.4f} "
              f"[{ci_lo:.4f},{ci_hi:.4f}] p={p_vs_baseline:.4f} "
              f"hit_both={both_in_topk/n_pairs:.3f}")

    # ── 4. Save results ────────────────────────────────────────────────────────
    out = {
        "model": "openai/clip-vit-base-patch32",
        "dataset": "haideraltahan/wds_winoground",
        "n_pairs": n_pairs,
        "caption_pool_size": n_pairs * 2,
        "k_sweep": K_SWEEP,
        "baseline": {
            "text_score":  float(baseline_T.mean()),
            "image_score": float(baseline_I.mean()),
            "group_score": float(baseline_G.mean()),
        },
        "results_per_k": results_per_k,
        "note": (
            "C_mean is approximated from top-K captions retrieved by standard cosine "
            "similarity. No foil caption access required at query time. "
            "The hit_rate_both metric measures how often the ground-truth paired "
            "captions both appear in the top-K, providing an oracle upper bound on "
            "how closely the dynamic C_mean approximates the ground-truth C_mean."
        ),
    }

    with open("E3_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → E3_results.json")
    print(json.dumps({k: v["group_score"] for k, v in results_per_k.items()}, indent=2))


if __name__ == "__main__":
    main()
