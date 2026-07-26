"""
Unified intervention experiment for Table 6 of the paper.

Fills the missing Text/Image/Group scores for:
  - SVD bottom-20% steering (alpha=2.0)
  - CAV steering (alpha=5.0)
  - Random subspace control (alpha=5.0)
  - Mahalanobis reweighting (for verification)

Also reports bootstrap p-values vs. baseline for each.

All interventions run on the SAME pre-extracted features so comparisons are
apples-to-apples; the model is loaded once.
"""
import os
import json
import warnings

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 10_000


def evaluate(v0, v1, t0, t1):
    """Return [text, image, group] booleans for a single Winoground 2x2."""
    s00 = (v0 @ t0).item()
    s01 = (v0 @ t1).item()
    s10 = (v1 @ t0).item()
    s11 = (v1 @ t1).item()
    t_ok = (s00 > s01) and (s11 > s10)
    i_ok = (s00 > s10) and (s11 > s01)
    return [int(t_ok), int(i_ok), int(t_ok and i_ok)]


def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def aggregate(scores_array):
    """scores_array: (N, 3) of {0,1}.  Returns dict with text/image/group means."""
    return {
        "text_score": float(scores_array[:, 0].mean()),
        "image_score": float(scores_array[:, 1].mean()),
        "group_score": float(scores_array[:, 2].mean()),
    }


def bootstrap_p_value(method_groups, baseline_groups, n_boot=N_BOOTSTRAP, rng=None):
    """One-sided bootstrap: P(method_acc - baseline_acc > 0) under resampling."""
    rng = rng or np.random.default_rng(SEED)
    method_groups = np.asarray(method_groups, dtype=float)
    baseline_groups = np.asarray(baseline_groups, dtype=float)
    n = len(method_groups)
    diffs = method_groups - baseline_groups  # paired
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = diffs[idx].mean()
    # P-value: probability the mean difference is <= 0 (one-sided test for improvement)
    p = float(np.mean(boots <= 0.0))
    return p, float(diffs.mean())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading CLIP-ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading Winoground ...")
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    n = len(dataset)

    # Pre-extract features (pre-projection image, post-projection text)
    pre_img0_list, pre_img1_list = [], []
    text_pair_list = []  # (2, 512) post-projection L2-normed text embeddings

    print("Extracting features ...")
    with torch.no_grad():
        for ex in tqdm(dataset):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            t_inputs = processor(text=[cap0, cap1], return_tensors="pt",
                                 padding=True, truncation=True).to(DEVICE)
            t_out = model.text_model(**t_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = normalize(t_embeds)
            text_pair_list.append(t_embeds.cpu())

            i_inputs = processor(images=[img0, img1], return_tensors="pt").to(DEVICE)
            v_out = model.vision_model(**i_inputs)
            pre_proj = v_out.pooler_output  # (2, 768)
            pre_img0_list.append(pre_proj[0].cpu())
            pre_img1_list.append(pre_proj[1].cpu())

    pre_img0 = torch.stack(pre_img0_list)  # (N, 768)
    pre_img1 = torch.stack(pre_img1_list)  # (N, 768)
    text_pairs = torch.stack(text_pair_list)  # (N, 2, 512)

    # SVD of visual projection matrix W in R^{512 x 768}
    W = model.visual_projection.weight.detach().cpu().numpy()
    U_np, S_np, _Vh_np = np.linalg.svd(W, full_matrices=False)  # U: (512,512), S: (512,)
    U = torch.tensor(U_np, dtype=torch.float32).to(DEVICE)
    S_t = torch.tensor(S_np, dtype=torch.float32).to(DEVICE)
    d_out = U.shape[0]  # 512

    # Bottom 20% indices (in *singular value* order)
    sorted_indices = np.argsort(S_np)  # ascending
    k_bottom = int(0.20 * d_out)  # 102
    bottom_idx = torch.tensor(sorted_indices[:k_bottom], dtype=torch.long).to(DEVICE)

    # Random orthogonal subspace of same dim as bottom 20%
    rng_torch = torch.Generator()
    rng_torch.manual_seed(SEED)
    R_full = torch.randn(d_out, d_out, generator=rng_torch)
    Q, _ = torch.linalg.qr(R_full)
    rand_subspace_basis = Q[:, :k_bottom].to(DEVICE)  # (512, k_bottom) orthonormal columns

    # CAV: pre-projection direction trained on first 100 pairs
    indices = np.arange(n)
    rng.shuffle(indices)
    train_idx = indices[:100]
    test_idx = indices[100:]
    diffs = pre_img0[train_idx] - pre_img1[train_idx]  # (100, 768)
    cav = (diffs.mean(0)).to(DEVICE)
    cav = cav / cav.norm()
    rand_pre = torch.randn(768, generator=rng_torch).to(DEVICE)
    rand_pre = rand_pre / rand_pre.norm()

    # ---- All interventions evaluated on the SAME indices (the test split) ----
    # We evaluate all 400 pairs for baseline / SVD / Mahalanobis / Random subspace
    # (they don't need a CAV training split). For CAV we restrict to test_idx.
    eval_idx_full = np.arange(n)
    eval_idx_test = test_idx

    # Storage: per-pair (text, image, group) for each method
    methods = {
        "baseline": [],
        "svd_bottom20_alpha2": [],
        "mahalanobis": [],
        "rand_subspace_alpha5": [],
        # CAV indexed under test_idx
        "cav_alpha5_test": [],
        "baseline_test": [],
        "rand_pre_alpha5_test": [],
    }

    EPSILON = 1e-3
    S_inv = 1.0 / np.clip(S_np, a_min=EPSILON, a_max=None)
    S_inv_t = torch.tensor(S_inv, dtype=torch.float32).to(DEVICE)

    def project_cosine(v_post0, v_post1, t_pair):
        """Both v_post* are (512,) raw post-projection (NOT yet normalized)."""
        v0n = normalize(v_post0)
        v1n = normalize(v_post1)
        t0n = t_pair[0].to(DEVICE)
        t1n = t_pair[1].to(DEVICE)
        return evaluate(v0n, v1n, t0n, t1n)

    print("Running all interventions ...")
    with torch.no_grad():
        for i in tqdm(range(n)):
            x0 = pre_img0[i].to(DEVICE)  # (768,)
            x1 = pre_img1[i].to(DEVICE)
            t_pair = text_pairs[i]  # (2, 512)
            t0 = t_pair[0].to(DEVICE)
            t1 = t_pair[1].to(DEVICE)

            # ---------- Baseline ----------
            v0 = model.visual_projection(x0.unsqueeze(0)).squeeze(0)
            v1 = model.visual_projection(x1.unsqueeze(0)).squeeze(0)
            methods["baseline"].append(project_cosine(v0, v1, t_pair))

            # ---------- SVD bottom 20% steering, alpha=2.0 ----------
            # Project to SVD basis, scale bottom-20% dims by (1+alpha), invert
            alpha_svd = 2.0
            v0_svd = v0 @ U  # (512,)
            v1_svd = v1 @ U
            scale = torch.ones(d_out, device=DEVICE)
            scale[bottom_idx] = 1.0 + alpha_svd
            v0_svd = v0_svd * scale
            v1_svd = v1_svd * scale
            v0_back = v0_svd @ U.T
            v1_back = v1_svd @ U.T
            methods["svd_bottom20_alpha2"].append(project_cosine(v0_back, v1_back, t_pair))

            # ---------- Mahalanobis (inverse singular value reweight) ----------
            v0_m = (v0 @ U) * S_inv_t
            v1_m = (v1 @ U) * S_inv_t
            t0_m = (t0 @ U) * S_inv_t
            t1_m = (t1 @ U) * S_inv_t
            v0_m = normalize(v0_m); v1_m = normalize(v1_m)
            t0_m = normalize(t0_m); t1_m = normalize(t1_m)
            methods["mahalanobis"].append(evaluate(v0_m, v1_m, t0_m, t1_m))

            # ---------- Random subspace control, alpha=5.0 (post-projection) ----------
            # Boost a random orthogonal subspace of dim = bottom_20% in the post-projection space
            alpha_rand = 5.0
            # Project onto random subspace, scale, add back
            v0_proj = (v0 @ rand_subspace_basis) @ rand_subspace_basis.T  # component in rand subspace
            v1_proj = (v1 @ rand_subspace_basis) @ rand_subspace_basis.T
            v0_rand = v0 + alpha_rand * v0_proj
            v1_rand = v1 + alpha_rand * v1_proj
            methods["rand_subspace_alpha5"].append(project_cosine(v0_rand, v1_rand, t_pair))

        # CAV evaluated on test_idx (held-out)
        for idx in tqdm(eval_idx_test, desc="CAV/RandPre"):
            x0 = pre_img0[idx].to(DEVICE)
            x1 = pre_img1[idx].to(DEVICE)
            t_pair = text_pairs[idx]

            # baseline on test split (for paired bootstrap with CAV)
            v0 = model.visual_projection(x0.unsqueeze(0)).squeeze(0)
            v1 = model.visual_projection(x1.unsqueeze(0)).squeeze(0)
            methods["baseline_test"].append(project_cosine(v0, v1, t_pair))

            # CAV alpha=5
            alpha_c = 5.0
            x0_s = x0 + alpha_c * cav
            x1_s = x1 + alpha_c * cav
            v0c = model.visual_projection(x0_s.unsqueeze(0)).squeeze(0)
            v1c = model.visual_projection(x1_s.unsqueeze(0)).squeeze(0)
            methods["cav_alpha5_test"].append(project_cosine(v0c, v1c, t_pair))

            # Pre-projection random vector, alpha=5 (matched control to CAV)
            x0_r = x0 + alpha_c * rand_pre
            x1_r = x1 + alpha_c * rand_pre
            v0r = model.visual_projection(x0_r.unsqueeze(0)).squeeze(0)
            v1r = model.visual_projection(x1_r.unsqueeze(0)).squeeze(0)
            methods["rand_pre_alpha5_test"].append(project_cosine(v0r, v1r, t_pair))

    # Convert to arrays
    arrs = {k: np.array(v) for k, v in methods.items()}
    summary = {k: aggregate(arrs[k]) for k in arrs}

    # Bootstrap p-values vs baseline (paired, same indices)
    bootstraps = {}
    base_groups_full = arrs["baseline"][:, 2]
    for k in ["svd_bottom20_alpha2", "mahalanobis", "rand_subspace_alpha5"]:
        p, dmean = bootstrap_p_value(arrs[k][:, 2], base_groups_full, rng=rng)
        bootstraps[k] = {"p_value_vs_baseline": p, "mean_diff": dmean}

    base_groups_test = arrs["baseline_test"][:, 2]
    for k in ["cav_alpha5_test", "rand_pre_alpha5_test"]:
        p, dmean = bootstrap_p_value(arrs[k][:, 2], base_groups_test, rng=rng)
        bootstraps[k] = {"p_value_vs_baseline": p, "mean_diff": dmean}

    out = {
        "model": "openai/clip-vit-base-patch32",
        "n_pairs": int(n),
        "n_test_pairs_for_cav": int(len(test_idx)),
        "k_bottom_20pct": int(k_bottom),
        "alpha_svd": 2.0,
        "alpha_cav": 5.0,
        "alpha_rand_subspace": 5.0,
        "summary": summary,
        "bootstrap_p_values": bootstraps,
    }
    print("\n" + json.dumps(out, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "..", "05_results",
                            "unified_interventions.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
