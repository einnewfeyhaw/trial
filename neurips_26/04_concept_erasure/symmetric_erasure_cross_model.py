"""
Symmetric (image-side) mean-erasure across five contrastive models.

Currently the paper only reports symmetric erasure for CLIP-ViT-B/32 (Section 5.4).
This script replicates it across all five models in Table 5 to confirm symmetry
is universal:
  Compute I_mean = (I_0 + I_1)/2, project text embeddings orthogonally away
  from I_mean, then re-evaluate Winoground 2x2.
"""
import os
import json
import warnings

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModel, AutoProcessor,
    CLIPModel, CLIPProcessor,
)

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(v0, v1, t0, t1):
    s00 = (v0 @ t0).item()
    s01 = (v0 @ t1).item()
    s10 = (v1 @ t0).item()
    s11 = (v1 @ t1).item()
    t_ok = (s00 > s01) and (s11 > s10)
    i_ok = (s00 > s10) and (s11 > s01)
    return [int(t_ok), int(i_ok), int(t_ok and i_ok)]


def normalize(x, eps=1e-12):
    return x / (x.norm() + eps)


MODELS = [
    {
        "name": "CLIP-ViT-B/32 (OpenAI)",
        "id": "openai/clip-vit-base-patch32",
        "loader": "clip",
    },
    {
        "name": "CLIP-ViT-B/32 (LAION)",
        "id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "loader": "clip",
    },
    {
        "name": "CLIP-ViT-L/14 (OpenAI)",
        "id": "openai/clip-vit-large-patch14",
        "loader": "clip",
    },
    {
        "name": "SigLIP-base",
        "id": "google/siglip-base-patch16-224",
        "loader": "auto",
    },
    {
        "name": "SigLIP-SO400M",
        "id": "google/siglip-so400m-patch14-384",
        "loader": "auto",
    },
]


def get_embeddings(model, processor, loader_kind, dataset):
    """Return per-pair (v0, v1, t0, t1) tensors as four lists, all L2-normalized."""
    v0_list, v1_list, t0_list, t1_list = [], [], [], []
    with torch.no_grad():
        for ex in tqdm(dataset, desc="extracting"):
            img0 = ex["0.webp"].convert("RGB")
            img1 = ex["1.webp"].convert("RGB")
            cap0, cap1 = ex["npy"][0], ex["npy"][1]

            if loader_kind == "clip":
                t_inputs = processor(text=[cap0, cap1], return_tensors="pt",
                                     padding=True, truncation=True).to(DEVICE)
                t_out = model.text_model(**t_inputs)
                t_embeds = model.text_projection(t_out.pooler_output)
                t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)

                i_inputs = processor(images=[img0, img1], return_tensors="pt").to(DEVICE)
                v_out = model.vision_model(**i_inputs)
                v_embeds = model.visual_projection(v_out.pooler_output)
                v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)
            else:
                inputs = processor(text=[cap0, cap1], images=[img0, img1],
                                   return_tensors="pt", padding="max_length",
                                   truncation=True).to(DEVICE)
                out = model(**inputs)
                t_embeds = out.text_embeds
                v_embeds = out.image_embeds
                t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
                v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)

            v0_list.append(v_embeds[0].cpu())
            v1_list.append(v_embeds[1].cpu())
            t0_list.append(t_embeds[0].cpu())
            t1_list.append(t_embeds[1].cpu())
    return v0_list, v1_list, t0_list, t1_list


def main():
    print("Loading Winoground ...")
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    n = len(dataset)
    all_results = {}

    for entry in MODELS:
        name = entry["name"]
        mid = entry["id"]
        print(f"\n=========================================")
        print(f"Model: {name} ({mid})")
        print(f"=========================================")
        try:
            if entry["loader"] == "clip":
                model = CLIPModel.from_pretrained(mid).to(DEVICE).eval()
                processor = CLIPProcessor.from_pretrained(mid)
            else:
                model = AutoModel.from_pretrained(mid).to(DEVICE).eval()
                processor = AutoProcessor.from_pretrained(mid)
        except Exception as e:
            print(f"Failed to load {mid}: {e}")
            continue

        v0_list, v1_list, t0_list, t1_list = get_embeddings(
            model, processor, entry["loader"], dataset)

        v0_all = torch.stack(v0_list).to(DEVICE)
        v1_all = torch.stack(v1_list).to(DEVICE)
        t0_all = torch.stack(t0_list).to(DEVICE)
        t1_all = torch.stack(t1_list).to(DEVICE)

        baseline_scores = []
        text_erasure_scores = []   # erase C_mean from images (standard)
        image_erasure_scores = []  # erase I_mean from texts (symmetric)

        with torch.no_grad():
            for i in range(n):
                v0 = v0_all[i]; v1 = v1_all[i]
                t0 = t0_all[i]; t1 = t1_all[i]

                baseline_scores.append(evaluate(v0, v1, t0, t1))

                # Text-side mean erasure (standard)
                c_mean = (t0 + t1) / 2.0
                c_hat = c_mean / (c_mean.norm() + 1e-12)
                v0_e = normalize(v0 - (v0 @ c_hat) * c_hat)
                v1_e = normalize(v1 - (v1 @ c_hat) * c_hat)
                text_erasure_scores.append(evaluate(v0_e, v1_e, t0, t1))

                # Image-side mean erasure (symmetric)
                i_mean = (v0 + v1) / 2.0
                i_hat = i_mean / (i_mean.norm() + 1e-12)
                t0_e = normalize(t0 - (t0 @ i_hat) * i_hat)
                t1_e = normalize(t1 - (t1 @ i_hat) * i_hat)
                image_erasure_scores.append(evaluate(v0, v1, t0_e, t1_e))

        b = np.array(baseline_scores)
        te = np.array(text_erasure_scores)
        ie = np.array(image_erasure_scores)

        all_results[name] = {
            "model_id": mid,
            "n_pairs": n,
            "baseline": {
                "text_score": float(b[:, 0].mean()),
                "image_score": float(b[:, 1].mean()),
                "group_score": float(b[:, 2].mean()),
            },
            "text_side_erasure": {
                "text_score": float(te[:, 0].mean()),
                "image_score": float(te[:, 1].mean()),
                "group_score": float(te[:, 2].mean()),
            },
            "image_side_erasure": {
                "text_score": float(ie[:, 0].mean()),
                "image_score": float(ie[:, 1].mean()),
                "group_score": float(ie[:, 2].mean()),
            },
        }
        print(json.dumps(all_results[name], indent=2))

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(os.path.dirname(__file__), "..", "05_results",
                            "symmetric_erasure_cross_model.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
