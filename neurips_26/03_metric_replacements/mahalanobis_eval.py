import torch
import numpy as np
import json
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ['OMP_NUM_THREADS'] = '4'

def main():
    print("Running Mahalanobis scaling zero-shot on Winoground...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Get Visual Projection matrix SVD
    W = model.visual_projection.weight.detach().cpu().numpy() # (512, 768)
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    
    # To reweight cosine similarity, we scale representations in the SVD basis
    # Normally: sim(x, y) = (x U) @ (y U)^T
    # Mahalanobis SVD metric: sim_M(x, y) = (x U S^{-1}) @ (y U S^{-1})^T
    # Since we want to test if inverse singular value scaling helps, we construct the scaling matrix
    
    # Clip tiny singular values to avoid explosion
    epsilon = 1e-3
    S_clipped = np.clip(S, a_min=epsilon, a_max=None)
    S_inv = 1.0 / S_clipped
    
    U_t = torch.tensor(U.T).float().to(device)
    S_inv_t = torch.tensor(S_inv).float().to(device)
    
    dataset_eval = load_dataset("haideraltahan/wds_winoground", split="test")
    
    text_scores = []
    image_scores = []
    group_scores = []
    
    # Baseline for reference
    text_scores_base = []
    image_scores_base = []
    group_scores_base = []
    
    with torch.no_grad():
        for example in tqdm(dataset_eval):
            img0 = example["0.webp"].convert("RGB")
            img1 = example["1.webp"].convert("RGB")
            cap0 = example["npy"][0]
            cap1 = example["npy"][1]
            
            # Texts
            t_inputs = processor(text=[cap0, cap1], return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = model.text_model(**t_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            
            # Images
            i_inputs = processor(images=[img0, img1], return_tensors="pt").to(device)
            v_out = model.vision_model(**i_inputs)
            v_embeds = model.visual_projection(v_out.pooler_output)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)
            
            # Baseline zero-shot cosine similarities
            sim_00_base = (v_embeds[0] @ t_embeds[0].T).item()
            sim_01_base = (v_embeds[0] @ t_embeds[1].T).item()
            sim_10_base = (v_embeds[1] @ t_embeds[0].T).item()
            sim_11_base = (v_embeds[1] @ t_embeds[1].T).item()
            
            t_score_base = (sim_00_base > sim_01_base) and (sim_11_base > sim_10_base)
            i_score_base = (sim_00_base > sim_10_base) and (sim_11_base > sim_01_base)
            
            text_scores_base.append(1 if t_score_base else 0)
            image_scores_base.append(1 if i_score_base else 0)
            group_scores_base.append(1 if (t_score_base and i_score_base) else 0)
            
            # MAHALANOBIS scaling
            # 1. Project to SVD basis
            # 2. Scale by Inverse Singular Values
            # 3. L2 Normalize again (because cosine similarity operates on normalized vectors)
            
            v0_m = (v_embeds[0] @ U_t.T) * S_inv_t
            v1_m = (v_embeds[1] @ U_t.T) * S_inv_t
            t0_m = (t_embeds[0] @ U_t.T) * S_inv_t
            t1_m = (t_embeds[1] @ U_t.T) * S_inv_t
            
            v0_m = v0_m / v0_m.norm()
            v1_m = v1_m / v1_m.norm()
            t0_m = t0_m / t0_m.norm()
            t1_m = t1_m / t1_m.norm()
            
            sim_00 = (v0_m @ t0_m.T).item()
            sim_01 = (v0_m @ t1_m.T).item()
            sim_10 = (v1_m @ t0_m.T).item()
            sim_11 = (v1_m @ t1_m.T).item()
            
            t_score = (sim_00 > sim_01) and (sim_11 > sim_10)
            i_score = (sim_00 > sim_10) and (sim_11 > sim_01)
            
            text_scores.append(1 if t_score else 0)
            image_scores.append(1 if i_score else 0)
            group_scores.append(1 if (t_score and i_score) else 0)
            
    results = {
        "baseline_text_score": float(np.mean(text_scores_base)),
        "baseline_image_score": float(np.mean(image_scores_base)),
        "baseline_group_score": float(np.mean(group_scores_base)),
        
        "mahalanobis_text_score": float(np.mean(text_scores)),
        "mahalanobis_image_score": float(np.mean(image_scores)),
        "mahalanobis_group_score": float(np.mean(group_scores))
    }
    
    print("Zero-Shot Results with Mahalanobis Scaling:", results)
    with open("mahalanobis_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
