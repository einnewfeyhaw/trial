import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

def get_group_score(img0_sims, img1_sims):
    # img0_sims: [sim(I0, C0), sim(I0, C1)]
    # img1_sims: [sim(I1, C0), sim(I1, C1)]
    # Text score: C0 matches I0 better than I1, and C1 matches I1 better than I0.
    # Image score: I0 matches C0 better than C1, and I1 matches C1 better than C0.
    # Group score: both text score and image score are true.
    text_score = (img0_sims[0] > img1_sims[0]) and (img1_sims[1] > img0_sims[1])
    image_score = (img0_sims[0] > img0_sims[1]) and (img1_sims[1] > img1_sims[0])
    return 1.0 if (text_score and image_score) else 0.0

def main():
    print("Loading model and dataset for Phase 2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    
    # Pre-extract all pre-projection image features and text features
    pre_img0_list = []
    pre_img1_list = []
    text_embeds_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            img0 = example["0.webp"].convert("RGB")
            img1 = example["1.webp"].convert("RGB")
            cap0 = example["npy"][0]
            cap1 = example["npy"][1]
            
            # Text embeds
            text_inputs = processor(text=[cap0, cap1], return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = model.text_model(**text_inputs)
            text_embed = model.text_projection(text_outputs.pooler_output)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embeds_list.append(text_embed.cpu()) # shape: (2, 512)
            
            # Image pre-proj
            img_inputs = processor(images=[img0, img1], return_tensors="pt").to(device)
            vision_outputs = model.vision_model(**img_inputs)
            pre_proj = vision_outputs.pooler_output # shape: (2, 768)
            pre_img0_list.append(pre_proj[0].cpu())
            pre_img1_list.append(pre_proj[1].cpu())

    pre_img0 = torch.stack(pre_img0_list) # (400, 768)
    pre_img1 = torch.stack(pre_img1_list) # (400, 768)
    text_embeds = torch.stack(text_embeds_list) # (400, 2, 512)
    
    # Train/Test Split (100 train, 300 test)
    indices = np.arange(400)
    np.random.shuffle(indices)
    train_idx = indices[:100]
    test_idx = indices[100:]
    
    # Compute CAV on Train set
    diffs = pre_img0[train_idx] - pre_img1[train_idx]
    cav = diffs.mean(dim=0) # (768,)
    cav = cav / cav.norm()
    cav = cav.to(device)
    
    # Random Control Vector
    rand_vec = torch.randn(768)
    rand_vec = rand_vec / rand_vec.norm()
    rand_vec = rand_vec.to(device)
    
    alphas = np.arange(0.0, 5.5, 0.5).tolist()
    
    def evaluate_steering(steering_vec, alpha, eval_indices):
        scores = []
        with torch.no_grad():
            for idx in eval_indices:
                x0 = pre_img0[idx].to(device)
                x1 = pre_img1[idx].to(device)
                t_embed = text_embeds[idx].to(device) # (2, 512)
                
                # Steer
                x0_steered = x0 + alpha * steering_vec
                x1_steered = x1 + alpha * steering_vec
                
                # Forward through projection
                z0 = model.visual_projection(x0_steered.unsqueeze(0)).squeeze(0)
                z1 = model.visual_projection(x1_steered.unsqueeze(0)).squeeze(0)
                
                z0 = z0 / z0.norm()
                z1 = z1 / z1.norm()
                
                sim_00 = (z0 @ t_embed[0]).item()
                sim_01 = (z0 @ t_embed[1]).item()
                sim_10 = (z1 @ t_embed[0]).item()
                sim_11 = (z1 @ t_embed[1]).item()
                
                scores.append(get_group_score([sim_00, sim_01], [sim_10, sim_11]))
        return scores

    results = {"cav_sweeps": {}, "rand_sweeps": {}}
    
    # Sweep CAV
    print("Evaluating CAV steering...")
    for a in alphas:
        sc = evaluate_steering(cav, a, test_idx)
        results["cav_sweeps"][a] = np.mean(sc)
        
    # Sweep Random
    print("Evaluating Random steering...")
    for a in alphas:
        sc = evaluate_steering(rand_vec, a, test_idx)
        results["rand_sweeps"][a] = np.mean(sc)
        
    baseline_scores = evaluate_steering(cav, 0.0, test_idx)
    baseline_acc = np.mean(baseline_scores)
    results["baseline"] = baseline_acc
    
    best_cav_alpha = max(results["cav_sweeps"], key=results["cav_sweeps"].get)
    best_cav_scores = evaluate_steering(cav, best_cav_alpha, test_idx)
    results["best_cav_alpha"] = best_cav_alpha
    results["best_cav_acc"] = np.mean(best_cav_scores)
    
    # Bootstrap Test
    print("Running bootstrap test...")
    n_bootstraps = 10000
    count_better = 0
    diff_baseline = np.mean(best_cav_scores) - np.mean(baseline_scores)
    
    # Under null hypothesis, the difference is 0. 
    # We bootstrap the difference to see if it's strictly > 0.
    for _ in tqdm(range(n_bootstraps)):
        boot_idx = np.random.choice(len(test_idx), len(test_idx), replace=True)
        boot_baseline = np.mean(np.array(baseline_scores)[boot_idx])
        boot_cav = np.mean(np.array(best_cav_scores)[boot_idx])
        if boot_cav > boot_baseline:
            count_better += 1
            
    p_value = 1.0 - (count_better / n_bootstraps)
    results["bootstrap_p_value"] = p_value
    
    print("Results:", results)
    with open("cav_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
