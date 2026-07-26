import torch
import numpy as np
import json
import os
import random
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'

def evaluate(v_0, v_1, t_0, t_1):
    s00 = (v_0 @ t_0).item()
    s01 = (v_0 @ t_1).item()
    s10 = (v_1 @ t_0).item()
    s11 = (v_1 @ t_1).item()
    t_match = (s00 > s01) and (s11 > s10)
    i_match = (s00 > s10) and (s11 > s01)
    return [1 if t_match else 0, 1 if i_match else 0, 1 if (t_match and i_match) else 0]

def main():
    print("Evaluating Winoground Concept Erasure (Matched vs Random Control)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    model.eval()
    
    # Set seed so random indices are reproducible
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    n = len(dataset)
    
    # To do random mean subtraction, we first need to pre-extract all text features
    print("Pre-extracting Text Embeddings for Random Control...")
    t0_list, t1_list = [], []
    v0_list, v1_list = [], []
    
    with torch.no_grad():
        for example in tqdm(dataset):
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
            
            t0_list.append(t_embeds[0])
            t1_list.append(t_embeds[1])
            v0_list.append(v_embeds[0])
            v1_list.append(v_embeds[1])
            
    counts = {"baseline": [0,0,0], "strat_a_matched": [0,0,0], "strat_c_random": [0,0,0]} # T, I, G
    
    print("Evaluating Strategies...")
    for i in range(n):
        v0, v1 = v0_list[i], v1_list[i]
        t0, t1 = t0_list[i], t1_list[i]
        
        # 1. Baseline
        base_scores = evaluate(v0, v1, t0, t1)
        counts["baseline"][0] += base_scores[0]
        counts["baseline"][1] += base_scores[1]
        counts["baseline"][2] += base_scores[2]
        
        # 2. Strategy A (Matched Mean Subtraction)
        c_mean_matched = (t0 + t1) / 2.0
        c_mean_matched = c_mean_matched / c_mean_matched.norm()
        
        v0_a = v0 - (v0 @ c_mean_matched) * c_mean_matched
        v1_a = v1 - (v1 @ c_mean_matched) * c_mean_matched
        v0_a = v0_a / v0_a.norm()
        v1_a = v1_a / v1_a.norm()
        
        a_scores = evaluate(v0_a, v1_a, t0, t1)
        counts["strat_a_matched"][0] += a_scores[0]
        counts["strat_a_matched"][1] += a_scores[1]
        counts["strat_a_matched"][2] += a_scores[2]
        
        # 3. Strategy C (Random Mean Subtraction)
        # Pick a random index j != i
        j = random.choice([x for x in range(n) if x != i])
        t0_rand, t1_rand = t0_list[j], t1_list[j]
        
        c_mean_rand = (t0_rand + t1_rand) / 2.0
        c_mean_rand = c_mean_rand / c_mean_rand.norm()
        
        v0_c = v0 - (v0 @ c_mean_rand) * c_mean_rand
        v1_c = v1 - (v1 @ c_mean_rand) * c_mean_rand
        v0_c = v0_c / v0_c.norm()
        v1_c = v1_c / v1_c.norm()
        
        c_scores = evaluate(v0_c, v1_c, t0, t1)
        counts["strat_c_random"][0] += c_scores[0]
        counts["strat_c_random"][1] += c_scores[1]
        counts["strat_c_random"][2] += c_scores[2]
        
    results = {
        "baseline": {
            "text_score": counts["baseline"][0] / n,
            "image_score": counts["baseline"][1] / n,
            "group_score": counts["baseline"][2] / n
        },
        "strategy_a_matched_mean_erasure": {
            "text_score": counts["strat_a_matched"][0] / n,
            "image_score": counts["strat_a_matched"][1] / n,
            "group_score": counts["strat_a_matched"][2] / n
        },
        "strategy_c_random_mean_erasure": {
            "text_score": counts["strat_c_random"][0] / n,
            "image_score": counts["strat_c_random"][1] / n,
            "group_score": counts["strat_c_random"][2] / n
        }
    }
    
    print("Random Control Results:", json.dumps(results, indent=2))
    with open("erasure_random_control.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
