import torch
import numpy as np
import json
import os
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    dataset = load_dataset("facebook/winoground", split="test")
    
    results = {
        "baseline": {"text_score": 0, "image_score": 0, "group_score": 0},
        "strategy_a_mean_erasure": {"text_score": 0, "image_score": 0, "group_score": 0},
        "ablation_project_c0_only": {"text_score": 0, "image_score": 0, "group_score": 0},
        "ablation_project_c1_only": {"text_score": 0, "image_score": 0, "group_score": 0},
        "per_category_baseline_group": {},
        "per_category_erasure_group": {}
    }
    
    n = len(dataset)
    counts = {k: [0,0,0] for k in results if not k.startswith("per_category")}
    cat_base = {}
    cat_erasure = {}
    cat_totals = {}
    
    with torch.no_grad():
        for example in tqdm(dataset):
            img0 = example["image_0"].convert("RGB")
            img1 = example["image_1"].convert("RGB")
            cap0 = example["caption_0"]
            cap1 = example["caption_1"]
            tag = example["collapsed_tag"] # Usually Object, Relation, Both
            
            if tag not in cat_totals:
                cat_totals[tag] = 0
                cat_base[tag] = 0
                cat_erasure[tag] = 0
                
            cat_totals[tag] += 1
            
            t_inputs = processor(text=[cap0, cap1], return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = model.text_model(**t_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            
            i_inputs = processor(images=[img0, img1], return_tensors="pt").to(device)
            v_out = model.vision_model(**i_inputs)
            v_embeds = model.visual_projection(v_out.pooler_output)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)
            
            v0, v1 = v_embeds[0], v_embeds[1]
            t0, t1 = t_embeds[0], t_embeds[1]
            
            # Baseline
            base_scores = evaluate(v0, v1, t0, t1)
            counts["baseline"] = [c + s for c, s in zip(counts["baseline"], base_scores)]
            cat_base[tag] += base_scores[2]
            
            # Strategy A (Mean)
            c_mean = (t0 + t1) / 2.0
            c_mean = c_mean / c_mean.norm()
            
            v0_a = v0 - (v0 @ c_mean) * c_mean
            v1_a = v1 - (v1 @ c_mean) * c_mean
            v0_a = v0_a / v0_a.norm()
            v1_a = v1_a / v1_a.norm()
            
            a_scores = evaluate(v0_a, v1_a, t0, t1)
            counts["strategy_a_mean_erasure"] = [c + s for c, s in zip(counts["strategy_a_mean_erasure"], a_scores)]
            cat_erasure[tag] += a_scores[2]
            
            # Ablation C0
            v0_c0 = v0 - (v0 @ t0) * t0
            v1_c0 = v1 - (v1 @ t0) * t0
            v0_c0 = v0_c0 / v0_c0.norm()
            v1_c0 = v1_c0 / v1_c0.norm()
            c0_scores = evaluate(v0_c0, v1_c0, t0, t1)
            counts["ablation_project_c0_only"] = [c + s for c, s in zip(counts["ablation_project_c0_only"], c0_scores)]
            
            # Ablation C1
            v0_c1 = v0 - (v0 @ t1) * t1
            v1_c1 = v1 - (v1 @ t1) * t1
            v0_c1 = v0_c1 / v0_c1.norm()
            v1_c1 = v1_c1 / v1_c1.norm()
            c1_scores = evaluate(v0_c1, v1_c1, t0, t1)
            counts["ablation_project_c1_only"] = [c + s for c, s in zip(counts["ablation_project_c1_only"], c1_scores)]

    for k in counts:
        results[k]["text_score"] = counts[k][0] / n
        results[k]["image_score"] = counts[k][1] / n
        results[k]["group_score"] = counts[k][2] / n
        
    for tag in cat_totals:
        results["per_category_baseline_group"][tag] = cat_base[tag] / cat_totals[tag]
        results["per_category_erasure_group"][tag] = cat_erasure[tag] / cat_totals[tag]
        results["per_category_baseline_group"][tag+"_count"] = cat_totals[tag]
        
    print(json.dumps(results, indent=2))
    with open("winoground_ablations.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
