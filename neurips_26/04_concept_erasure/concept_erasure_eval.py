import torch
import numpy as np
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('universal_tagset', quiet=True)

def extract_nouns(text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    # NN, NNS, NNP, NNPS
    nouns = [word for word, tag in tags if tag.startswith('NN')]
    return nouns

def main():
    print("Evaluating Winoground with Concept Erasure...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    model.eval()
    dataset = load_dataset("haideraltahan/wds_winoground", split="test")
    
    results = {
        "baseline": {"text_score": 0, "image_score": 0, "group_score": 0},
        "strategy_a_mean_erasure": {"text_score": 0, "image_score": 0, "group_score": 0},
        "strategy_b_noun_erasure": {"text_score": 0, "image_score": 0, "group_score": 0}
    }
    
    counts = {"baseline": [0,0,0], "strat_a": [0,0,0], "strat_b": [0,0,0]} # T, I, G
    
    with torch.no_grad():
        for example in tqdm(dataset):
            img0 = example["0.webp"].convert("RGB")
            img1 = example["1.webp"].convert("RGB")
            cap0 = example["npy"][0]
            cap1 = example["npy"][1]
            
            # Extract standard representations
            t_inputs = processor(text=[cap0, cap1], return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = model.text_model(**t_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            
            i_inputs = processor(images=[img0, img1], return_tensors="pt").to(device)
            v_out = model.vision_model(**i_inputs)
            v_embeds = model.visual_projection(v_out.pooler_output)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)
            
            v0 = v_embeds[0]
            v1 = v_embeds[1]
            t0 = t_embeds[0]
            t1 = t_embeds[1]
            
            def evaluate(v_0, v_1, t_0, t_1):
                s00 = (v_0 @ t_0).item()
                s01 = (v_0 @ t_1).item()
                s10 = (v_1 @ t_0).item()
                s11 = (v_1 @ t_1).item()
                t_match = (s00 > s01) and (s11 > s10)
                i_match = (s00 > s10) and (s11 > s01)
                return [1 if t_match else 0, 1 if i_match else 0, 1 if (t_match and i_match) else 0]

            # Baseline
            base_scores = evaluate(v0, v1, t0, t1)
            counts["baseline"][0] += base_scores[0]
            counts["baseline"][1] += base_scores[1]
            counts["baseline"][2] += base_scores[2]
            
            # Strategy A: Mean Erasure
            c_mean = (t0 + t1) / 2.0
            c_mean = c_mean / c_mean.norm()
            
            v0_a = v0 - (v0 @ c_mean) * c_mean
            v1_a = v1 - (v1 @ c_mean) * c_mean
            v0_a = v0_a / v0_a.norm()
            v1_a = v1_a / v1_a.norm()
            
            a_scores = evaluate(v0_a, v1_a, t0, t1)
            counts["strat_a"][0] += a_scores[0]
            counts["strat_a"][1] += a_scores[1]
            counts["strat_a"][2] += a_scores[2]
            
            # Strategy B: Noun Erasure
            # Winoground captions have same words, so nouns are identical. 
            nouns = extract_nouns(cap0)
            if not nouns:
                noun_str = "objects"
            else:
                noun_str = " and ".join(list(set(nouns)))
            
            noun_cap = f"A photo of a {noun_str}"
            n_inputs = processor(text=[noun_cap], return_tensors="pt", padding=True, truncation=True).to(device)
            n_out = model.text_model(**n_inputs)
            n_embed = model.text_projection(n_out.pooler_output)[0]
            n_embed = n_embed / n_embed.norm()
            
            v0_b = v0 - (v0 @ n_embed) * n_embed
            v1_b = v1 - (v1 @ n_embed) * n_embed
            v0_b = v0_b / v0_b.norm()
            v1_b = v1_b / v1_b.norm()
            
            b_scores = evaluate(v0_b, v1_b, t0, t1)
            counts["strat_b"][0] += b_scores[0]
            counts["strat_b"][1] += b_scores[1]
            counts["strat_b"][2] += b_scores[2]
            
    n = len(dataset)
    results["baseline"] = {
        "text_score": counts["baseline"][0] / n,
        "image_score": counts["baseline"][1] / n,
        "group_score": counts["baseline"][2] / n
    }
    results["strategy_a_mean_erasure"] = {
        "text_score": counts["strat_a"][0] / n,
        "image_score": counts["strat_a"][1] / n,
        "group_score": counts["strat_a"][2] / n
    }
    results["strategy_b_noun_erasure"] = {
        "text_score": counts["strat_b"][0] / n,
        "image_score": counts["strat_b"][1] / n,
        "group_score": counts["strat_b"][2] / n
    }
    
    print("Erasure Results:", json.dumps(results, indent=2))
    with open("erasure_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
