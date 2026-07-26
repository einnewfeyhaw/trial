import torch
import numpy as np
import json
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'

def main():
    print("Training MLP on ARO (Visual Genome Relations) Post-Projection for importance analysis...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # We use a 1000-pair subset to match the statistical power of the SugarCrepe run
    dataset = load_dataset("haideraltahan/wds_flickr30k_order", split="test")
    
    post_samples = []
    labels = []
    
    limit = 1000
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset.select(range(min(limit, len(dataset)))))):
            img = example["0.webp"].convert("RGB")
            true_cap = example["npy"][0]
            false_cap = example["npy"][1]
            
            text_inputs = processor(text=[true_cap, false_cap], return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = model.text_model(**text_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            t_true = t_embeds[0].cpu().numpy()
            t_false = t_embeds[1].cpu().numpy()
            
            img_inputs = processor(images=[img], return_tensors="pt").to(device)
            v_out = model.vision_model(**img_inputs)
            post_proj = model.visual_projection(v_out.pooler_output)[0]
            post_proj = post_proj / post_proj.norm(dim=-1, keepdim=True)
            post_proj = post_proj.cpu().numpy()
            
            post_samples.append(post_proj * t_true)
            labels.append(1)
            
            post_samples.append(post_proj * t_false)
            labels.append(0)
            
    X = np.array(post_samples)
    y = np.array(labels)
    
    # SVD of Visual Projection matrix
    W = model.visual_projection.weight.detach().cpu().numpy() # shape (512, 768)
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    
    print("Projecting ARO representations onto Singular Basis...")
    U_t = torch.tensor(U.T).float().to(device) # (512, 512)
    
    post_samples_svd = []
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset.select(range(min(limit, len(dataset)))))):
            img = example["0.webp"].convert("RGB")
            true_cap = example["npy"][0]
            false_cap = example["npy"][1]
            
            text_inputs = processor(text=[true_cap, false_cap], return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = model.text_model(**text_inputs)
            t_embeds = model.text_projection(t_out.pooler_output)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            
            img_inputs = processor(images=[img], return_tensors="pt").to(device)
            v_out = model.vision_model(**img_inputs)
            post_proj = model.visual_projection(v_out.pooler_output)[0]
            post_proj = post_proj / post_proj.norm(dim=-1, keepdim=True)
            
            z_svd = post_proj @ U_t.T # (512,)
            t_true_svd = t_embeds[0] @ U_t.T
            t_false_svd = t_embeds[1] @ U_t.T
            
            post_samples_svd.append((z_svd * t_true_svd).cpu().numpy())
            post_samples_svd.append((z_svd * t_false_svd).cpu().numpy())
            
    X_svd = np.array(post_samples_svd)
    
    clf_svd = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    clf_svd.fit(X_svd, y)
    print("SVD-basis MLP trained. Acc:", clf_svd.score(X_svd, y))
    
    w1_svd = clf_svd.coefs_[0]
    svd_importance = np.linalg.norm(w1_svd, axis=1) # (512,)
    
    corr, p_val = spearmanr(S, svd_importance)
    
    top_k = 50
    top_indices = np.argsort(svd_importance)[-top_k:][::-1]
    bottom_indices = np.argsort(svd_importance)[:top_k]
    
    results = {
        "dataset": "ARO/Flickr30k Order",
        "spearman_correlation": float(corr),
        "p_value": float(p_val),
        "mean_singular_value_of_top_50_important": float(np.mean(S[top_indices])),
        "mean_singular_value_of_bottom_50_important": float(np.mean(S[bottom_indices])),
        "global_mean_singular_value": float(np.mean(S))
    }
    
    print("ARO SVD Results:", results)
    with open("aro_svd_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
