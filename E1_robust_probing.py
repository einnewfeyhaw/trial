import torch
import numpy as np
import json
import os
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'

MODELS_TO_TEST = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384"
]

def analyze_model_robust(model_name, dataset, device, limit=1500, seeds=[42, 43, 44, 45, 46]):
    print(f"\n--- Robust Analyzing {model_name} ---")
    
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None
        
    model.eval()
    
    if hasattr(model, 'visual_projection') and model.visual_projection is not None:
        W = model.visual_projection.weight.detach().cpu().numpy()
    elif "siglip" in model_name.lower():
        W = model.vision_model.head.attention.out_proj.weight.detach().cpu().numpy()
    else:
        print(f"Could not find visual projection matrix for {model_name}. Skipping SVD.")
        return None
        
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    U_t = torch.tensor(U.T).float().to(device)
    
    post_samples_svd = []
    labels = []
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset.select(range(min(limit, len(dataset)))))):
            img = example["0.webp"].convert("RGB")
            true_cap = example["npy"][0]
            false_cap = example["npy"][1]
            
            if "siglip" in model_name.lower():
                inputs = processor(text=[true_cap, false_cap], images=img, padding="max_length", return_tensors="pt", truncation=True).to(device)
                outputs = model(**inputs)
                t_embeds = outputs.text_embeds
                v_embeds = outputs.image_embeds
            else:
                inputs = processor(text=[true_cap, false_cap], images=img, padding=True, return_tensors="pt", truncation=True).to(device)
                outputs = model(**inputs)
                t_embeds = outputs.text_embeds
                v_embeds = outputs.image_embeds
                
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            v_embeds = v_embeds / v_embeds.norm(dim=-1, keepdim=True)
                
            z_svd = v_embeds[0] @ U_t.T
            t_true_svd = t_embeds[0] @ U_t.T
            t_false_svd = t_embeds[1] @ U_t.T
            
            post_samples_svd.append((z_svd * t_true_svd).cpu().numpy())
            labels.append(1)
            
            post_samples_svd.append((z_svd * t_false_svd).cpu().numpy())
            labels.append(0)
            
    X = np.array(post_samples_svd)
    y = np.array(labels)
    
    # PROPER 80/20 SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlp_rhos = []
    mlp_accs = []
    
    for seed in seeds:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=seed)
        clf.fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        mlp_accs.append(acc)
        
        w1 = clf.coefs_[0]
        importance = np.linalg.norm(w1, axis=1)
        
        corr, p_val = spearmanr(S, importance)
        mlp_rhos.append(corr)

    # Linear baseline
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    lr_importance = np.abs(lr.coef_[0])
    lr_corr, lr_pval = spearmanr(S, lr_importance)

    res = {
        "model": model_name,
        "mlp_test_acc_mean": float(np.mean(mlp_accs)),
        "mlp_test_acc_std": float(np.std(mlp_accs)),
        "mlp_spearman_rho_mean": float(np.mean(mlp_rhos)),
        "mlp_spearman_rho_std": float(np.std(mlp_rhos)),
        "lr_test_acc": float(lr_acc),
        "lr_spearman_rho": float(lr_corr)
    }
    print("Result:", res)
    return res

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We use a smaller dataset subset for speed during debugging but use 1500 for good measure.
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")
    
    results = {}
    # Test just base CLIP first
    res = analyze_model_robust("openai/clip-vit-base-patch32", dataset, device, limit=1500)
    results["openai/clip-vit-base-patch32"] = res
    
    # Test SigLIP large to verify the reversal
    res2 = analyze_model_robust("google/siglip-so400m-patch14-384", dataset, device, limit=1500)
    results["google/siglip-so400m-patch14-384"] = res2
    
    with open("E1_robust_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
