import torch
import numpy as np
import json
import os
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'

MODELS_TO_TEST = [
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "openai/clip-vit-large-patch14",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384"
]

def analyze_model(model_name, dataset, device, limit=1000, batch_size=32):
    print(f"\n--- Analyzing {model_name} on ARO ---")

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

    print(f"Projection Matrix shape: {W.shape}")
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    U_t = torch.tensor(U.T).float().to(device)

    post_samples_svd = []
    labels = []

    print("Extracting SVD-basis features...")
    with torch.no_grad():
        # ARO dataset structure
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

            if v_embeds.shape[-1] != U_t.shape[0]:
                print(f"Dim mismatch: v_embeds {v_embeds.shape}, U_t {U_t.shape}. Skipping SVD.")
                return None

            z_svd = v_embeds[0] @ U_t.T
            t_true_svd = t_embeds[0] @ U_t.T
            t_false_svd = t_embeds[1] @ U_t.T

            # Element-wise product for XOR logic interaction
            post_samples_svd.append((z_svd * t_true_svd).cpu().numpy())
            labels.append(1)

            post_samples_svd.append((z_svd * t_false_svd).cpu().numpy())
            labels.append(0)

    X = np.array(post_samples_svd)
    y = np.array(labels)

    print("Training MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    clf.fit(X, y)

    w1 = clf.coefs_[0]
    importance = np.linalg.norm(w1, axis=1)

    corr, p_val = spearmanr(S, importance)

    res = {
        "model": model_name,
        "mlp_accuracy": float(clf.score(X, y)),
        "spearman_correlation": float(corr),
        "p_value": float(p_val)
    }
    print("Result:", res)
    return res

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Using ARO Flickr30k Order dataset
    dataset = load_dataset("haideraltahan/wds_flickr30k_order", split="test")

    results = {}
    for model_name in MODELS_TO_TEST:
        try:
            res = analyze_model(model_name, dataset, device, limit=1000)
            if res:
                results[model_name] = res
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {model_name}, attempting with smaller limit...")
            try:
                res = analyze_model(model_name, dataset, device, limit=500)
                if res:
                    results[model_name] = res
            except Exception as e:
                print(f"Failed even with reduced limit for {model_name}: {e}")
        except Exception as e:
            print(f"Failed entirely for {model_name}: {e}")

    with open("multi_model_aro_correlations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to multi_model_aro_correlations.json")
    print(f"Successfully completed {len(results)}/{len(MODELS_TO_TEST)} models")

if __name__ == "__main__":
    import sys
    print(f"Python executable: {sys.executable}", flush=True)
    main()
