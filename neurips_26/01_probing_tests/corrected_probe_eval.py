import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os

# Limit thread usage for sklearn to avoid locking up
os.environ['OMP_NUM_THREADS'] = '4'

def main():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading dataset...")
    # Using the wds_sugarcrepe subset which has true/false captions
    # Note: we use 4000 pairs to make sure we have enough data to train an MLP
    dataset = load_dataset("haideraltahan/wds_sugarcrepe", split="test")
    
    pre_samples = []
    post_samples = []
    labels = []
    
    print("Extracting features...")
    limit = 3000  # Extract up to 3000 pairs (6000 total samples)
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset.select(range(min(limit, len(dataset)))))):
            img = example["0.webp"].convert("RGB")
            captions = example["npy"]
            # the original wds structure typically has true caption at index 0, false at index 1.
            # let's verify if that's standard for wds_sugarcrepe:
            true_cap = captions[0]
            false_cap = captions[1]
            
            # Process texts
            text_inputs = processor(text=[true_cap, false_cap], return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = model.text_model(**text_inputs)
            text_embeds = model.text_projection(text_outputs.pooler_output)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            t_true = text_embeds[0].cpu().numpy()
            t_false = text_embeds[1].cpu().numpy()
            
            # Use concatenation instead of element-wise product because dims are different
            
            # Process image
            img_inputs = processor(images=[img], return_tensors="pt").to(device)
            vision_outputs = model.vision_model(**img_inputs)
            
            pre_proj = vision_outputs.pooler_output[0].cpu().numpy() # dim 768
            
            post_proj = model.visual_projection(vision_outputs.pooler_output)[0]
            post_proj = post_proj / post_proj.norm(dim=-1, keepdim=True)
            post_proj = post_proj.cpu().numpy() # dim 512
            
            # Match (True Caption) -> Label 1
            pre_samples.append(np.concatenate([pre_proj, t_true]))
            post_samples.append(np.concatenate([post_proj, t_true]))
            labels.append(1)
            
            # Mismatch (False Caption) -> Label 0
            pre_samples.append(np.concatenate([pre_proj, t_false]))
            post_samples.append(np.concatenate([post_proj, t_false]))
            labels.append(0)
            
    X_pre = np.array(pre_samples)
    X_post = np.array(post_samples)
    y = np.array(labels)
    
    # Train/Test Split (80/20)
    n_pairs = len(X_pre) // 2
    indices = np.arange(n_pairs)
    train_idx_pairs, test_idx_pairs = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_idx = np.concatenate([train_idx_pairs*2, train_idx_pairs*2+1])
    test_idx = np.concatenate([test_idx_pairs*2, test_idx_pairs*2+1])
    
    X_pre_train, X_pre_test = X_pre[train_idx], X_pre[test_idx]
    X_post_train, X_post_test = X_post[train_idx], X_post[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("Training pre-projection MLP...")
    # Increase capacity to learn the interaction from concatenated inputs
    clf_pre = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, random_state=42, early_stopping=True)
    clf_pre.fit(X_pre_train, y_train)
    pre_acc = clf_pre.score(X_pre_test, y_test)
    
    print("Training post-projection MLP...")
    clf_post = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, random_state=42, early_stopping=True)
    clf_post.fit(X_post_train, y_train)
    post_acc = clf_post.score(X_post_test, y_test)
    
    results = {
        "pre_projection_accuracy": float(pre_acc),
        "post_projection_accuracy": float(post_acc),
        "train_samples": len(y_train),
        "test_samples": len(y_test)
    }
    
    print("Results:", results)
    with open("corrected_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
