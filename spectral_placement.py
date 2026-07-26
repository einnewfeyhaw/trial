import torch
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import spearmanr

def evaluate_spectral_placement():
    print("Loading model and dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    
    dataset = load_dataset("facebook/winoground", split="test", use_auth_token=False)
    
    # Get SVD of projection matrix
    W = model.visual_projection.weight.detach().cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    U_t = torch.tensor(U.T).float().to(device)
    
    top_20_cutoff = int(0.2 * len(S))
    bottom_20_cutoff = int(0.8 * len(S))
    
    cmean_top_norms = []
    cmean_bot_norms = []
    delta_top_norms = []
    delta_bot_norms = []
    
    print("Computing spectral norms...")
    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            c0_text = ex["caption_0"]
            c1_text = ex["caption_1"]
            
            inputs = processor(text=[c0_text, c1_text], return_tensors="pt", padding=True, truncation=True).to(device)
            t_embeds = model.get_text_features(**inputs)
            t_embeds = t_embeds / t_embeds.norm(dim=-1, keepdim=True)
            
            c0 = t_embeds[0]
            c1 = t_embeds[1]
            
            cmean = 0.5 * (c0 + c1)
            delta = 0.5 * (c0 - c1)
            
            # Project to SVD basis
            cmean_svd = (cmean @ U_t.T).abs()
            delta_svd = (delta @ U_t.T).abs()
            
            # Norms in top 20% vs bot 20%
            cmean_top_norms.append(cmean_svd[:top_20_cutoff].sum().item() / cmean_svd.sum().item())
            cmean_bot_norms.append(cmean_svd[bottom_20_cutoff:].sum().item() / cmean_svd.sum().item())
            
            delta_top_norms.append(delta_svd[:top_20_cutoff].sum().item() / delta_svd.sum().item())
            delta_bot_norms.append(delta_svd[bottom_20_cutoff:].sum().item() / delta_svd.sum().item())

    print(f"Cmean norm concentrated in top 20% SVs: {np.mean(cmean_top_norms):.1%}")
    print(f"Cmean norm concentrated in bot 20% SVs: {np.mean(cmean_bot_norms):.1%}")
    print(f"Delta norm concentrated in top 20% SVs: {np.mean(delta_top_norms):.1%}")
    print(f"Delta norm concentrated in bot 20% SVs: {np.mean(delta_bot_norms):.1%}")

if __name__ == "__main__":
    evaluate_spectral_placement()
