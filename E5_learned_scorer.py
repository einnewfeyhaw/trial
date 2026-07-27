"""
E5: Different matching function, same frozen representations.

Addresses Reviewer v8Kz directly: "controlled comparisons using ... different
matching functions applied to the same representations." Also the single most
direct test of the paper's central thesis -- more direct than E2/E6, since it
does not touch magnitude, SVD, or erasure at all.

Question: on the EXACT frozen embeddings that score ~9% Group Score under
cosine similarity, does a learned matching function recover more? If yes,
"information survives the projection but cosine can't reach it" is a shown
fact, not an inference from indirect measures.

Design:
  - Encoder is frozen throughout. No fine-tuning, ever.
  - Replace cos(v,t) = v^T t with a learned bilinear form v^T A t.
    A = I reproduces the paper's baseline exactly (sanity-checkable).
  - A is regularized toward I (weight decay pulls it back to cosine absent
    signal) so the model cannot become an unconstrained classifier decoupled
    from the geometry -- it has to be a *correction* to cosine.
  - TRAIN on ColorSwap match/mismatch pairs, EVALUATE zero-shot on
    Winoground. Cross-benchmark, not k-fold-within-Winoground: a stronger
    generalization claim, and avoids any leakage argument.
  - No candidate-caption / foil access at eval time -- A is fixed after
    training and scores (image, single caption) pairs exactly like cosine
    does. This is NOT mean-erasure; it is a deployable scoring function.

Report: baseline cosine (A=I, should match paper exactly) vs learned-A Winoground
Text/Image/Group scores, plus train-set accuracy for sanity.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

MODELS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
]


def encode_pairs(model, processor, device, items, is_siglip):
    """items: list of (img0, img1, cap0, cap1). Returns V0,V1,T0,T1 tensors."""
    V0, V1, T0, T1 = [], [], [], []
    with torch.no_grad():
        for img0, img1, c0, c1 in tqdm(items, desc="  encoding"):
            kwargs = {"padding": "max_length"} if is_siglip else {"padding": True}
            inputs = processor(
                text=[c0, c1], images=[img0, img1],
                return_tensors="pt", truncation=True, **kwargs,
            ).to(device)
            out = model(**inputs)
            t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            v = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            V0.append(v[0].cpu()); V1.append(v[1].cpu())
            T0.append(t[0].cpu()); T1.append(t[1].cpu())
    return torch.stack(V0), torch.stack(V1), torch.stack(T0), torch.stack(T1)


def load_colorswap_items(path, limit, split="test"):
    """Load local ColorSwap via E3_colorswap_local's loader (same data_dir
    layout: <path>/<split>.json + <path>/images/), so this uses the exact
    same data the E3 ColorSwap run used -- no separate/inconsistent loading
    logic. Returns (img0,img1,cap0,cap1) tuples."""
    from E3_colorswap_local import load_colorswap
    rows = load_colorswap(path, split)
    items = []
    for r in rows[:limit]:
        items.append((r["image_1"], r["image_2"], str(r["caption_1"]), str(r["caption_2"])))
    return items


def load_winoground_items(limit):
    ds = load_dataset("haideraltahan/wds_winoground", split="test")
    items = []
    for ex in ds.select(range(min(limit, len(ds)))):
        items.append((
            ex["0.webp"].convert("RGB"), ex["1.webp"].convert("RGB"),
            str(ex["npy"][0]), str(ex["npy"][1]),
        ))
    return items


class Bilinear(nn.Module):
    """score(v,t) = v^T A t, A initialized to identity (= cosine)."""

    def __init__(self, dim):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(dim, dim))

    def A(self):
        return torch.eye(self.delta.shape[0], device=self.delta.device) + self.delta

    def forward(self, v, t):
        return (v @ self.A() * t).sum(-1)


def score_all(scorer, v0, v1, t0, t1):
    return (
        scorer(v0, t0).item(), scorer(v0, t1).item(),
        scorer(v1, t0).item(), scorer(v1, t1).item(),
    )


def group_scores(scorer, V0, V1, T0, T1):
    text_c = image_c = group_c = 0
    n = len(V0)
    for i in range(n):
        s00, s01, s10, s11 = score_all(scorer, V0[i], V1[i], T0[i], T1[i])
        t_ok = (s00 > s01) and (s11 > s10)
        i_ok = (s00 > s10) and (s11 > s01)
        text_c += t_ok; image_c += i_ok; group_c += (t_ok and i_ok)
    return text_c / n, image_c / n, group_c / n


def train_bilinear(V0, V1, T0, T1, dim, device, epochs=200, lr=1e-2, weight_decay=1e-2):
    """Train A on ColorSwap match/mismatch pairs via pairwise logistic loss,
    regularized toward the identity (= cosine) by weight decay on the delta."""
    scorer = Bilinear(dim).to(device)
    opt = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    V0, V1, T0, T1 = V0.to(device), V1.to(device), T0.to(device), T1.to(device)

    for ep in range(epochs):
        opt.zero_grad()
        # matched: (v0,t0) and (v1,t1); mismatched: (v0,t1) and (v1,t0)
        s_match = scorer(V0, T0) + scorer(V1, T1)
        s_mismatch = scorer(V0, T1) + scorer(V1, T0)
        loss = torch.nn.functional.softplus(s_mismatch - s_match).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_group = group_scores(scorer, V0, V1, T0, T1)
    return scorer, float(loss.item()), train_group


def analyze(model_name, colorswap_items, winoground_items, device, args):
    print(f"\n=== {model_name} ===")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    is_siglip = "siglip" in model_name.lower()

    print("  encoding ColorSwap (train)...")
    V0tr, V1tr, T0tr, T1tr = encode_pairs(model, processor, device, colorswap_items, is_siglip)
    print("  encoding Winoground (eval, held out)...")
    V0te, V1te, T0te, T1te = encode_pairs(model, processor, device, winoground_items, is_siglip)

    dim = V0tr.shape[1]

    # cosine baseline == A=I; must reproduce the paper's number
    cos_scorer = Bilinear(dim).to(device)  # delta=0 => A=I
    base_scores = group_scores(cos_scorer, V0te.to(device), V1te.to(device),
                                T0te.to(device), T1te.to(device))

    scorer, final_loss, train_scores = train_bilinear(
        V0tr, V1tr, T0tr, T1tr, dim, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
    )

    with torch.no_grad():
        eval_scores = group_scores(scorer, V0te.to(device), V1te.to(device),
                                    T0te.to(device), T1te.to(device))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    res = {
        "model": model_name,
        "n_train_colorswap": len(colorswap_items),
        "n_eval_winoground": len(winoground_items),
        "cosine_baseline_winoground": {
            "text": base_scores[0], "image": base_scores[1], "group": base_scores[2],
        },
        "learned_bilinear_train_colorswap": {
            "text": train_scores[0], "image": train_scores[1], "group": train_scores[2],
        },
        "learned_bilinear_eval_winoground_ZEROSHOT": {
            "text": eval_scores[0], "image": eval_scores[1], "group": eval_scores[2],
        },
        "final_train_loss": final_loss,
        "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground",
    }
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colorswap-path", required=True,
                    help="ColorSwap data_dir (same --data-dir passed to "
                         "E3_colorswap_local.py: contains <split>.json + images/)")
    ap.add_argument("--colorswap-split", default="train",
                    help="use ColorSwap's train split (700 pairs) to fit A, "
                         "keeping the test split reserved for the separate "
                         "G2 mean-erasure result -- no reuse across findings")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--max-train", type=int, default=700)
    ap.add_argument("--max-eval", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--out", default="E5_learned_scorer.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    colorswap_items = load_colorswap_items(
        args.colorswap_path, args.max_train, split=args.colorswap_split)
    winoground_items = load_winoground_items(args.max_eval)
    print(f"ColorSwap train pairs: {len(colorswap_items)}")
    print(f"Winoground eval pairs: {len(winoground_items)}")

    results = {}
    for name in args.models:
        try:
            r = analyze(name, colorswap_items, winoground_items, device, args)
            results[name] = r
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"FAILED {name}: {e}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
