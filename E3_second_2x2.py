"""
E3: Mean-erasure on a SECOND 2x2 benchmark.

Addresses the AC's binding demand ("establish performance gains on at least one
other standard compositionality benchmark") and Reviewer iv2d Q2.

Any genuine 2x2 benchmark works: two images x two captions per group, where both
captions share vocabulary and differ in arrangement. Candidates, in rough order
of how easily they load:

    stanfordnlp/colorswap      ColorSwap (Burapacheep et al. 2024), ~2k pairs,
                               purpose-built Winoground format
    BAAI/EqBen                 EqBen (Wang et al., ICCV 2023)
    haideraltahan/wds_winoground   Winoground -- use as the SANITY CHECK; it
                               must reproduce 9.0% -> 31.0% on CLIP-B/32

DATASET IDS ARE NOT GUARANTEED. Run with --inspect first to confirm the id
resolves and to see the field names, then run the evaluation.

Protocol mirrors 04_concept_erasure/concept_erasure_eval.py exactly, so numbers
are directly comparable to Table 2 / Table 4 of the paper. Crucially it also
runs the paper's CONTROLS on the new benchmark -- reporting only the headline
gain invites "you showed a gain, not specificity."

    baseline            no intervention
    mean_erasure        erase C_mean = (C0+C1)/2   [the method]
    c0_only             erase C0 alone             [control]
    c1_only             erase C1 alone             [control]
    random_pair         erase C_mean of an unrelated pair  [control]
    image_side          erase I_mean from text embeddings  [symmetry check]

Bootstrap CIs and a one-sided p-value are computed over pairs.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
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

# (image_0, image_1) and (caption_0, caption_1) field-name patterns seen in the wild
IMAGE_PATTERNS = [
    ("image_0", "image_1"),
    ("image_1", "image_2"),
    ("0.webp", "1.webp"),
    ("img0", "img1"),
    ("image1", "image2"),
]
CAPTION_PATTERNS = [
    ("caption_0", "caption_1"),
    ("caption_1", "caption_2"),
    ("text_0", "text_1"),
    ("caption1", "caption2"),
]


def inspect(dataset):
    print("\n--- dataset inspection ---")
    print("columns:", dataset.column_names)
    print("n examples:", len(dataset))
    ex = dataset[0]
    for k, v in ex.items():
        if hasattr(v, "convert"):
            print(f"  {k}: <PIL image {v.size}>")
        else:
            s = str(v)
            print(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")
    print("--- end inspection ---\n")


def resolve_fields(example, img_fields, cap_fields):
    """Return (get_images, get_captions) callables for this dataset's schema."""
    keys = set(example.keys())

    if img_fields:
        a, b = img_fields
        get_img = lambda ex: (ex[a], ex[b])
    else:
        get_img = None
        for a, b in IMAGE_PATTERNS:
            if a in keys and b in keys:
                get_img = lambda ex, a=a, b=b: (ex[a], ex[b])
                print(f"  images  <- ('{a}', '{b}')")
                break

    if cap_fields:
        a, b = cap_fields
        get_cap = lambda ex: (ex[a], ex[b])
    else:
        get_cap = None
        for a, b in CAPTION_PATTERNS:
            if a in keys and b in keys:
                get_cap = lambda ex, a=a, b=b: (str(ex[a]), str(ex[b]))
                print(f"  captions <- ('{a}', '{b}')")
                break
        # webdataset style: a single 'npy' holding a list of captions
        if get_cap is None and "npy" in keys:
            get_cap = lambda ex: (str(ex["npy"][0]), str(ex["npy"][1]))
            print("  captions <- npy[0], npy[1]")

    if get_img is None or get_cap is None:
        raise RuntimeError(
            "Could not resolve fields. Run --inspect and pass "
            "--img-fields A B --cap-fields A B explicitly."
        )
    return get_img, get_cap


def to_pil(x):
    return x.convert("RGB") if hasattr(x, "convert") else x


def winoground_scores(v0, v1, t0, t1):
    """Paper's scoring: identical to concept_erasure_eval.py evaluate()."""
    s00 = float(v0 @ t0)
    s01 = float(v0 @ t1)
    s10 = float(v1 @ t0)
    s11 = float(v1 @ t1)
    text = (s00 > s01) and (s11 > s10)
    image = (s00 > s10) and (s11 > s01)
    return int(text), int(image), int(text and image)


def erase(x, direction):
    """Project x orthogonally away from direction, then renormalize."""
    d = direction / direction.norm()
    out = x - (x @ d) * d
    n = out.norm()
    return out / n if float(n) > 1e-8 else out


def encode_all(model_name, dataset, device, get_img, get_cap, max_pairs):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    is_siglip = "siglip" in model_name.lower()

    V0, V1, T0, T1 = [], [], [], []
    with torch.no_grad():
        for i, ex in enumerate(tqdm(dataset, desc="  encoding")):
            if i >= max_pairs:
                break
            try:
                i0, i1 = (to_pil(z) for z in get_img(ex))
                c0, c1 = get_cap(ex)
            except Exception:
                continue

            kwargs = {"padding": "max_length"} if is_siglip else {"padding": True}
            inputs = processor(
                text=[c0, c1], images=[i0, i1],
                return_tensors="pt", truncation=True, **kwargs,
            ).to(device)
            out = model(**inputs)

            t = out.text_embeds
            v = out.image_embeds
            t = t / t.norm(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True)

            V0.append(v[0].cpu())
            V1.append(v[1].cpu())
            T0.append(t[0].cpu())
            T1.append(t[1].cpu())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(V0), torch.stack(V1), torch.stack(T0), torch.stack(T1)


def run_conditions(V0, V1, T0, T1, seed=0):
    """Return per-pair (text, image, group) flags for every condition."""
    n = len(V0)
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n)

    conds = {k: [] for k in
             ["baseline", "mean_erasure", "c0_only", "c1_only",
              "random_pair", "image_side"]}

    for i in range(n):
        v0, v1, t0, t1 = V0[i], V1[i], T0[i], T1[i]
        c_mean = 0.5 * (t0 + t1)

        conds["baseline"].append(winoground_scores(v0, v1, t0, t1))
        conds["mean_erasure"].append(
            winoground_scores(erase(v0, c_mean), erase(v1, c_mean), t0, t1))
        conds["c0_only"].append(
            winoground_scores(erase(v0, t0), erase(v1, t0), t0, t1))
        conds["c1_only"].append(
            winoground_scores(erase(v0, t1), erase(v1, t1), t0, t1))

        j = shuffled[i]
        rand_mean = 0.5 * (T0[j] + T1[j])
        conds["random_pair"].append(
            winoground_scores(erase(v0, rand_mean), erase(v1, rand_mean), t0, t1))

        i_mean = 0.5 * (v0 + v1)
        conds["image_side"].append(
            winoground_scores(v0, v1, erase(t0, i_mean), erase(t1, i_mean)))

    return {k: np.array(v) for k, v in conds.items()}


def bootstrap_delta(base, treat, col=2, n_boot=10000, seed=0):
    """Bootstrap CI and one-sided p for treat-minus-base on a score column."""
    rng = np.random.RandomState(seed)
    n = len(base)
    b, t = base[:, col], treat[:, col]
    obs = t.mean() - b.mean()
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[k] = t[idx].mean() - b[idx].mean()
    return {
        "delta": float(obs),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
        "p_one_sided": float((deltas <= 0).mean()),
    }


def analyze(model_name, dataset, device, get_img, get_cap, args):
    print(f"\n=== {model_name} ===")
    V0, V1, T0, T1 = encode_all(
        model_name, dataset, device, get_img, get_cap, args.max_pairs)
    n = len(V0)
    print(f"  encoded {n} pairs")

    conds = run_conditions(V0, V1, T0, T1)
    base = conds["baseline"]

    res = {"model": model_name, "n_pairs": int(n), "conditions": {}}
    for name, arr in conds.items():
        entry = {
            "text_score": float(arr[:, 0].mean()),
            "image_score": float(arr[:, 1].mean()),
            "group_score": float(arr[:, 2].mean()),
        }
        if name != "baseline":
            entry["group_vs_baseline"] = bootstrap_delta(
                base, arr, col=2, n_boot=args.n_boot)
            g0 = base[:, 2].mean()
            entry["relative_gain"] = (
                float(arr[:, 2].mean() / g0) if g0 > 0 else None)
        res["conditions"][name] = entry

    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="stanfordnlp/colorswap")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--img-fields", nargs=2, default=None)
    ap.add_argument("--cap-fields", nargs=2, default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"dataset: {args.dataset} (config={args.config}, split={args.split})")

    kw = {"split": args.split}
    if args.config:
        kw["name"] = args.config
    dataset = load_dataset(args.dataset, **kw)

    if args.inspect:
        inspect(dataset)
        return

    get_img, get_cap = resolve_fields(dataset[0], args.img_fields, args.cap_fields)

    out_path = args.out or (
        "E3_" + args.dataset.replace("/", "_") + ".json")

    results = {"dataset": args.dataset, "split": args.split, "models": {}}
    for name in args.models:
        try:
            r = analyze(name, dataset, device, get_img, get_cap, args)
            results["models"][name] = r
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"FAILED {name}: {e}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
