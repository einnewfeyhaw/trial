"""Per-pair overlap analysis: text-side vs image-side mean-erasure on Winoground.

Two erasure variants:
  * Text-side (erase C_mean from images):
        I0' = I0 - (I0·Cm)Cm,   I1' = I1 - (I1·Cm)Cm
        score with (I0', I1', C0, C1)
  * Image-side (erase I_mean from captions):
        C0' = C0 - (C0·Im)Im,   C1' = C1 - (C1·Im)Im
        score with (I0, I1, C0', C1')

For each of 400 pairs we record whether each variant passes Group Score, then
compute the overlap statistics:
    both_pass         (recoverable from BOTH directions)
    text_only_pass    (only image-content is "fixable")
    image_only_pass   (only caption-content is "fixable")
    neither_pass      (residual — true representation gap)

Saves per-pair labels + summary to ../05_results/erasure_pass_overlap.json.
"""

import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

HF_TOKEN = os.environ.get("HF_TOKEN")
dataset = load_dataset("facebook/winoground", split="test", token=HF_TOKEN)


def group_pass(I0, I1, C0, C1):
    """Group Score = (Image Score) AND (Text Score) for a single pair."""
    s00 = float(torch.dot(I0, C0).item())
    s01 = float(torch.dot(I0, C1).item())
    s10 = float(torch.dot(I1, C0).item())
    s11 = float(torch.dot(I1, C1).item())
    image_ok = (s00 > s10) and (s11 > s01)   # each caption picks right image
    text_ok  = (s00 > s01) and (s11 > s10)   # each image picks right caption
    return image_ok and text_ok


per_pair = []
counts = {"baseline": 0, "text_side": 0, "image_side": 0,
          "both_pass": 0, "text_only": 0, "image_only": 0, "neither": 0}
by_cat = {"Object": dict(counts), "Relation": dict(counts), "Both": dict(counts)}

with torch.no_grad():
    for ex in tqdm(dataset):
        img0 = ex["image_0"].convert("RGB")
        img1 = ex["image_1"].convert("RGB")
        cap0 = ex["caption_0"]
        cap1 = ex["caption_1"]
        tag = ex["collapsed_tag"]
        ex_id = int(ex["id"])

        t_inputs = processor(text=[cap0, cap1], return_tensors="pt",
                             padding=True, truncation=True).to(device)
        t_out = model.text_model(**t_inputs)
        t_embeds = model.text_projection(t_out.pooler_output)
        t_embeds = F.normalize(t_embeds, dim=-1)
        C0, C1 = t_embeds[0], t_embeds[1]

        i_inputs = processor(images=[img0, img1], return_tensors="pt").to(device)
        v_out = model.vision_model(**i_inputs)
        v_embeds = model.visual_projection(v_out.pooler_output)
        v_embeds = F.normalize(v_embeds, dim=-1)
        I0, I1 = v_embeds[0], v_embeds[1]

        # Baseline
        base_g = group_pass(I0, I1, C0, C1)

        # Text-side erasure: remove C_mean from images
        Cm  = F.normalize((C0 + C1) / 2.0, dim=-1)
        I0t = F.normalize(I0 - torch.dot(I0, Cm) * Cm, dim=-1)
        I1t = F.normalize(I1 - torch.dot(I1, Cm) * Cm, dim=-1)
        text_g = group_pass(I0t, I1t, C0, C1)

        # Image-side erasure: remove I_mean from captions
        Im  = F.normalize((I0 + I1) / 2.0, dim=-1)
        C0i = F.normalize(C0 - torch.dot(C0, Im) * Im, dim=-1)
        C1i = F.normalize(C1 - torch.dot(C1, Im) * Im, dim=-1)
        image_g = group_pass(I0, I1, C0i, C1i)

        per_pair.append({
            "id": ex_id, "tag": tag,
            "caption_0": cap0, "caption_1": cap1,
            "baseline_pass":   base_g,
            "text_side_pass":  text_g,
            "image_side_pass": image_g,
        })

        # Update counters
        for d in (counts, by_cat[tag]):
            d["baseline"]   += int(base_g)
            d["text_side"]  += int(text_g)
            d["image_side"] += int(image_g)
            if text_g and image_g:        d["both_pass"]  += 1
            elif text_g and not image_g:  d["text_only"]  += 1
            elif image_g and not text_g:  d["image_only"] += 1
            else:                         d["neither"]    += 1


def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def report_block(d, total):
    return {
        "n":                total,
        "baseline":         {"n": d["baseline"],   "pct": pct(d["baseline"],   total)},
        "text_side":        {"n": d["text_side"],  "pct": pct(d["text_side"],  total)},
        "image_side":       {"n": d["image_side"], "pct": pct(d["image_side"], total)},
        "both_pass":        {"n": d["both_pass"],  "pct": pct(d["both_pass"],  total)},
        "text_only":        {"n": d["text_only"],  "pct": pct(d["text_only"],  total)},
        "image_only":       {"n": d["image_only"], "pct": pct(d["image_only"], total)},
        "neither":          {"n": d["neither"],    "pct": pct(d["neither"],    total)},
    }


N = len(per_pair)
report = {
    "total_pairs": N,
    "overall": report_block(counts, N),
    "by_category": {
        cat: report_block(by_cat[cat], by_cat[cat]["baseline"]
                          + (by_cat[cat]["text_side"] - by_cat[cat]["both_pass"]
                             - by_cat[cat]["text_only"])  # placeholder; corrected below
                          )
        for cat in ["Object", "Relation", "Both"]
    },
}

# Replace by_category with correct totals (sum of categories' baseline+text_only+image_only+neither+both_pass-baseline...
# simpler: count category sizes directly from per_pair)
cat_totals = {"Object": 0, "Relation": 0, "Both": 0}
for p in per_pair:
    cat_totals[p["tag"]] += 1
report["by_category"] = {
    cat: report_block(by_cat[cat], cat_totals[cat])
    for cat in ["Object", "Relation", "Both"]
}

# Pretty-print
print("\n=== Erasure pass-overlap analysis (CLIP-B/32, Winoground) ===\n")

def print_block(name, b):
    print(f"[{name}]   n = {b['n']}")
    print(f"  Baseline group:               {b['baseline']['n']:3d}  ({b['baseline']['pct']:.1f}%)")
    print(f"  Text-side erasure passes:     {b['text_side']['n']:3d}  ({b['text_side']['pct']:.1f}%)")
    print(f"  Image-side erasure passes:    {b['image_side']['n']:3d}  ({b['image_side']['pct']:.1f}%)")
    print(f"  --- overlap of the two pass sets ---")
    print(f"  Both sides pass:              {b['both_pass']['n']:3d}  ({b['both_pass']['pct']:.1f}%)")
    print(f"  Only text-side passes:        {b['text_only']['n']:3d}  ({b['text_only']['pct']:.1f}%)")
    print(f"  Only image-side passes:       {b['image_only']['n']:3d}  ({b['image_only']['pct']:.1f}%)")
    print(f"  Neither passes (residual):    {b['neither']['n']:3d}  ({b['neither']['pct']:.1f}%)")
    print()

print_block("OVERALL", report["overall"])
for cat in ["Object", "Relation", "Both"]:
    print_block(cat, report["by_category"][cat])

# Save full per-pair data
out = {"summary": report, "per_pair": per_pair}
out_path = os.path.join(os.path.dirname(__file__), "..", "05_results",
                        "erasure_pass_overlap.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved per-pair data and summary -> {out_path}")
