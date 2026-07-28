"""
E8: LLaVA-1.5-7B on Winoground -- MLLM Bridge (fresh GPU run).

Addresses Reviewer iv2d Q3:
  "Since the diagnostic argues compositional information survives in CLIP
   features, have the authors tested whether these features, when consumed by
   a downstream LLM, retain measurable compositional signal?"

LLaVA-1.5-7B uses a *frozen* CLIP ViT-L/14-336 visual encoder. Its patch
features are projected into Vicuna-7B via a 2-layer MLP and consumed by the LLM
autoregressively -- cosine similarity is NEVER computed. If LLaVA extracts
compositional signal from the SAME frozen CLIP tower that scores ~7% Group on
Winoground under cosine ITC, that supports the paper's thesis that the
information survives the encoder and cosine is (part of) the bottleneck.

PROTOCOL (as requested by the rebuttal task, with a bias control added):
  For each Winoground pair (I0, I1, C0, C1) we ask LLaVA, per image, which of two
  captions matches better, and parse the A/B letter.
    - I0: correct caption is C0.
    - I1: correct caption is C1.
  This yields a Text/Pair-Score analog: did the LLM assign the correct caption
  to BOTH images of the pair?

  BIAS CONTROL: LLaVA has a known option-position bias (tendency to prefer "A").
  We therefore query each image in BOTH caption orderings:
    order1: A=C0, B=C1     order2: A=C1, B=C0
  - naive score  = correct in order1 only (single-shot, matches the raw ask).
  - robust score = correct in BOTH orderings (order-consistent; bias-immune).
  We also log the global A-vs-B choice rate to quantify position bias directly.

FAIR BASELINES (CLIP ViT-L/14 cosine ITC, from paper Table 4 / E4):
  Text Score = 28.25%, Image Score = 10.5%, Group Score = 7.25-7.5%.
  The LLaVA pair-score protocol is a TEXT/PAIR-score analog (each image selects
  its caption); it is NOT a true Winoground Image Score, so the honest
  comparisons are: LLaVA pair-score vs CLIP Text Score AND vs CLIP Group Score.

Outputs: E8_results.json
"""

import json
import os
import re
import random
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")

MODEL_ID  = "llava-hf/llava-1.5-7b-hf"
WG_DATA   = "haideraltahan/wds_winoground"
OUT_FILE  = "E8_results.json"
MAX_PAIRS = int(os.environ.get("E8_MAX_PAIRS", "150"))  # random subset for tractable T4 runtime
SEED      = 42
MAX_NEW_TOKENS = 5

# CLIP ViT-L/14 cosine ITC baseline (paper Table 4 / E4, in-session verified).
CLIP_L14_BASELINE = {
    "text_score":  0.2825,
    "image_score": 0.105,
    "group_score": 0.075,
}

PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "Which caption better describes the image?\n"
    "Caption A: {a}\n"
    "Caption B: {b}\n"
    "Answer with exactly one letter, A or B.\n"
    "ASSISTANT:"
)


def parse_ab(text):
    """Return 'A' or 'B' (first letter found), else None."""
    m = re.search(r"[ABab]", text.strip())
    return m.group(0).upper() if m else None


def ask(model, processor, image, cap_a, cap_b, device):
    """Ask LLaVA which caption (A or B) matches the image. Returns ('A'|'B'|None, raw)."""
    prompt = PROMPT_TEMPLATE.format(a=cap_a, b=cap_b)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen, skip_special_tokens=True).strip()
    return parse_ab(raw), raw


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={MODEL_ID}  data={WG_DATA}  n={MAX_PAIRS}  seed={SEED}")

    print("Loading LLaVA-1.5-7B in 8-bit...")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=qcfg, device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()

    print("Loading Winoground...")
    ds = load_dataset(WG_DATA, split="test")
    idxs = list(range(len(ds)))
    random.Random(SEED).shuffle(idxs)
    idxs = sorted(idxs[:MAX_PAIRS])

    naive_pair = robust_pair = 0
    a_choices = b_choices = none_choices = total_q = 0
    per_pair = []

    for i in tqdm(idxs, desc="scoring"):
        ex = ds[i]
        i0 = ex["0.webp"].convert("RGB")
        i1 = ex["1.webp"].convert("RGB")
        c0 = str(ex["npy"][0])
        c1 = str(ex["npy"][1])

        # I0 (correct = C0): order1 A=C0/B=C1 -> want A ; order2 A=C1/B=C0 -> want B
        p0_o1, r0_o1 = ask(model, proc, i0, c0, c1, device)
        p0_o2, r0_o2 = ask(model, proc, i0, c1, c0, device)
        # I1 (correct = C1): order1 A=C0/B=C1 -> want B ; order2 A=C1/B=C0 -> want A
        p1_o1, r1_o1 = ask(model, proc, i1, c0, c1, device)
        p1_o2, r1_o2 = ask(model, proc, i1, c1, c0, device)

        for p in (p0_o1, p0_o2, p1_o1, p1_o2):
            total_q += 1
            if p == "A": a_choices += 1
            elif p == "B": b_choices += 1
            else: none_choices += 1

        i0_naive  = (p0_o1 == "A")
        i1_naive  = (p1_o1 == "B")
        i0_robust = (p0_o1 == "A") and (p0_o2 == "B")
        i1_robust = (p1_o1 == "B") and (p1_o2 == "A")

        naive_ok  = i0_naive and i1_naive
        robust_ok = i0_robust and i1_robust
        naive_pair  += int(naive_ok)
        robust_pair += int(robust_ok)

        per_pair.append({
            "idx": i, "c0": c0[:60], "c1": c1[:60],
            "i0": [p0_o1, p0_o2], "i1": [p1_o1, p1_o2],
            "naive": int(naive_ok), "robust": int(robust_ok),
        })

    n = len(idxs)
    naive = naive_pair / n
    robust = robust_pair / n
    results = {
        "model": MODEL_ID,
        "dataset": WG_DATA,
        "n_pairs": n,
        "seed": SEED,
        "load_in_8bit": True,
        "LLaVA_pair_score": {
            "naive_single_order": naive,
            "robust_order_consistent": robust,
        },
        "position_bias": {
            "total_queries": total_q,
            "chose_A": a_choices, "chose_B": b_choices, "unparsed": none_choices,
            "A_rate": a_choices / total_q, "B_rate": b_choices / total_q,
        },
        "CLIP_L14_ITC_baseline": CLIP_L14_BASELINE,
        "comparison": {
            "note": ("LLaVA pair-score is a TEXT/PAIR-score analog (each image picks its "
                     "caption), not a true Winoground Image/Group score. Compare to CLIP "
                     "Text (0.2825) and Group (0.075)."),
            "naive_minus_CLIP_text":  naive  - CLIP_L14_BASELINE["text_score"],
            "naive_minus_CLIP_group": naive  - CLIP_L14_BASELINE["group_score"],
            "robust_minus_CLIP_text": robust - CLIP_L14_BASELINE["text_score"],
            "robust_minus_CLIP_group":robust - CLIP_L14_BASELINE["group_score"],
        },
        "per_pair": per_pair,
    }

    print("\n" + "=" * 58)
    print(f"  n_pairs={n}  A_rate={results['position_bias']['A_rate']:.2f}  "
          f"B_rate={results['position_bias']['B_rate']:.2f}  unparsed={none_choices}")
    print(f"  LLaVA pair-score  naive={naive:.1%}   robust={robust:.1%}")
    print(f"  CLIP-L/14 cosine  Text={CLIP_L14_BASELINE['text_score']:.1%}  "
          f"Group={CLIP_L14_BASELINE['group_score']:.1%}")
    print("=" * 58)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
