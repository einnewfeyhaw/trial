"""
E8 v2: LLaVA-1.5-7B on Winoground -- proper bidirectional scoring.

Fixes a critical bug in E8_mllm_bridge.py: image_score was a literal copy of
text_score's boolean expression (i_ok = int(correct_i0 and correct_i1), same
as t_ok) -- Image Score was never independently computed. That protocol
(single-image generate() + regex-parse A/B) can only ever answer "for this
image, which caption fits better" -- which by the paper's own Section 4
definitions IS Text Score, not Image Score -- because it never compares
across images for a fixed caption. The mislabeled result made LLaVA look
like a 3.1x win on Group Score when the only thing actually measured
(23.5%) was lower than CLIP-L/14's real cosine Text Score (28.2%).

This version computes a genuine scalar compatibility score s(image, caption)
for all FOUR (image, caption) combinations per pair, using log-probability
of a forced yes/no answer rather than greedy text generation + parsing:

  prompt: "Does this image show: {caption}? Answer yes or no."
  score  = logprob("yes") - logprob("no")  at the first generated token

From s00, s01, s10, s11 -- exactly the four scores the paper's own cosine
evaluation uses -- Text/Image/Group Score are derived with the IDENTICAL
comparison structure as E3_second_2x2.py's winoground_scores(). This makes
the LLaVA numbers genuinely comparable to the CLIP-L/14 cosine baseline for
the first time.

Output: E8_results_v2.json
"""

import json
import os
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

warnings.filterwarnings("ignore")

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
WG_DATA = "haideraltahan/wds_winoground"
OUT_FILE = "E8_results_v2.json"
MAX_PAIRS = 400

# CLIP-L/14 cosine baseline, verified in-session (matches Table 4 / E3 sanity checks)
CLIP_L14_BASELINE = {
    "text_score": 0.2825,
    "image_score": 0.105,
    "group_score": 0.075,
}

PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "Does this image show: {caption}\n"
    "Answer yes or no.\n"
    "ASSISTANT:"
)


def find_token_variants(tok, words):
    """Collect first-token ids for case/leading-space variants of each word.
    Vicuna's tokenizer is SentencePiece-based, so 'yes' and ' yes' commonly
    tokenize differently -- print the decoded results and sanity-check them
    before trusting the run."""
    ids = set()
    for w in words:
        for variant in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) >= 1:
                ids.add(enc[0])
    return sorted(ids)


def get_yesno_score(model, processor, image, caption, device, yes_ids, no_ids):
    """logprob(yes) - logprob(no) at the next-token position, via a single
    forward pass (no generation, no text parsing)."""
    prompt = PROMPT_TEMPLATE.format(caption=caption)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    yes_lp = torch.logsumexp(logprobs[yes_ids], dim=0).item()
    no_lp = torch.logsumexp(logprobs[no_ids], dim=0).item()
    return yes_lp - no_lp


def winoground_scores(s00, s01, s10, s11):
    """Identical comparison structure to E3_second_2x2.py's winoground_scores()."""
    t_ok = (s00 > s01) and (s11 > s10)
    i_ok = (s00 > s10) and (s11 > s01)
    return int(t_ok), int(i_ok), int(t_ok and i_ok)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\nmodel:  {MODEL_ID}")

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    yes_ids = find_token_variants(tok, ["yes"])
    no_ids = find_token_variants(tok, ["no"])
    print(f"yes token ids: {yes_ids} -> {[tok.decode([i]) for i in yes_ids]}")
    print(f"no  token ids: {no_ids} -> {[tok.decode([i]) for i in no_ids]}")
    print("SANITY-CHECK these decoded strings before trusting the run below.")

    ds = load_dataset(WG_DATA, split="test")
    n = min(MAX_PAIRS, len(ds))

    text_c = image_c = group_c = 0
    per_pair = []

    for i, ex in enumerate(tqdm(ds.select(range(n)), desc="scoring")):
        i0 = ex["0.webp"].convert("RGB")
        i1 = ex["1.webp"].convert("RGB")
        c0 = str(ex["npy"][0])
        c1 = str(ex["npy"][1])

        s00 = get_yesno_score(model, proc, i0, c0, device, yes_ids, no_ids)
        s01 = get_yesno_score(model, proc, i0, c1, device, yes_ids, no_ids)
        s10 = get_yesno_score(model, proc, i1, c0, device, yes_ids, no_ids)
        s11 = get_yesno_score(model, proc, i1, c1, device, yes_ids, no_ids)

        t_ok, i_ok, g_ok = winoground_scores(s00, s01, s10, s11)
        text_c += t_ok; image_c += i_ok; group_c += g_ok

        per_pair.append({
            "idx": i, "s00": s00, "s01": s01, "s10": s10, "s11": s11,
            "text": t_ok, "image": i_ok, "group": g_ok,
        })

    results = {
        "model": MODEL_ID,
        "dataset": WG_DATA,
        "n_pairs": n,
        "scoring_method": (
            "logprob(yes)-logprob(no) per (image,caption) via single forward "
            "pass (no generation/parsing); Text/Image/Group derived with the "
            "same comparison structure as E3_second_2x2.py's winoground_scores()"
        ),
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "LLaVA": {
            "text_score": text_c / n,
            "image_score": image_c / n,
            "group_score": group_c / n,
        },
        "CLIP_L14_ITC_baseline": CLIP_L14_BASELINE,
        "delta_vs_CLIP_L14": {
            "text_score": text_c / n - CLIP_L14_BASELINE["text_score"],
            "image_score": image_c / n - CLIP_L14_BASELINE["image_score"],
            "group_score": group_c / n - CLIP_L14_BASELINE["group_score"],
        },
        "per_pair": per_pair[:20],
    }

    print(f"\n{'='*55}")
    print(f"  n_pairs={n}")
    print(f"  {'Metric':<16} {'CLIP-L/14 ITC':>14} {'LLaVA-1.5-7B':>14} {'Delta':>8}")
    print(f"  {'-'*54}")
    for k in ["text_score", "image_score", "group_score"]:
        print(f"  {k:<16} {CLIP_L14_BASELINE[k]:>14.1%} {results['LLaVA'][k]:>14.1%} "
              f"{results['delta_vs_CLIP_L14'][k]:>+8.1%}")
    print(f"{'='*55}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
