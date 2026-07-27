# E5 Results: Learned Bilinear Scorer — ColorSwap-train → Winoground Zero-shot
**NeurIPS 2026 Rebuttal — Submission 18981**
*Run: 2026-07-27 | GPU: A100-SXM4-40GB | Repo: github.com/einnewfeyhaw/trial (commit 00a1b89)*

---

## STEP 1 — SANITY CHECK: PASSED

**Model**: `openai/clip-vit-base-patch32`

`cosine_baseline_winoground.group = 0.09` ✅

This equals the paper's 9.0% baseline (A = I reduces exactly to cosine similarity — confirmed).

---

## RAW JSON OUTPUT (all 5 models, unedited)

```json
{
  "openai/clip-vit-base-patch32": {
    "model": "openai/clip-vit-base-patch32",
    "n_train_colorswap": 700,
    "n_eval_winoground": 400,
    "cosine_baseline_winoground": {
      "text": 0.31,
      "image": 0.1125,
      "group": 0.09
    },
    "learned_bilinear_train_colorswap": {
      "text": 0.4328571428571429,
      "image": 0.20857142857142857,
      "group": 0.15857142857142856
    },
    "learned_bilinear_eval_winoground_ZEROSHOT": {
      "text": 0.305,
      "image": 0.115,
      "group": 0.095
    },
    "final_train_loss": 0.687048614025116,
    "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground"
  },
  "openai/clip-vit-large-patch14": {
    "model": "openai/clip-vit-large-patch14",
    "n_train_colorswap": 700,
    "n_eval_winoground": 400,
    "cosine_baseline_winoground": {
      "text": 0.2825,
      "image": 0.105,
      "group": 0.075
    },
    "learned_bilinear_train_colorswap": {
      "text": 0.36428571428571427,
      "image": 0.18857142857142858,
      "group": 0.13142857142857142
    },
    "learned_bilinear_eval_winoground_ZEROSHOT": {
      "text": 0.28,
      "image": 0.1025,
      "group": 0.0775
    },
    "final_train_loss": 0.6875097751617432,
    "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground"
  },
  "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
    "model": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "n_train_colorswap": 700,
    "n_eval_winoground": 400,
    "cosine_baseline_winoground": {
      "text": 0.35,
      "image": 0.1125,
      "group": 0.0825
    },
    "learned_bilinear_train_colorswap": {
      "text": 0.6557142857142857,
      "image": 0.3357142857142857,
      "group": 0.3028571428571429
    },
    "learned_bilinear_eval_winoground_ZEROSHOT": {
      "text": 0.3525,
      "image": 0.115,
      "group": 0.09
    },
    "final_train_loss": 0.6780341267585754,
    "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground"
  },
  "google/siglip-base-patch16-224": {
    "model": "google/siglip-base-patch16-224",
    "n_train_colorswap": 700,
    "n_eval_winoground": 400,
    "cosine_baseline_winoground": {
      "text": 0.33,
      "image": 0.13,
      "group": 0.1025
    },
    "learned_bilinear_train_colorswap": {
      "text": 0.6942857142857143,
      "image": 0.48428571428571426,
      "group": 0.4228571428571429
    },
    "learned_bilinear_eval_winoground_ZEROSHOT": {
      "text": 0.325,
      "image": 0.1425,
      "group": 0.11
    },
    "final_train_loss": 0.6769757270812988,
    "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground"
  },
  "google/siglip-so400m-patch14-384": {
    "model": "google/siglip-so400m-patch14-384",
    "n_train_colorswap": 700,
    "n_eval_winoground": 400,
    "cosine_baseline_winoground": {
      "text": 0.375,
      "image": 0.165,
      "group": 0.1225
    },
    "learned_bilinear_train_colorswap": {
      "text": 0.7314285714285714,
      "image": 0.5514285714285714,
      "group": 0.5
    },
    "learned_bilinear_eval_winoground_ZEROSHOT": {
      "text": 0.3725,
      "image": 0.1625,
      "group": 0.13
    },
    "final_train_loss": 0.6720477938652039,
    "note": "eval is zero-shot: A trained on ColorSwap only, never sees Winoground"
  }
}
```

---

## Per-model zero-shot Winoground group score vs cosine baseline

| Model | cosine_baseline_winoground.group | learned_bilinear_eval_winoground_ZEROSHOT.group | Direction |
|---|---|---|---|
| clip-vit-base-patch32 | 0.09 | 0.095 | **up** (+0.005) |
| clip-vit-large-patch14 | 0.075 | 0.0775 | **up** (+0.0025) |
| CLIP-ViT-B-32-laion | 0.0825 | 0.09 | **up** (+0.0075) |
| siglip-base-patch16-224 | 0.1025 | 0.11 | **up** (+0.0075) |
| siglip-so400m-patch14-384 | 0.1225 | 0.13 | **up** (+0.0075) |

All five models went **up** on zero-shot Winoground group score relative to the cosine baseline. The increases range from +0.0025 to +0.0075 (absolute).

---

## ColorSwap-train vs Winoground-zeroshot comparison (overfitting signature check)

| Model | learned_bilinear_train_colorswap.group | learned_bilinear_eval_winoground_ZEROSHOT.group | Gap |
|---|---|---|---|
| clip-vit-base-patch32 | 0.1586 | 0.095 | 0.0636 |
| clip-vit-large-patch14 | 0.1314 | 0.0775 | 0.0539 |
| CLIP-ViT-B-32-laion | 0.3029 | 0.090 | 0.2129 |
| siglip-base-patch16-224 | 0.4229 | 0.110 | 0.3129 |
| siglip-so400m-patch14-384 | 0.5000 | 0.130 | 0.3700 |

Per reporting requirement #3: For LAION-CLIP, SigLIP-base, and SigLIP-SO400M, the train ColorSwap group score is substantially higher than the zero-shot Winoground group score (gaps of 0.21, 0.31, 0.37 respectively). This is the **overfitting-to-ColorSwap signature** described in the reporting instructions. It is reported as-is.

For CLIP-B/32 and CLIP-L/14 the train ColorSwap group score is also elevated but more modestly (gaps of 0.06, 0.05).

---

## Model failures

None. All 5 models ran to completion without error.

---

## Provenance

- **Training data**: ColorSwap TRAIN split, 700 pairs, local at `/tmp/colorswap/data` (same `--data-dir` used in E3)
- **Eval data**: `haideraltahan/wds_winoground` test split, 400 pairs
- **Script**: `E5_learned_scorer.py` — no hyperparameter modifications (epochs=200, lr=1e-2, weight_decay=1e-2)
- **Sanity JSON**: `E5_sanity.json`
- **Full results JSON**: `E5_learned_scorer.json`
- **Git commit**: `00a1b89` (local)
- **Git push**: ❌ BLOCKED — no GitHub PAT. To push: `cd /content/trial && git remote set-url origin https://<PAT>@github.com/einnewfeyhaw/trial.git && git push`
