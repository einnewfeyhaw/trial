# E9 — Per-Subset Probe Accuracy Across All 7 SugarCrepe Categories

**Purpose:** Answer Reviewer BTbD's core complaint — the paper's 80.3% probe
accuracy was reported for only 1 of 7 SugarCrepe categories (labeled swap_obj).
This runs the paper's exact held-out probe protocol on **all 7 categories**.

**Status:** Experiment complete. `E9_results.json` committed locally
(`54f76f2`). `git push` **BLOCKED** — no GitHub credentials in this environment
(gh not logged in, no token). Commit is staged and ready to push.

---

## 1. Setup (what was run)

- Repo: `github.com/einnewfeyhaw/trial` (cloned to `/content/trial`)
- Script: `E9_probe_per_subset.py` (unmodified)
- Command: `python3 E9_probe_per_subset.py --limit-per-subset 1000`
- Model: `openai/clip-vit-base-patch32`
- Dataset: `haideraltahan/wds_sugarcrepe` (split=test, 7511 examples)
- Protocol (identical to paper's `corrected_probe_eval.py`): features =
  concat[L2-normed post-projection image embed, L2-normed text embed]; label 1
  for true caption, 0 for false caption; `MLPClassifier(hidden_layer_sizes=(512,256),
  max_iter=1000, random_state=42, early_stopping=True)`; pair-wise 80/20 held-out
  split, seed 42.
- Hardware: single Tesla T4 (CUDA).

The **only** change vs. the original probe script is a per-subset filter
(`subset_of`, keyed on the `split.txt` field), so each split genuinely contains
only its own category.

---

## 2. Results — printed summary table (all 7 categories)

Post-projection held-out probe accuracy. Chance = 50%.

| Category      | n_pairs | test_samples | Accuracy |
|---------------|---------|--------------|----------|
| swap_obj      | 245     | 98           | **50.0%** |
| swap_att      | 666     | 268          | **51.5%** |
| replace_obj   | 1000    | 400          | **58.2%** |
| replace_att   | 788     | 316          | **50.0%** |
| replace_rel   | 1000    | 400          | **61.0%** |
| add_obj       | 1000    | 400          | **80.5%** |
| add_att       | 692     | 278          | **70.5%** |

Raw output: `/content/trial/E9_results.json` (full JSON reproduced in Appendix).

**Plain reading (no spin):** Only `add_obj` (80.5%) and `add_att` (70.5%) are
clearly above chance. `replace_rel` (61.0%) and `replace_obj` (58.2%) are weakly
above chance. `swap_obj`, `swap_att`, and `replace_att` are at/near chance
(50.0–51.5%). Several categories at ~50% is the **expected honest outcome** and
directly corroborates the paper's own limitations section (which already reports
~51.8% swap_att, 50.0% replace_rel).

---

## 3. Correctness observation about the paper's 80.3% headline

This is a factual protocol observation surfaced during verification — not
interpretation or rebuttal text.

The paper's headline "MLP probe: **80.3% on SugarCrepe (Swap Object split)**" is
produced by `neurips_26/01_probing_tests/corrected_probe_eval.py`. That script
does **not filter by subset**. It uses:

```python
limit = 3000
for i, example in enumerate(dataset.select(range(min(limit, len(dataset))))):
```

i.e. the **first 3000 examples** of the dataset. The dataset is ordered by
category, so the first 3000 examples decompose as (verified):

| In first 3000 | count | share |
|---------------|-------|-------|
| add_obj       | 2062  | 68.7% |
| add_att       | 692   | 23.1% |
| replace_obj   | 246   | 8.2%  |
| **swap_obj**  | **0** | **0%** |

There are **zero swap_obj examples** in the data that produced the 80.3% number.
The probe labeled "Swap Object split" is actually an add_obj-dominated mixed
probe. Consistent with this, E9's `add_obj`-only probe scores **80.5%** — matching
the 80.3% headline — while the true held-out `swap_obj` probe is at **chance
(50.0%)**.

Implication for the response to BTbD: the reviewer's complaint is not only valid
but conservative. The 80.3% figure is not just "1 of 7 splits" — it is mislabeled;
the split it is attributed to (swap_obj) contains no swap_obj data and probes at
chance when run correctly. The honest per-subset picture is the table in §2.

*(Note: this observation should be verified independently by the authors against
their own run logs before being used in any revision; it is reported here as an
experimental finding, not a recommendation on how to frame the rebuttal.)*

---

## 4. Reproducibility / provenance

- `E9_results.json` — committed at `54f76f2` in `/content/trial` (push pending
  credentials).
- Composition of first-3000 verified with:
  `python3 -c "... subset_of over dataset.select(range(3000)) ..."` →
  `{add_obj:2062, add_att:692, replace_obj:246}`.
- Full dataset composition: add_obj 2062, add_att 692, replace_obj 1652,
  replace_att 788, replace_rel 1406, swap_obj 245, swap_att 666 (total 7511).
- Smoke test (`--limit-per-subset 25`) ran end-to-end first; all 50% at n=25 as
  expected for a 5-pair test set.

### Appendix — full E9_results.json
```json
{
  "swap_obj":    {"n_pairs": 245,  "train_samples": 392,  "test_samples": 98,  "post_projection_accuracy": 0.5},
  "swap_att":    {"n_pairs": 666,  "train_samples": 1064, "test_samples": 268, "post_projection_accuracy": 0.5149253731343284},
  "replace_obj": {"n_pairs": 1000, "train_samples": 1600, "test_samples": 400, "post_projection_accuracy": 0.5825},
  "replace_att": {"n_pairs": 788,  "train_samples": 1260, "test_samples": 316, "post_projection_accuracy": 0.5},
  "replace_rel": {"n_pairs": 1000, "train_samples": 1600, "test_samples": 400, "post_projection_accuracy": 0.61},
  "add_obj":     {"n_pairs": 1000, "train_samples": 1600, "test_samples": 400, "post_projection_accuracy": 0.805},
  "add_att":     {"n_pairs": 692,  "train_samples": 1106, "test_samples": 278, "post_projection_accuracy": 0.7050359712230215}
}
```

---

## 5. Text-only fixes for BTbD (paste-ready)

**BLOCKED capability:** no LaTeX/`.tex` source is present in the workspace (only
the compiled PDF and the code release). These fixes therefore could not be
committed "into the revision" here. They are provided below as ready-to-paste
edits. Line numbers refer to the reviewer copy PDF.

### 5.1 Add the 7 missing citations
Add to `.bib` and cite in Related Work:

- **[1] TripletCLIP** — Patel et al., 2024. *TripletCLIP: Improving Compositional
  Reasoning of CLIP via Synthetic Vision-Language Negatives.* (training-time
  hard-negative approach) → cite in "Compositionality Recovery Methods" alongside
  Yuksekgonul et al. / Doveh et al.
- **[2]** Peleg et al., 2025. *Advancing Compositional Awareness in CLIP with
  Efficient Fine-Tuning.* → same paragraph as [1].
- **[3] SUGARCREPE++** — Dumpala et al., 2024. *Vision-Language Model Sensitivity
  to Semantic and Lexical Alterations.* → "Compositionality Benchmarks" list.
- **[4] VALSE** — Parcalabescu et al., 2022. *A Task-Independent Benchmark ...
  Centered on Linguistic Phenomena.* → "Compositionality Benchmarks" list.
- **[5]** Kang et al., 2025. *Is CLIP ideal? No. Can we fix it? Yes!* → cite for
  patch-token alignment over frozen encoders AND the geometric bottleneck of
  cosine similarity (Related Work + Intro L40 question).
- **[6]** Koishigarina et al., 2025. *CLIP Behaves like a Bag-of-Words Model
  Cross-modally but not Uni-modally.* → cite at Intro L40 and in the modality-gap
  paragraph; directly relevant to the "geometrically inaccessible to cosine
  similarity" framing.
- **[7]** Miranda et al., 2026. *Revisiting Compositionality in Dual-Encoder VLMs:
  The Role of Inference.* (already cited as Miranda et al. 2026 — ensure the
  arXiv id/title match the reviewer's [7] and cite it in the L40 discussion too).

Also address L40 explicitly: add one sentence relating the "destroyed vs.
geometrically inaccessible" question to [5], [6], [7], e.g.:
> "This question is closely related to recent analyses of the cosine-similarity
> geometric bottleneck [5,6] and of inference-time compositional recovery [7];
> we differ by localizing the effect to singular-value magnitude."

### 5.2 Define Eq. (1) explicitly
Current Eq. (1) uses `C0`, `C1` ambiguously (caption vs. embedding). Replace the
lead-in with:
> "Let C0, C1 ∈ R^{d_out} denote the **L2-normalized text embeddings** (not the
> raw captions) of the two candidate captions, as produced by the text encoder
> and projection. Define Cmean = ½(C0 + C1) and Δ = ½(C0 − C1)."

This makes the addition/subtraction well-defined (element-wise in embedding
space) and consistent with Eq. (2) and Proposition 1's normalization assumption.

### 5.3 One sentence on Section 4.3
Add to the start of §4.3:
> "All mean-erasure operations are applied at evaluation time to the **frozen,
> post-projection** image embeddings (Eq. 2); no model weights are modified and
> no training occurs — the projection is a deterministic geometric transform of
> the encoder's output embeddings."

### 5.4 Citation format + spacing
- Replace narrative `\citet{}` with `\citep{}` where the citation is parenthetical,
  e.g. L21 "Contrastive Language-Image Pre-training (CLIP) Radford et al. [2021]"
  → "Contrastive Language-Image Pre-training (CLIP)~\citep{radford2021}". Sweep
  the whole paper for the "Name et al. [year]" pattern used as a parenthetical.
- Fix L26 spacing bug: `like"a mug` → `like ``a mug` (add the missing space and
  use proper opening quotes). Grep the source for `like"` / word-adjacent quotes.

### 5.5 State the probe protocol explicitly
Add to §3.4 (and/or §3.2) a protocol sentence:
> "Each probe is trained with a **pair-wise 80/20 held-out split** (both samples
> of a pair kept on the same side of the split), **random seed 42**, and the
> **identical MLP configuration and learning rate across all models and
> datasets** (`MLPClassifier(hidden_layer_sizes=(512,256), max_iter=1000,
> early_stopping=True)`, Adam defaults). Reported accuracy is on the held-out
> test pairs only."

Additionally — and importantly given §3 of this report — clarify which SugarCrepe
data each probe number is computed on. If the intended claim is the swap_obj
split, the probe must be **filtered to swap_obj** (E9 result: 50.0%, n=245).
The current headline number is computed on the first 3000 (unfiltered,
add_obj-dominated) examples.
