# Rebuttal Checklist — Submission 18981

Master tracking doc. Supersedes `rebuttal_strategy.md` and `rebuttal_final.md`
(both contained claims later found false against the code — see git history).

**Rule, unchanged:** nothing here is final rebuttal text until marked ✅ DONE
with a result attached. 🔄 RUNNING / ⬜ TODO items get honest "committed for
revision" language, not invented numbers.

Status key: ✅ DONE &nbsp; 🔄 RUNNING &nbsp; ⬜ TODO &nbsp; ⚠️ NEEDS DECISION

---

## AC gating items (from meta-review — these are mandatory)

| # | Mandate | Status | Evidence |
|---|---|---|---|
| G1 | Robustness of feature-importance measure | ⚠️ MIXED | E1 (permutation importance): only 2/5 models show significant scale-free anticorrelation (CLIP-L/14 masked, SO400M reversed; B/32, LAION, SigLIP-B null). E2 (access ratio, probe-free): cleaner signal but LAION contradicts the CLIP-family story (2.02, reversed) and SigLIP-B is basis-unstable (0.63 proj vs 5.99 pca). **Needs bootstrap CIs on E2 before this is presentable — see Open Items.** |
| G2 | Gains on another benchmark, or proof 1×2 nullifies it | ✅ **DONE** | ColorSwap, all 5 models, full control suite. 1.8×–4.5× relative gain, p<0.0001 every model, `random_pair` null (p=0.30–0.78), single-caption controls hurt as expected. See draft text below. |
| G3 | Reversal on more than one model | ⚠️ MIXED | Table 1 as submitted: SigLIP-base masked (−0.223), SO400M aligned (+0.394) — opposite signs, same family. E1 permutation importance: SigLIP-B is null (not masked, not aligned) — doesn't fully corroborate the original Table 1 direction. E2 PCA: both SigLIP models >1 (reversed), but access ratio for SigLIP-B is basis-dependent (0.63 vs 5.99). Reversal for SO400M is robust across every measure; SigLIP-B's status is genuinely unsettled. State that honestly. |
| G4 | Why 1.7% variance is "structurally significant" | ✅ **DONE** | Retire ρ² entirely. E7 (bootstrapped, N=10,000, correctly filtered to swap_obj, n=245): access_ratio=0.484, 95% CI [0.448, 0.530], p=0.0 that ratio≥1 — masking reliable across every resample. Cross-validated independently by E2's point estimate (0.47, same model/dataset/subset, different script). Two independently-coded, now bug-fixed pipelines converging this tightly is strong evidence the number is real, not a proxy artifact. See draft text below. |
| G5 | Erasure: comparable gains across 5 models | ✅ DONE (already in submission) | Table 4 of the paper already shows this for Winoground. ColorSwap (G2) now replicates it on a second benchmark. |
| G6 | Survival claim too broad | ✅ DONE | Scope conceded: object-level binding / shared-content interference in 2×2 cosine retrieval, not compositionality in general. Matches paper's own limitations section — this is a rescoping, not a retraction. |

---

## Per-reviewer items

### BTbD (2 → target 3–4)

| Item | Status | Notes |
|---|---|---|
| 7 missing citations | ⬜ TODO | Mechanical. Add to related work. |
| Differentiate from Koishigarina et al. 2025 (the real driver of originality=1) | ⬜ TODO | Do NOT reuse the Prop-1-predicts-their-asymmetry argument from the earlier draft — it conflated two different axes (uni/cross-modal vs. 1×2/2×2) and doesn't hold. Need either a real differentiating experiment (uni-modal vs cross-modal analysis under our framework) or a more careful textual distinction. **Unresolved — flag as open.** |
| Eq. 1 definition ambiguity | ⬜ TODO | One sentence: addition/subtraction on L2-normalized text embeddings. Free. |
| §4.3 mechanics unclear | ⬜ TODO | One sentence: applied to frozen post-projection embeddings at eval time. Free. |
| Probe train/test protocol unstated | ⬜ TODO | State plainly, including the correction from E1/E2 (pair-wise split, held-out). |
| Citation format / spacing bug | ⬜ TODO | Mechanical. |
| "Only 2 of 7 SugarCrepe splits" | ⬜ TODO | E2 `--per-subset` not yet run. See Open Items. |

### iv2d (3 → target 4)

| Item | Status | Notes |
|---|---|---|
| Q1: reversal beyond one model | ⚠️ SEE G3 | Genuinely mixed evidence — do not overclaim. SO400M reversal is solid; SigLIP-B is not clean either direction. |
| Q2: any benchmark beyond Winoground | ✅ DONE | ColorSwap (G2). This is the headline new result for iv2d. |
| Q3: downstream LLM consumption (LLaVA) | ⬜ TODO / open question | Not run. State as committed future work, not a claim. |
| Objective vs. architecture vs. matching-function isolation | ✅ DONE (negative) | See v8Kz item 4 below. |

### v8Kz (4 → target 5)

| Item | Status | Notes |
|---|---|---|
| 1. Robustness of feature-importance measure | ⚠️ MIXED | See G1. Report honestly — this is not a clean win. |
| 2. Each dim's actual contribution to cosine similarity | ✅ DONE | E2 cosine-mass-share is an exact identity, not a proxy. This fully answers the literal request regardless of how G1 nets out. |
| 3. Spectral placement of C_mean/Δ vs. erasure gain | ✅ DONE (null result) | E6: all 5 bridge correlations null (p=0.14–0.74). C_mean concentrates in *low*-σ dims, opposite the paper's narrative. Concede directly: global SVD story and instance-specific erasure are independent mechanisms. |
| 4. Objective vs. architecture vs. matching-function controls | ✅ DONE (negative result) | E5 sweep complete: as weight_decay relaxes and the learned bilinear scorer A moves further from cosine, zero-shot Winoground performance falls monotonically for all 5 models, dropping below cosine baseline at the loosest setting. Best zero-shot result across the whole sweep is the tightest regularization (+0.25–0.75pp over cosine) — i.e. barely different from cosine at all. Answers the "matching function" half of v8Kz's ask with a genuine negative: a global bilinear correction trained cross-benchmark (ColorSwap→Winoground) does not recover compositional signal, and more capacity only overfits. The "objective vs architecture" half (BLIP ITC/ITM or similar) was never built — see Open Items #7 for the go/no-go decision. |
| 5. Rescope conclusions | ✅ DONE | See G6. |

### g6iB (4 → target 5)

| Item | Status | Notes |
|---|---|---|
| Q1: deployable inference mechanics | ✅ DONE | E3: top-K candidate-set erasure (retrieve top-K captions by cosine, erase their mean, re-rank). K=2 gives significant Group Score gain (9.0%→14.5%, p=0.0012), degrades toward baseline as K grows -- exactly the generic-intervention argument from Section 5, now demonstrated as a curve rather than asserted. **Caveat that must be in the rebuttal text:** Text Score drops at every K (each image gets its own independent erasure direction, breaking Prop 1's shared-C_mean symmetry) -- this is a real trade-off, not a free win. State it plainly. |
| Q2: 1.7% variance justification | ✅ DONE | See G4. access_ratio=0.484 [0.448, 0.530], cross-validated by E2's 0.47. Draft text ready below. |
| Q3: what else explains remaining variance | ⬜ TODO | Open question, can answer with prose (candidate factors: cross-encoder direction alignment, training data frequency, attention-pooling geometry) without new experiments. |

---

## Draft rebuttal text — ready to use

### G2 / iv2d Q2 (ColorSwap generalization) — the strongest result so far

> We address this with a new experiment on **ColorSwap** (Burapacheep et al.,
> 2024), a second genuinely 2×2 compositional benchmark: two images × two
> captions per group, testing color-attribute binding rather than spatial
> arrangement. We ran the paper's full protocol unchanged — identical scoring,
> identical control suite (single-caption erasure, random-pair erasure,
> image-side symmetry) — on 300 test pairs across all five models.
>
> | Model | Baseline GS | Mean-Erasure GS | Relative gain | p (one-sided) |
> |---|---|---|---|---|
> | CLIP-B/32 | 12.0% | 35.7% | 2.97× | <0.0001 |
> | CLIP-L/14 | 7.3% | 32.7% | 4.45× | <0.0001 |
> | LAION-CLIP-B/32 | 23.3% | 56.3% | 2.41× | <0.0001 |
> | SigLIP-B/16 | 30.3% | 61.3% | 2.02× | <0.0001 |
> | SigLIP-SO400M | 37.0% | 67.3% | 1.82× | <0.0001 |
>
> As on Winoground, the effect is specific to the matched instance-level
> direction: `random_pair` erasure shows no significant effect on any model
> (p = 0.30–0.78), and single-caption-only erasure (`c0_only`/`c1_only`)
> *reduces* Group Score below baseline for every model, exactly the
> specificity signature reported in Table 2. This demonstrates the phenomenon
> is not a Winoground-specific artifact but generalizes across compositional
> benchmarks testing a different swap axis.

**Caveat to resolve before camera-ready, not before rebuttal:** the ColorSwap
data we used was sourced from the authors' public Google Drive link (linked
from their GitHub), because `stanfordnlp/colorswap` on HF is gated and we
could not get access approved in time. Cite the actual paper's canonical test
split, not the GDrive URL, in any camera-ready methods section.

### G4 / g6iB Q2 (why 1.7% variance is structurally significant)

> We agree ρ² is the wrong statistic and retire it. We replace it with a
> direct, probe-free measurement requiring no MLP, no seeds, and no
> standardization choice: for each singular-basis dimension, its exact
> contribution to cosine similarity (`v'_i · t'_i`, which sums exactly to
> `cos(v,t)` by orthonormality of the basis) and its discriminability
> (|AUC−0.5| as a univariate match/mismatch classifier). We then ask: does
> cosine similarity give the most discriminative dimensions their fair share
> of the score?
>
> On CLIP-B/32, SugarCrepe swap-object (n=245, matching the probe's original
> evaluation set), the top-5% most discriminative dimensions receive only
> **0.484× their proportional cosine weight** (95% CI [0.448, 0.530],
> 10,000-sample bootstrap; P(ratio ≥ 1.0) = 0.0 across every resample). We
> independently cross-validated this with a second implementation
> (a different script computing the same quantity from scratch): 0.47,
> agreeing within the first CI's width. Two independently-coded pipelines
> converging this tightly is strong evidence this is a real geometric
> property, not an artifact of either implementation.
>
> A complementary view of the same data: the top-20% highest-singular-value
> dimensions carry 57.3% of total cosine mass but only 20.2% of
> discriminative signal — a 2.84× over-representation of object-identity
> content relative to compositional-discriminative content in the score
> itself. This is the structural mechanism the original ρ²=1.7% was gesturing
> at but measuring indirectly and noisily; the direct measurement is both
> stronger and more interpretable.

### E6 (spectral bridge) — concession, not a defense

> We tested directly whether the global SVD spectral-placement analysis
> (Sections 3–4) predicts per-pair mean-erasure benefit, as Reviewer v8Kz
> requested. It does not: across 400 Winoground pairs, correlations between
> Δ's spectral placement and per-pair erasure gain are null (ρ ≈ 0.05,
> p = 0.31), and C_mean concentrates in low-, not high-, singular-value
> dimensions (34.8% of its energy in the bottom quintile vs. a 20% uniform
> baseline). We revise Sections 3–5 to present the spectral analysis and the
> instance-specific erasure result as **complementary, independently
> supported findings** rather than a single causal chain — the erasure result
> stands on its own controls (Table 2, and now ColorSwap above) regardless of
> the spectral placement question.

### v8Kz item 4 (matching-function control) — honest negative result

> To isolate whether cosine-based matching specifically (rather than the
> representation itself) is responsible, we tested a learned matching
> function on the same frozen embeddings: `score(v,t) = v^T A t`, with `A`
> initialized to identity (reproducing cosine exactly) and regularized
> toward it by weight decay. We trained `A` on ColorSwap match/mismatch
> pairs (700 train pairs) and evaluated zero-shot on Winoground (`A` never
> sees Winoground during training), sweeping weight decay over
> {1e-2, 1e-3, 1e-4} to test regularization strength directly rather than
> assume a single setting is representative.
>
> The result is a genuine negative: as weight decay relaxes and `A` moves
> further from cosine, zero-shot Winoground Group Score *falls*
> monotonically for all five models, dropping below the cosine baseline at
> the loosest setting, while training-set performance on ColorSwap keeps
> improving — the standard signature of overfitting rather than a
> recoverable correction. The best zero-shot result across the entire sweep
> is the most tightly regularized setting, statistically indistinguishable
> from cosine (+0.25 to +0.75 percentage points).
>
> We report this directly rather than omit it: a global bilinear correction
> to cosine, learned on one compositional axis (color-attribute swaps) and
> transferred to another (spatial-arrangement swaps), does not recover the
> signal our probes suggest survives the projection. This does not
> contradict the paper's central claim — mean-erasure (Table 2, and our new
> ColorSwap result) demonstrates the information is recoverable by an
> *instance-specific* intervention with access to both candidates — but it
> does mean a fixed, learned, cross-benchmark matching function is not a
> substitute for that, consistent with Section 5's finding that generic
> (non-instance-specific) interventions fail. We did not additionally test
> objective-vs-architecture isolation (e.g., matched-architecture comparisons
> across training objectives); we flag this as a specific, well-scoped
> direction for future work rather than claim it.

**Strategic note:** this result reinforces, rather than undermines, the
paper's actual core argument once reframed around instance-specificity
(see Discussion section on the strategic pivot). It became a *point in favor*
of "only instance-specific, foil-aware intervention works" — Section 5's
argument, now doubly supported by both the original generic-intervention
failures AND this cross-benchmark matching-function failure.

---

## Open items before rebuttal is submission-ready

1. ~~E2 bootstrap CIs~~ — ✅ DONE via E7 (cross-validated, see G4).
   ~~`--per-subset`~~ — ✅ DONE via E7b, all 7 SugarCrepe categories. Key
   finding: the aggregate top20pct_sv_dims amplification (2.35–3.07×) is
   consistent in the SAME direction across all 7 categories with no
   exceptions — the robust, generalizable claim. The finer-grained
   access_ratio statistic is bimodal (masking for swap_obj/swap_att/
   replace_obj/replace_att; alignment for add_obj/add_att/replace_rel) --
   add_obj/add_att's flip is well-explained by a verified caption-length
   asymmetry (false caption is a strict superset, consistent with Hsieh
   et al. 2023's hackability finding, already cited). replace_rel's flip
   has NO verified mechanistic explanation yet (the captions are same-length,
   structurally identical to the masking categories) -- do not assert a
   causal story for it in any rebuttal text; report the aggregate finding
   only, or flag replace_rel as an open question if pressed.
2. ~~E5 results~~ — ✅ DONE (negative result, see v8Kz item 4 above).
3. **Koishigarina differentiation** — still unresolved; the earlier draft's
   argument didn't hold up. Needs either a real experiment or a more careful
   rewrite.
4. ~~Top-k re-ranking deployability sketch~~ — ✅ DONE via E3 (candidate-set
   mean, not corpus mean — the corpus-mean framing from the earlier draft
   would have contradicted Section 5's own negative result). Remember the
   Text Score trade-off caveat when writing this into the rebuttal text.
5. **ColorSwap provenance** — GDrive link is fine for rebuttal, needs a
   citable source before camera-ready.
6. Decide how honestly to present G1/G3 — the evidence is genuinely mixed
   (SigLIP-B in particular doesn't cleanly support either "masked" or
   "reversed" depending on which measure you read). Recommend leading with
   what's solid (SO400M reversal, ColorSwap generalization, E2 cosine-mass
   identity) and stating the SigLIP-B ambiguity plainly rather than picking
   whichever number reads best per-paragraph — that's the failure mode from
   the last draft.
7. **Go/no-go on E5a (BLIP ITC vs ITM, or another objective-isolation
   control).** Three follow-up mechanism experiments in a row (E1/E2 mixed,
   E6 null, E5 null) point the same direction: the paper's explanatory
   story is weak, the erasure phenomenon itself is not. A fourth experiment
   in the same family risks the same outcome for real engineering cost (new
   model family, new preprocessing pipeline). Recommend deprioritizing E5a
   and spending remaining time on items 1, 3, 4 above instead, which are
   cheaper and directly close specific reviewer asks rather than probe the
   mechanism further. Revisit only if rebuttal time budget allows after 1/3/4.
