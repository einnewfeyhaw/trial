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
| G4 | Why 1.7% variance is "structurally significant" | 🔄 PARTIAL | Retire ρ² framing entirely, replace with E2 access ratio (mass share vs. discriminability rank). Needs the bootstrap CI (below) before it's a defensible number rather than a point estimate off n=245. |
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
| Objective vs. architecture vs. matching-function isolation | 🔄 RUNNING | E5 (learned bilinear scorer, frozen embeddings, trained on ColorSwap, zero-shot eval on Winoground). This is v8Kz's item but answers iv2d's architecture question too. Results pending. |

### v8Kz (4 → target 5)

| Item | Status | Notes |
|---|---|---|
| 1. Robustness of feature-importance measure | ⚠️ MIXED | See G1. Report honestly — this is not a clean win. |
| 2. Each dim's actual contribution to cosine similarity | ✅ DONE | E2 cosine-mass-share is an exact identity, not a proxy. This fully answers the literal request regardless of how G1 nets out. |
| 3. Spectral placement of C_mean/Δ vs. erasure gain | ✅ DONE (null result) | E6: all 5 bridge correlations null (p=0.14–0.74). C_mean concentrates in *low*-σ dims, opposite the paper's narrative. Concede directly: global SVD story and instance-specific erasure are independent mechanisms. |
| 4. Objective vs. architecture vs. matching-function controls | 🔄 RUNNING | E5, in progress. |
| 5. Rescope conclusions | ✅ DONE | See G6. |

### g6iB (4 → target 5)

| Item | Status | Notes |
|---|---|---|
| Q1: deployable inference mechanics | ⬜ TODO | Correct answer is top-k re-ranking erasure (candidate-set mean, not global corpus mean — a global mean is exactly the generic intervention Section 5 already shows fails). Not yet built. Prior draft wrongly proposed "corpus-mean erasure," which contradicts the paper's own Section 5. Do not reuse that language. |
| Q2: 1.7% variance justification | 🔄 PARTIAL | Same as G4 — needs bootstrap before it's presentable. |
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

---

## Open items before rebuttal is submission-ready

1. **E2 bootstrap CIs + `--per-subset`** — access ratio currently a point
   estimate on n=245; also answers BTbD's "2 of 7 splits" complaint directly.
2. **E5 results** (running) — objective/architecture/matching-function
   control for v8Kz + iv2d.
3. **Koishigarina differentiation** — still unresolved; the earlier draft's
   argument didn't hold up. Needs either a real experiment or a more careful
   rewrite.
4. **Top-k re-ranking deployability sketch** for g6iB Q1 — not yet designed;
   do not reuse the "corpus-mean" framing from the earlier draft, it
   contradicts the paper's own negative result in Section 5.
5. **ColorSwap provenance** — GDrive link is fine for rebuttal, needs a
   citable source before camera-ready.
6. Decide how honestly to present G1/G3 — the evidence is genuinely mixed
   (SigLIP-B in particular doesn't cleanly support either "masked" or
   "reversed" depending on which measure you read). Recommend leading with
   what's solid (SO400M reversal, ColorSwap generalization, E2 cosine-mass
   identity) and stating the SigLIP-B ambiguity plainly rather than picking
   whichever number reads best per-paragraph — that's the failure mode from
   the last draft.
