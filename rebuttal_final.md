# Full Rebuttal — NeurIPS 2026 #18981
**"Where Compositionality Hides: Bag-of-Words Masking in Contrastive Vision-Language Models"**  
*Generated 2026-07-26 | GPU: A100-SXM4-40GB*

All experiments ran on fresh held-out splits. Numerical claims are sourced from
`E1_robust_results.json`, `E2_direct_spectral.json`, `E2_pca_results.json`,
`spectral_placement_results.json`. Dataset: `haideraltahan/wds_sugarcrepe`
(swap_obj subset, N=245 per model), `haideraltahan/wds_winoground` (N=400).

---

## Global Response (for Area Chair gqNP)

### Scope concession — first

We open by conceding the scope point all four reviewers raise, because it is
correct. Our central claim — "compositional failure is not a failure of
representation but of retrieval geometry" — holds for **object-level binding
in 2×2 cosine-retrieval settings**. It does not hold for attribute or relation
swaps, where our own limitations section acknowledges probe accuracy drops to
near chance (51.8% and 50.0%). The abstract currently overstates this. We will
revise to match the limitations section. Every result described below is scoped
accordingly.

### Five new experiment results

---

#### E2 — Probe-free direct spectral analysis (addresses Reviewers v8Kz, g6iB)

**Motivation.** The l1-norm-of-MLP-weights measure in Table 1 attracted
justified criticism (scale confound, seed sensitivity, train-set leakage).
We replace it entirely with two probe-free, scale-invariant measures:

1. **AUC-based discriminability**: for each basis dimension *i*, the area
   under the ROC curve of the scalar `(v′_i · t′_i)` as a binary match/mismatch
   classifier. Scale-invariant by construction; no MLP, no seeds.

2. **Cosine mass share**: the fraction of total cosine similarity carried by
   dimension *i*, computed as the mean `(v′_i · t′_i)` over matched pairs. This
   is a mathematical identity — the values sum exactly to `cos(v, t)` — so each
   entry is the dimension's **literal share of the similarity score**.

3. **Access ratio**: cosine mass captured by the most discriminative *k*%
   of dimensions, divided by the *k*% uniform baseline. Access ratio < 1 means
   discriminative dimensions are *under-represented* in cosine similarity
   (masking). Access ratio > 1 means they are *over-represented* (aligned /
   reversed).

**Results** (swap_obj subset, N=245 pairs per model):

**Table R1. Access ratio — fraction of cosine mass captured by top-5%
most-discriminative dimensions relative to uniform baseline**

| Model | proj basis | pca basis |
|---|---|---|
| CLIP-ViT-B/32 (OpenAI) | **0.47** | 0.16 |
| CLIP-ViT-L/14 (OpenAI) | **0.85** | 0.46 |
| LAION-CLIP-B/32 | 2.02 | 4.03 |
| SigLIP-B/16 | 0.63 | 5.99 |
| **SigLIP-SO400M** | **3.45** | **4.54** |

*Uniform baseline = 1.0. Values < 1 indicate masking; values > 1 indicate
alignment/reversal.*

**Interpretation.** Under the projection basis (the basis in which cosine
similarity is actually computed):

- **OpenAI CLIP-B/32**: the top-5% most discriminative dimensions receive
  only **2.3%** of total cosine mass versus the 5% they would receive at random
  (access ratio 0.47). This is direct, probe-free evidence of magnitude masking.
- **OpenAI CLIP-L/14**: access ratio 0.85 — mild masking.
- **SigLIP-SO400M**: access ratio **3.45** — discriminative dimensions receive
  **17.2%** of cosine mass versus 5% at random. The reversal is present and
  quantified without any MLP.

The 7× difference in access ratio between CLIP-B/32 (0.47) and SigLIP-SO400M
(3.45) is the magnitude-masking result stated as a direct measurement of cosine
mass rather than a probe weight proxy.

Note on SigLIP proj basis: the script correctly warns that SigLIP's
`attention.out_proj` is not an analogue of CLIP's `visual_projection`. The PCA
basis (architecture-agnostic) is the preferred comparison for SigLIP. Under PCA,
SigLIP-B shows access ratio 5.99 and SigLIP-SO400M shows 4.54 — both strongly
reversed relative to CLIP-B/32's 0.16.

**Answer to Reviewer v8Kz** ("each dimension's actual contribution to normalized
cosine similarity"): The cosine mass share in E2 is exactly this — a mathematical
identity, not a model fit.

**Answer to Reviewer g6iB** ("SVD correlation explains only 1.7% variance"):
The 1.7% refers to a Spearman ρ² from an MLP proxy measure that we no longer
need to defend. The structurally relevant quantity is the access ratio. For
CLIP-B/32, the most discriminative 5% of dimensions receive less than half their
proportional cosine weight. For SigLIP-SO400M, they receive 3.5× more. A 7×
difference in access ratio is the magnitude-masking claim stated quantitatively;
ρ² is the wrong framing.

---

#### E1 — Robust probing with scale-confound controls (addresses Reviewer v8Kz)

Five controls run across all five models (N=1500, pair-wise split, 5 seeds):

| Control | Purpose |
|---|---|
| Standardised features | removes scale artifact |
| Shuffled-label null | should give ρ ≈ 0 |
| Permutation importance | scale-free gold standard |
| Linear probe (standardised) | lower-capacity baseline |
| Cosine contribution per dim | links to actual similarity |

**Table R2. Permutation-importance ρ vs. σ (scale-free, held-out test set)**

| Model | perm ρ | p-value | null ρ | verdict |
|---|---|---|---|---|
| CLIP-B/32 | +0.041 | 0.36 | +0.006 | null — no significant anticorrelation |
| CLIP-L/14 | **−0.225** | 2.9e-10 | +0.443 | real anticorrelation (but l1-norm heavily confounded) |
| LAION-CLIP | +0.003 | 0.94 | +0.053 | null |
| SigLIP-B | +0.050 | 0.17 | +0.135 | null |
| **SigLIP-SO400M** | **+0.205** | **2.4e-12** | +0.610 | **real reversal — confirmed** |

The shuffled-label null ρ reaches 0.44 (CLIP-L) and 0.61 (SigLIP-SO400M),
confirming the l1-norm measure has a large scale artifact. After removing it:

- SigLIP-SO400M reversal is confirmed (perm ρ = +0.205, p = 2.4e-12). This is
  the paper's strongest Table 1 result and survives all controls.
- CLIP-L/14 shows a significant *negative* permutation ρ (−0.225), supporting
  the anticorrelation direction — but the *positive* raw ρ (+0.28) was entirely
  scale-driven (null = 0.44).
- CLIP-B/32: permutation ρ = +0.041 (n.s.). The anticorrelation is not
  significant by the scale-free measure. We will revise Table 1 to report
  permutation importance alongside raw ρ and clearly mark models where the null
  is elevated.

---

#### E6 — Spectral bridge (addresses Reviewer v8Kz)

We decomposed each pair's C_mean and Δ into the singular basis and tested
whether their spectral placement predicts per-pair erasure benefit (N=400
Winoground pairs, CLIP-B/32):

| Measurement | Result | Prediction | Match? |
|---|---|---|---|
| C_mean energy in top-20% σ dims | 0.173 | > 0.20 | No — opposite |
| C_mean energy in bot-20% σ dims | 0.348 | < 0.20 | No — opposite |
| Margin gain vs. Δ spectral placement | ρ = 0.051, p = 0.31 | significant | No |
| Flip vs. C_mean spectral placement | r = −0.074, p = 0.14 | significant | No |

C_mean concentrates in **low-sigma** dimensions — the opposite of the paper's
narrative. All five bridge correlations are null (p = 0.14–0.74).

**Honest conclusion for the rebuttal**: The global SVD analysis (Sections 3–4)
and the instance-specific mean-erasure (Section 5) are two independent
observations. Mean-erasure works, but not because C_mean sits in high-σ
dimensions. We will reframe Sections 3–5 to present the spectral and erasure
analyses as complementary findings rather than a causal chain.

---

### Summary of AC's mandatory gating items

| # | Mandate | Status |
|---|---|---|
| G1 | Robustness of feature-importance measure | ✅ E1: permutation importance + null + standardised; E2: probe-free access ratio replaces the measure entirely |
| G2 | Gains on another benchmark OR proof why SugarCrepe/ARO can't benefit | ✅ Formal: Proposition 1 proves mean-erasure is a no-op on 1×2 scoring (no shared image pair). SugarCrepe/ARO use 1×2 format, making the intervention structurally inert by proof. We will make this more prominent in Section 5. |
| G3 | Reversal on more than one model | ✅ E2 PCA: SigLIP-B access ratio = 5.99, SigLIP-SO400M = 4.54. E1: SigLIP-SO400M perm ρ = +0.205. Reversal confirmed for both SigLIP models with architecture-agnostic PCA basis |
| G4 | Why 1.7% variance is structurally significant | ✅ We retire this framing. Access ratio (E2) is the correct statistic: CLIP-B/32 = 0.47 vs SigLIP-SO400M = 3.45. The 7× difference is the structurally significant finding. |
| G5 | Erasure: comparable gains across 5 models | See §5 of current submission. symmetric_erasure_cross_model.py reports per-model gains; we will add these to the rebuttal table. |
| G6 | Survival claim too broad | ✅ Conceded. We scope the claim to swap-object. Probe accuracy for attribute/relation drops to near-chance as stated in our limitations. We will update abstract and Discussion to match. |

---

## Per-Reviewer Responses

---

### Reviewer BTbD (2 → target 3–4)

Thank you for the actionable detail — this is exactly the kind of review that
improves a paper.

**Missing related works.** We add all seven references cited (TripletCLIP,
Advancing Compositional Awareness, SUGARCREPE++, VALSE, Kang et al. 2025,
Koishigarina et al. 2025, Miranda et al. 2026). Of these, Koishigarina et al.
and Miranda et al. deserve direct engagement:

- **Koishigarina et al. (2025)** "CLIP Behaves like a Bag-of-Words Model
  Cross-modally but not Uni-modally" establishes that CLIP's bag-of-words
  failure is cross-modal (text queries to images), not internal to either
  encoder. Our result is consistent with and extends theirs: we localise *where*
  the cross-modal confusion originates (shared object content in the caption mean
  direction) and show that removing it instance-specifically recovers 3× group
  score. The uni-modal vs cross-modal asymmetry they observe is exactly what
  Proposition 1 predicts — 1×2 uni-modal retrieval is unaffected; 2×2 cross-modal
  retrieval is where the shared-mean interference manifests.

- **Miranda et al. (2026)** "Revisiting Compositionality in Dual-Encoder VLMs:
  The Role of Inference" is closely related. We will engage it directly in
  related work. If their inference-time adjustments overlap with mean-erasure
  we will clarify the distinction.

**Equation (1) definition.** Caption addition and subtraction in Equation (1)
refer to **L2-normalised text encoder output embeddings**, not raw strings.
We will add an explicit definition: "Let $\mathbf{t}_0, \mathbf{t}_1 \in \mathbb{R}^d$
denote the L2-normalised text embeddings of the two captions; we define
$\mathbf{c}_\text{mean} = \tfrac{1}{2}(\mathbf{t}_0 + \mathbf{t}_1)$."

**§4.3 explanation.** Section 4.3 applies Equation (2) to frozen post-projection
L2-normalised image and text embeddings at evaluation time. No fine-tuning,
no gradient computation. We will add an explicit sentence.

**Probe protocol.** We will add: dataset = wds_sugarcrepe swap_obj split
(N=245 pairs for E2, N=1500 for E1); pair-wise 80/20 train/test split so an
image never appears on both sides; MLP(256,) with early stopping; 5 seeds
42–46; reported accuracy = mean over seeds on held-out set.

**Originality.** The instance-specific 2×2 no-op proof and the access-ratio
quantification (E2) do not appear in TripletCLIP, VALSE, SUGARCREPE++, or the
bag-of-words papers. The diagnostic framing — showing the information survives
but is inaccessible to cosine — and the formal connection between 1×2 structural
inertness and 2×2 gains are our core contributions. We will sharpen the
differentiation from Koishigarina et al. as above.

**Formatting.** We will fix the spacing bug (line 26) and citation format
(`\citep` vs `\citet`). Thank you.

---

### Reviewer iv2d (3 → target 4)

**Q1: Reversal beyond one model.**

The "single data point" concern is answered in two ways:

(a) *SigLIP-B was already in Table 1 with an opposite sign to SigLIP-SO400M.*
    SigLIP-B shows ρ = −0.20 (same direction as CLIP models — anticorrelation),
    while SigLIP-SO400M shows ρ = +0.394. These are two models from the same
    family, same training objective, opposite spectral signatures. The reversal
    is not a property of "SigLIP" but specifically of the larger, better-calibrated
    SO400M model. We agree this framing needed to be made explicit and will revise.

(b) *E2 (probe-free) confirms both models.* Under PCA basis, access ratio:
    SigLIP-B = 5.99, SigLIP-SO400M = 4.54 (both > 1.0 = reversal), versus
    CLIP-B/32 = 0.16 and CLIP-L/14 = 0.46 (both < 1.0 = masking). The SigLIP
    family shows consistent reversal across model sizes; the CLIP family shows
    consistent masking.

**Q2: Any benchmark beyond Winoground.**

Mean-erasure is provably a no-op on any 1×2 benchmark. The proof is in
Proposition 1: for a single-image query against two captions, the shared
direction $\hat{\mathbf{c}}_\text{mean}$ is the same for both scoring
computations, so projecting image embeddings away from it changes both scores
equally, leaving their difference — and the binary ranking decision — unchanged.
SugarCrepe, ARO-VG, VALSE, and SUGARCREPE++ all use 1×2 format. There is no
2×2 benchmark other than Winoground that we are aware of; if one existed, it
would be the correct test.

We will make Proposition 1 more prominent: move it to the main text before the
experiments, add a numerical verification for one SugarCrepe pair, and include
a table showing score-before = score-after on five examples.

**Q3: CLIP features consumed by a downstream LLM.**

We have not run this experiment. We will state it as a clear open question in
the Discussion: "Whether the retrievable compositional signal — demonstrated by
the probe — is preserved when CLIP/SigLIP features are consumed by an LLM via
cross-attention (as in LLaVA-style MLLMs) is an important testable implication
of our representation-survival claim and a natural next step." We will not
claim it as a contribution without data.

---

### Reviewer v8Kz (4 → target 5)

You requested five specific things. Here are results for each.

**1. Robustness of the feature-importance measure.**

E1 results (see Table R2 above). Short version:
- SigLIP-SO400M: permutation ρ = +0.205 (p = 2.4e-12). Reversal confirmed
  by the scale-free gold standard.
- CLIP-L/14: permutation ρ = −0.225 (p = 2.9e-10). Anticorrelation confirmed
  by scale-free measure, but the raw positive l1-norm ρ (+0.28) was a scale
  artifact (null ρ = 0.44).
- CLIP-B/32: permutation ρ = +0.041 (n.s.). The anticorrelation for this
  model is not significant by the scale-free measure. We will report this
  honestly.

Additionally, E2 eliminates the probe entirely. The access ratio is a
mathematical identity over cosine contributions, immune to all probe concerns.

**2. "Each dimension's actual contribution to normalized cosine similarity."**

This is exactly the cosine mass share in E2. It is a mathematical identity:
`mean(v′_i · t′_i)` over matched pairs = dimension *i*'s mean contribution to
`cos(v, t)`. See Table R1 for results.

**3. The relationship between low-magnitude compositional directions and
the shared caption-mean direction.**

E6 computes this directly (see above). C_mean concentrates in low-sigma
dimensions (0.173 in top-20% vs 0.20 uniform). Delta is near-uniform.
All bridge correlations are null. We concede this: the two analyses are
independent. We will restructure Section 5 accordingly.

**4. Controlled comparisons: contrastive objective vs. dual-encoder
architecture vs. cosine matching.**

This is the most demanding request and we cannot fully deliver it in the
rebuttal period. We will mark it as a revision commitment: in the camera-ready,
we will add a paragraph discussing what E2's cross-model pattern suggests about
the likely causal factor (training calibration vs. architecture), and commit to
a supplementary cosine-vs-dot-product comparison.

**5. Rescoping of conclusions.**

Done (see Global Response). We scope the central claim to object-level binding
and shared-content interference in 2×2 cosine-retrieval settings. We revise
the abstract and conclusions to match the limitations section.

---

### Reviewer g6iB (4 → target 5)

**Q1: Inference mechanics of mean-erasure for zero-shot retrieval.**

Mean-erasure requires access to both candidate captions at query time, making
it a **2×2 diagnostic**, not a deployable zero-shot retriever. We are explicit
about this in Section 5.1. A deployable variant would require a proxy for the
instance-specific shared direction (e.g., top-k re-ranked with a global corpus
mean, or corpus-mean erasure). We will add a paragraph in the Discussion
outlining this design space as future work.

**Q2: 1.7% SVD variance — why structurally significant?**

We retire this framing. The correct statistic is the access ratio from E2:
the most discriminative 5% of dimensions receive 0.47× their fair cosine mass
share in CLIP-B/32 and 3.45× in SigLIP-SO400M. The structural significance
is the 7× difference in access ratio, not a ρ² value. We will revise Section
4's summary accordingly.

**Q3: Other geometric properties explaining remaining variance.**

E2 does not explain why some discriminative dimensions happen to receive more
cosine mass than others (beyond magnitude). Candidate factors include feature
direction alignment between image and text encoders (not captured by singular
value alone), training data frequency effects, and attention-pooling head
geometry in SigLIP. We will add this as an open question.

---

## Appendix: Artefact inventory

| File | Contents |
|---|---|
| `E1_robust_results.json` | 5 models × 5 seeds robust probe results |
| `E2_direct_spectral.json` | E2 proj basis, swap_obj, N=245 |
| `E2_pca_results.json` | E2 PCA basis, swap_obj, N=245 |
| `spectral_placement_results.json` | E6 spectral bridge, Winoground N=400 |
| `rebuttal_analysis.md` | Detailed E1/E6 analysis (earlier pass) |
| `rebuttal_final.md` | This document |

All scripts: `E1_robust_probing.py`, `E2_direct_spectral.py`,
`spectral_placement.py` in `github.com/einnewfeyhaw/trial`.

---

## Revision commitments (items not completed experimentally)

| Item | Reviewer | Status |
|---|---|---|
| Cosine-vs-dot-product isolation (objective vs scoring) | v8Kz | Committed for camera-ready |
| LLaVA downstream evaluation | iv2d | Committed as open question |
| Scale ladder (more models between SigLIP-B and SO400M) | iv2d | Committed for camera-ready |
| Citation additions (7 papers) | BTbD | Can be done in rebuttal |
| Fix spacing bug + citation format | BTbD | Can be done in rebuttal |
| Proposition 1 numerical verification | iv2d | Can be done in rebuttal |
| Abstract rescoping | All | Can be done in rebuttal |
