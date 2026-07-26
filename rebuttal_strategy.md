# Rebuttal Strategy — "Where Compositionality Hides" (Submission 18981)

## Scoring situation

| Reviewer | Rating | Posture |
|---|---|---|
| v8Kz | 4 (borderline accept) | Explicitly offered to raise if weaknesses addressed |
| g6iB | 4 (borderline accept) | Confidence 2; two concrete questions |
| iv2d | 3 (borderline reject) | Three specific, answerable questions |
| BTbD | 2 (reject), **originality 1** | The critical problem |

BTbD's originality=1 is almost certainly driven by Koishigarina et al. 2025
("CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally"),
which sounds like our thesis from the title alone. Tone will not fix this;
we need a differentiating result (see §7).

---

## RULE FOR THIS REBUTTAL

**Claim only what has been run.** Reviewers punish unfulfilled promises harder
than acknowledged limitations. Every numeric claim below is marked
[MEASURED] or [PENDING]. Nothing marked [PENDING] goes into submitted text
until the number exists.

---

## 0. The correction we must volunteer (highest priority)

**Issue.** The scripts producing Table 1 —
`feature_importance.py`, `multi_model_correlation.py`,
`multi_model_aro_correlation.py`, `strong_model_correlation.py`,
`aro_svd_correlation.py` — all call `clf.score(X, y)` on training data.
Table 1's entire "Probe Acc." column is **train accuracy**, and the rho values
come from an unsplit, unregularized fit. Only `corrected_probe_eval.py`
(the 80.3% / 77.0% figures) uses a proper 80/20 split.

**This code is in the submitted supplementary zip.** We must not claim
"our split was 80/20" for Table 1. If a reviewer opens the zip, that assertion
becomes misrepresentation and justifies BTbD's reject.

**Action.** Run E1 (`E1_robust_probing.py`), report corrected held-out numbers,
and state the correction plainly. Volunteered, this reads as rigor. Discovered,
it is fatal.

**Second, larger issue found while fixing E1 — the scale confound.**
Features are `(v.U)_i * (t.U)_i`. Since `v' = vU = xV(Sigma)`, coordinate *i* is
scaled by sigma_i, so low-sigma dims have intrinsically small variance. A probe
needs *larger* weights on small-variance inputs, so an l1-weight-norm importance
is mechanically anti-correlated with sigma **even under zero compositional
signal**. E1 now tests this directly via standardized features, held-out
permutation importance, and a shuffled-label null.

Two possible worlds:
- **rho survives standardization + permutation importance, null is flat**
  -> magnitude masking is real and now far better evidenced than in the
  submitted version. Table 1 gets strictly stronger.
- **rho collapses when standardized and the shuffled-label null reproduces the
  original negative value** -> the Table 1 finding was an artifact. The
  mean-erasure result (Sections 4-5) is untouched, since it is a closed-form
  projection with no probe anywhere in it. In that world we retract the
  magnitude-masking framing and rebuild the paper around retrieval geometry
  and instance-specific erasure.

Do not draft §2 or §3 language below until E1 returns.

---

## 1. Generalizability beyond Winoground (iv2d, BTbD, AC — binding demand)

**Critique.** Only Winoground's 2x2 format; no gains on SugarCrepe/ARO.

**Do NOT** simply re-assert Proposition 1. The AC read it and asked anyway;
restating it as a "full resolution" will read as non-responsive.

**Response.** Two parts.
(a) Restate Prop 1 briefly as the *reason* 1x2 cannot show gains — this is a
    structural property of the benchmark, not a limitation of the method.
(b) [PENDING] Demonstrate gains on a **second genuine 2x2 benchmark**.
    Candidates: **ColorSwap** (Burapacheep et al. 2024, purpose-built
    Winoground-format, ~2k pairs, fastest to wire up — our `eval_pair()` works
    unmodified), **Cola** (Ray et al., NeurIPS 2023), **EqBen** (Wang et al.,
    ICCV 2023). Run all five models so results slot into Table 4's format.

This is the single most important new experiment. Without it the AC's mandate
is unmet.

---

## 2. The 1.7% variance question (g6iB, AC)

**Critique.** AC: "must quantitatively justify why this 1.7% is structurally
significant, rather than just statistical noise."

**Do NOT** answer with prose about "systematic global rank shift." The AC asked
for a quantity; the current draft answer contains none.

**Response.** [PENDING — from E1] Replace global Spearman r^2, which is the
wrong statistic to defend, with:
- **Enrichment**: mean singular-value percentile of top-20 vs bottom-20
  importance dims, with a permutation test.
- **Cosine mass**: fraction of total cosine-similarity mass carried by
  compositionally-important dims. If top-20% importance dims contribute ~3% of
  the cosine sum while object-identity dims carry the bulk, *that* is the
  structural claim — far more damning than rho = -0.13. E1 computes this as
  `cosine_contribution`.
- **Probe-guided causal test**: amplify dims the *probe* flags as important
  (rather than bottom-20% by magnitude as in the current SVD steering) and
  measure Winoground. Separates "importance" from "magnitude."

---

## 3. Bridging global SVD and instance-specific C_mean (v8Kz, AC)

**Critique.** SVD analysis is global; erasure is pair-specific; nothing connects
them. v8Kz asks whether per-example gains depend on spectral placement.

**Do NOT** assert that "C_mean inherently projects heavily onto the top singular
dimensions." That is the hypothesis `spectral_placement.py` tests. If it comes
back negative the argument inverts.

**Response.** [PENDING — from E6 `spectral_placement.py`] Report measured
placement of C_mean and Delta, and the correlation between Delta's spectral
placement and per-pair erasure gain. If null, say so: the two mechanisms are
independent, which is itself an honest and publishable refinement.

---

## 4. Real-world applicability (g6iB, iv2d, AC)

**Critique.** Mean-erasure needs both captions; non-deployable.

**Response.**
(a) Clarify inference mechanics explicitly, as the AC demands: erasure is a
    closed-form projection applied to frozen post-projection embeddings at
    evaluation time. No training, no fine-tuning, no gradient.
(b) [PENDING] **Top-k re-ranking erasure** — the strong answer. For a query
    image, retrieve top-k captions by ordinary cosine, compute C_mean over
    *those k retrieved candidates*, erase, re-rank. No foil access; the
    candidate set is something every retrieval system already has.

    **Critical design note:** do NOT use the whole-gallery mean. That is a fixed
    direction = the generic intervention Section 5 already proves fails.
    Instance-specificity requires k small and query-dependent. Sweep
    k in {2,4,8,16,32,64} on Winoground + COCO-5k/Flickr30k (R@1/5/10). The
    decay point empirically confirms Section 5.

    **Cite and baseline against hubness correction** (QB-Norm, CSLS, inverted
    softmax) — this is adjacent work and a reviewer will raise it if we don't.

    **Risk:** if the top-k mean is too stable across queries it collapses into
    the generic regime and shows no gain. Still publishable as
    "instance-specificity requires k < K*", but do not promise success before
    seeing the curve.

---

## 5. The SigLIP reversal (iv2d, AC)

**Critique.** One data point; causal attribution unidentifiable.

**Response.**
(a) [MEASURED] Immediate correction: it is **already not one data point**.
    Table 1 shows SigLIP-**base** at rho = -0.223 / -0.303 (masked) and
    SigLIP-**SO400M** at rho = +0.394 / +0.420 (aligned). Same objective, same
    family, opposite signs — so the sigmoid loss alone does not cause the
    reversal, and scale is the prime suspect. State this in the first
    paragraph of the response to iv2d.
(b) [PENDING] Make it decisive with a **scale ladder**: OpenCLIP LAION
    (B/32, B/16, L/14, H/14, g/14, bigG/14 — same objective and data, varying
    scale) isolates scale; SigLIP base/large/so400m at matched resolution
    isolates scale within the sigmoid objective; CLIP-B/32 OpenAI vs LAION
    isolates data. Regress rho on {params, data size, objective, patch size,
    resolution}. Money plot: rho vs Winoground/SugarCrepe accuracy across all
    checkpoints.

**Do NOT** call two points "strong evidence" for an emergent-scale claim —
that invites a rebuttal-of-the-rebuttal from iv2d specifically.

---

## 6. Objective vs architecture vs scoring rule (v8Kz, AC)

**Critique.** Cannot tell whether the behavior comes from the contrastive
objective, the dual-encoder architecture, or cosine matching.

**Response.** [PENDING] Two controlled tests:
(a) **BLIP ITC vs ITM heads** — same vision encoder, cosine head vs
    cross-attention head, on Winoground. If ITM >> ITC, the culprit is provably
    the matching function, not the representation.
(b) **Learned scorer on frozen CLIP embeddings** — train a small bilinear/MLP
    scorer on frozen embeddings (train on ColorSwap/Cola, evaluate held-out on
    Winoground). If a non-cosine scorer on *unmodified* embeddings recovers
    Winoground, that proves "information survives, cosine cannot reach it"
    with no erasure at all. This is the most direct possible test of our
    central thesis — run it first.

---

## 7. Related work and the originality problem (BTbD)

**Critique.** Missing [1]-[7]; claims not connected to prior work.

**Response.**
(a) Add all seven citations. Non-negotiable, zero cost.
(b) [PENDING] **Differentiate from Koishigarina et al. 2025** — this is what
    drives originality=1. Their finding: CLIP is BoW-like cross-modally but not
    uni-modally. Ours: BoW dominance is localized to singular-value magnitude,
    and instance-specific erasure recovers it. Make it concrete — run our
    uni-modal (image-image, text-text) similarity analysis alongside
    cross-modal and show our magnitude account *explains* their observation
    (uni-modal comparisons do not involve the shared cross-modal C_mean
    direction). If our framework predicts their result, we generalize them
    rather than duplicate them.
(c) Distinguish from Kang et al. 2025 (patch-token alignment over frozen
    encoders) and Miranda et al. 2026 (inference-time structural reasoning).

---

## 8. Scope of claims (v8Kz, BTbD, AC)

**Critique.** Conclusions broader than evidence. Probe is strong on object swaps
(80.3%) but near chance on Swap Attribute (51.8%) and Replace Relation (50.0%) —
which the limitations section itself concedes, contradicting the abstract.

**Response.** Concede and rescope. This costs nothing and buys credibility with
three reviewers at once.
- [PENDING] Run the held-out probe on **all seven** SugarCrepe splits plus
  SugarCrepe++ and VALSE. Report honestly, including near-chance results.
- Rewrite abstract/intro/conclusion to scope claims to **object-level binding
  and shared-content interference in 2x2 cosine retrieval**, not "compositional
  failure" in general. The paper's conclusion already gestures at this; make it
  the headline framing.

---

## 9. Presentation fixes (v8Kz, BTbD — free points)

- **Eq. 1 ambiguity** (BTbD): state explicitly that addition/subtraction operate
  on final L2-normalized **text embeddings**, not tokens or strings.
- **Section 4.3**: spell out that Eq. 2 is applied to frozen post-projection
  embeddings at eval time.
- **Probe details** (BTbD's direct question): specify train/test protocol,
  whether the full split is used, and learning rates per model/dataset.
- **Citation format**: switch `\citet{}` -> `\citep{}` throughout.
- **Spacing bug** line 26: `like"a mug`.
- Reorganize around a smaller set of hypotheses with explicit scope per
  v8Kz's first weakness.

---

## Suggested run order (by ROI per GPU-hour)

1. **E1** — `E1_robust_probing.py`. Hours. Gates everything in §0/§2.
2. **E6** — `spectral_placement.py`. Minutes on cached embeddings. Gates §3.
3. **E2** — ColorSwap 2x2. The AC's binding demand.
4. **E5b** — learned scorer on frozen embeddings. Most direct thesis test.
5. **E3** — top-k re-ranking sweep.
6. **E4** — scale ladder.
7. **E9** — all seven SugarCrepe splits.
8. **E5a** — BLIP ITC vs ITM.
9. **E10** — uni-modal vs cross-modal (Koishigarina differentiation).
10. **E8** — LLaVA on Winoground.

Promise in the rebuttal only what has finished. List the rest as revision
commitments.
