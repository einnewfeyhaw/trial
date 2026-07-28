We thank the reviewer for identifying two concrete, well-posed questions. We address both directly with new experiments.

## Q1: Deployable zero-shot adaptation

Mean-erasure requires both candidate captions because it computes $C_\text{mean} = \tfrac{1}{2}(C_0+C_1)$ exactly. We adapt this to true zero-shot retrieval by approximating $C_\text{mean}$ from the query's own top-K retrieved captions instead of the foil:

1. Score the query image against the full caption pool with standard cosine similarity.
2. Take the top-K highest-scoring captions.
3. Compute $\tilde{C}_\text{mean}$ as the mean of those K caption embeddings.
4. Project the image away from $\tilde{C}_\text{mean}$ and re-rank.

No foil access is required. The retrieval step itself identifies the shared object content, since captions sharing an image's objects naturally rank highly by cosine similarity even from a large pool.

K=2 is the principled choice, not a post-hoc selection: it directly mirrors the two-caption structure of the original diagnostic ($C_\text{mean}$ over exactly two captions). We report the full sweep to K=32 specifically to demonstrate the predicted degradation as K grows toward a corpus-wide mean — this is validation of the mechanism, not a search for the best-performing K.

| K | Group Score | Δ vs. baseline (9.0%) | p (one-sided, bootstrap) |
|---|---|---|---|
| baseline | 9.0% | — | — |
| **2** | **14.5%** | **+5.5pp** | **0.0012** |
| 4 | 10.5% | +1.5pp | 0.182 |
| 8 | 9.25% | +0.25pp | 0.469 |
| 16 | 8.75% | −0.25pp | 0.611 |
| 32 | 8.25% | −0.75pp | 0.749 |

At K=2, Group Score improves significantly (p=0.0012, surviving Bonferroni correction across the 5 tested K values: 0.0012 × 5 = 0.006 < 0.05). The effect decays toward baseline as K grows, which is exactly what our Section 5 controls predict: a large K approaches a corpus-wide mean, the generic (non-instance-specific) intervention we already show fails. This isn't just consistent with that argument, it traces out the same curve empirically.

In absolute terms, 14.5% remains well below the diagnostic's oracle ceiling (31.0% Group Score with true foil access, Table 2). Framed as a fraction of the recoverable gap, K=2 closes 25% of the distance to the oracle (+5.5pp out of +22.0pp), using only standard retrieval signal and no foil access.

One trade-off worth flagging directly: because each image's erasure direction is computed independently here rather than shared as in the diagnostic, Text Score is not invariant and decreases modestly at every K (−2.5 to −8pp) as Image Score increases substantially. Net Group Score still improves at small K since it requires both to pass on the same pairs, but this is a genuine trade-off, not a free improvement, and we'll state it plainly in the revision.

## Q2: Why the 1.7% is structurally significant

$\rho^2$ measures a global monotonic rank correlation across all 512 dimensions, most of which carry little compositional signal either way — a small global correlation is fully compatible with a large, concentrated effect once the relevant subset is isolated directly. We complement it with a direct, probe-free measurement requiring no MLP, no seeds, and no standardization choice: for each basis dimension, its exact contribution to cosine similarity ($v'_i \cdot t'_i$, which sums exactly to $\cos(v,t)$) and its discriminability ($|\text{AUC}-0.5|$ as a univariate match/mismatch classifier).

On CLIP-B/32, SugarCrepe swap-object (n=245, matching the original probe's evaluation set), the top-5% most discriminative dimensions receive only **0.484× their proportional cosine weight** (95% CI [0.448, 0.530], 10,000-sample bootstrap; P(ratio ≥ 1.0) = 0.0 across every resample). We cross-checked this with a second, separately implemented script computing the same quantity from scratch, which gave 0.47, agreeing within the first result's confidence interval — two independently-coded pipelines converging this tightly is strong evidence this is a real geometric property for object-level swaps, not an artifact of either implementation.

We extended this measurement to all seven SugarCrepe categories to check generalization beyond object-level swaps. One aspect of the pattern is remarkably stable: the top-20% highest-singular-value dimensions carry disproportionately more cosine mass than their discriminative contribution justifies **in every single category, with no exceptions** (2.35–3.07× amplification across all seven, n=245–300 each; full numbers in supplementary code). This indicates the underlying geometric mechanism — high-magnitude dimensions dominating the similarity score beyond their discriminative share — is not specific to object-level swaps, even though our probe-based evidence for its practical consequences is currently strongest there.

A finer-grained measurement (which individual dimensions are most discriminative, rather than the aggregate top-20%/bottom-80% split above) shows more category-dependent structure. For the two categories testing object/attribute *addition* rather than spatial rearrangement, this finer measurement flips: the most discriminative dimensions are well-represented in the cosine score, not suppressed. The reason is visible directly in the captions — true: *"A plate is filled with broccoli and noodles"* vs. false: *"...broccoli, noodles, and carrots"*. The false caption is a strict superset, containing extra content absent from the image; this is a lexical-length asymmetry, not a genuine compositional swap, so cosine can solve it without resolving arrangement at all. This corroborates Hsieh et al. [2023]'s finding that some hard-negative constructions in this benchmark family are solvable via distributional artifacts rather than compositional reasoning — our measurement independently identifies which specific categories are affected, and why.

A complementary view of the swap-object data specifically: the top-20% highest-singular-value dimensions carry 57.3% of total cosine mass but only 20.2% of discriminative signal — a 2.84× over-representation of object-identity content relative to compositional content in the score itself. This is the structural mechanism the original $\rho^2$=1.7% was gesturing at but measuring indirectly; the direct measurement is both stronger and more interpretable, and we will present this probe-free measurement alongside (not in place of) $\rho^2$ in the revision.

## Citation format

Thank you — we will correct this (switching to `\citep{}` where appropriate) in the revision.
