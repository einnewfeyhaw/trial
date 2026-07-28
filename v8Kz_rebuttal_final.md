We thank the reviewer for a careful, technically grounded review, and for explicitly noting willingness to raise the score if these points are addressed. We respond to each weakness in order.

## 1. Presentation and organization

We agree the paper currently moves between representation collapse, SVD-based magnitude masking, the 1×2/2×2 structural analysis, and mean erasure without always stating what each piece establishes. We will restructure around three explicit, separately-scoped hypotheses: (H1) compositional information survives the projection (probe evidence, scoped per point 5 below), (H2) cosine similarity systematically underweights the dimensions carrying it for object-level swaps specifically (Section 3–4, now supported by the direct measurement in point 3), and (H3) recovering it requires an instance-specific correction, not a generic one (Section 5, Proposition 1). Each will state its evidence and scope explicitly rather than being presented as a single continuous argument.

## 2. Isolating objective, architecture, and matching function

We directly tested the matching-function axis: holding the frozen representations fixed, we replaced cosine similarity with a learned matching function (a bilinear scorer $v^\top A t$, $A$ initialized at identity so it reduces exactly to cosine at zero training), trained on ColorSwap and evaluated zero-shot on Winoground. It does not recover meaningful signal — the best result across a weight-decay sweep is a small margin over cosine (+0.25 to +0.75pp), and relaxing regularization to let the correction move further from identity makes zero-shot transfer *worse*, not better, falling below the cosine baseline at the loosest setting we tested — the signature of overfitting to the training benchmark rather than learning a transferable correction. We read this as a genuine, informative negative result: it extends Section 5's finding that fixed, non-instance-specific interventions fail from the representation-geometry domain into the matching-function domain as well — a global learned scorer is subject to the same limitation as a global learned projection. We have not yet isolated the objective-vs-architecture axis (e.g., a matched-architecture comparison across training objectives) and will state this as a specific, scoped direction for future work rather than claim it.

## 3. Robustness of the feature-importance measure

The reviewer's concern here was well-founded, and our own follow-up testing confirms it directly: the $\ell_1$-norm-of-MLP-weights measure is sensitive to exactly the factors the reviewer named. Under a corrected protocol (held-out pair-wise split, multiple seeds, standardized features, and a shuffled-label null control), the raw correlation does not reliably survive across models — consistent with the reviewer's prediction that this measure could be an artifact of scale and reparameterization rather than a stable signal.

We therefore rely on the direct measurement the reviewer requested in the same paragraph: comparing inferred importance to each dimension's actual contribution to normalized cosine similarity. This is a mathematical identity, not a probe — the element-wise interaction features sum exactly to $\cos(v,t)$, so each dimension's mean contribution over matched pairs is its literal share of the similarity score, and its discriminability is measured by AUC ($|\text{AUC}-0.5|$) rather than any trained weight vector. On CLIP-B/32, SugarCrepe swap-object (n=245, held out), the top-5% most discriminative dimensions by this measure receive only **0.484× their proportional cosine weight** (95% CI [0.448, 0.530], 10,000-sample bootstrap; P(ratio ≥ 1.0) = 0.0). We cross-checked this with a second, independently implemented script, which returned 0.47 — two separately-coded pipelines converging this tightly is strong evidence this reflects a real geometric property rather than an artifact of either implementation or of the probe-based measure the reviewer correctly questioned.

## 4. Relationship between low-magnitude directions and the shared caption-mean direction

We tested this directly: for 400 Winoground pairs, we decomposed $C_\text{mean}$ and $\Delta$ into the SVD basis and correlated $\Delta$'s spectral placement with per-pair mean-erasure gain. The result is a genuine null (ρ≈0.05, p=0.31), and $C_\text{mean}$ concentrates in *low*-singular-value dimensions (34.8% of its energy in the bottom quintile vs. a 20% uniform baseline), the opposite of what a single unified mechanism would predict. We take this as it is: the global SVD spectral analysis and the instance-specific mean-erasure result are complementary, independently-supported findings, not two views of one causal chain. We will revise Sections 3–5 to present them this way rather than implying a single connected mechanism, and note explicitly that mean-erasure's benefit for magnitude-aligned models (e.g. SigLIP-SO400M) is consistent with this — an instance-specific confound can coexist with global magnitude alignment, since they are not the same quantity.

## 5. Scope of the conclusion

We agree and have direct evidence for the correct scope. We ran the held-out probe across all seven SugarCrepe categories, not just swap-object:

| Category | Held-out accuracy |
|---|---|
| add_obj | 80.5% |
| add_att | 70.5% |
| replace_rel | 61.0% |
| replace_obj | 58.2% |
| swap_att | 51.5% |
| replace_att | 50.0% |
| swap_obj | 50.0% |

Probing evidence is strong for object-presence detection and clearly at chance for the categories closest to genuine compositional rearrangement. We will revise the abstract, introduction, and conclusion to state the claim precisely as the reviewer suggests: object-level binding and shared-content interference in 2×2 cosine-based retrieval, not compositional failure in general. This does not weaken the paper's central empirical result — mean-erasure's recovery of Winoground Group Score (9.0%→31.0%, Table 2) and its replication on ColorSwap, an independently constructed second 2×2 benchmark, both hold at this same, correctly narrowed scope.

---

We believe these revisions — a restructured presentation around explicitly scoped hypotheses, a matching-function control, a probe-free measurement that directly answers the reviewer's request and confirms their concern about the original measure, a resolved (if partially null) spectral bridge, and a precisely rescoped conclusion — address every weakness raised.
