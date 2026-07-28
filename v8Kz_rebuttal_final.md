We thank the reviewer for a careful, technically grounded review, and for explicitly noting willingness to raise the score if these points are addressed. We respond to each weakness with new analysis, in order.

## 1. Presentation and organization

We will restructure the paper around three explicit, separately-scoped hypotheses, each with its own evidence and boundary stated up front rather than folded into one continuous narrative: **(H1)** compositional information survives the projection, evidenced by held-out probing at the scope established in point 5; **(H2)** cosine similarity systematically underweights the dimensions carrying it, for object-level swaps specifically, evidenced by the direct measurement in point 3; and **(H3)** recovering it requires an instance-specific correction rather than a generic one, evidenced by Section 5 and, now, point 2 below. This directly targets the reviewer's diagnosis: each hypothesis will carry its own scope, so a reader can evaluate each claim against its actual evidence rather than one blended argument.

## 2. Isolating objective, architecture, and matching function

We ran a direct, controlled test of the matching-function axis the reviewer specified: holding the frozen representations fixed, we replaced cosine similarity with a learned bilinear scorer ($v^\top A t$, $A$ initialized at identity), trained on ColorSwap and evaluated zero-shot on Winoground. The result **independently confirms our central claim through an entirely different mechanism than Section 5 already tests**. Section 5 shows a fixed, globally-learned *projection* cannot substitute for instance-specific mean-erasure; this experiment shows a fixed, globally-learned *matching function* fails the same way — the best setting across a weight-decay sweep gives only a marginal edge over cosine (+0.25 to +0.75pp), and relaxing the correction to move further from identity makes it actively worse, the signature of overfitting to one training distribution rather than learning a transferable rule. Two structurally different global corrections — one on the representation, one on the scoring function — hit the same wall for the same underlying reason: the confound they'd need to correct is instance-specific, not global. This is a second, independent line of evidence for the paper's core structural claim, not a separate negative finding. We have not yet isolated the objective-vs-architecture axis specifically (e.g., matched-architecture comparison across training objectives) and will state this precisely as a scoped direction for future work.

## 3. Robustness of the feature-importance measure

We adopted the reviewer's own suggested test — comparing inferred importance directly against each dimension's actual contribution to normalized cosine similarity — and it gives a cleaner, stronger result than the original probe-based measure. This is a mathematical identity, not a fitted model: the element-wise interaction features sum exactly to $\cos(v,t)$, so each dimension's mean contribution over matched pairs is its literal share of the similarity score, with discriminability measured by AUC ($|\text{AUC}-0.5|$) rather than any trained weight vector — no MLP, no seeds, no reparameterization sensitivity to control for. On CLIP-B/32, SugarCrepe swap-object (n=245, held out), the top-5% most discriminative dimensions by this measure receive only **0.484× their proportional cosine weight** (95% CI [0.448, 0.530], 10,000-sample bootstrap; P(ratio ≥ 1.0) = 0.0). We cross-checked this with a second, independently implemented pipeline, which returned 0.47 — two separately-coded measurements converging this tightly is strong evidence for a real geometric property. This measure also resolves the specific concern the reviewer raised about the original $\ell_1$-norm measure: under a corrected protocol (held-out split, multiple seeds, standardization, shuffled-label null), that measure's raw correlation does not reliably survive across models, exactly the reparameterization and scale sensitivity the reviewer anticipated. We now rely on the identity-based measure as the primary evidence for this finding going forward.

## 4. Relationship between low-magnitude directions and the shared caption-mean direction

We decomposed $C_\text{mean}$ and $\Delta$ into the SVD basis for 400 Winoground pairs and tested whether spectral placement predicts per-pair mean-erasure gain directly. It does not (ρ≈0.05, p=0.31), and $C_\text{mean}$ itself concentrates in *low*-singular-value dimensions (34.8% of its energy in the bottom quintile vs. a 20% uniform baseline) rather than the high-magnitude space a single unified mechanism would predict. We read this as establishing that the paper documents **two independent geometric phenomena degrading compositional retrieval, not one**: the global spectral pattern in Sections 3–4, and the instance-specific shared-content confound mean-erasure removes in Section 5. Each stands on its own evidence — the erasure result requires nothing about spectral placement to hold, and vice versa. We will revise Sections 3–5 to present them explicitly as two separate, independently-supported contributions rather than implying a single causal chain between them, which is a more precise and defensible account of what the evidence actually shows.

## 5. Scope of the conclusion

We ran the held-out probe across all seven SugarCrepe categories, not only swap-object, giving the precise scope the reviewer's critique calls for:

| Category | Held-out accuracy |
|---|---|
| add_obj | 80.5% |
| add_att | 70.5% |
| replace_rel | 61.0% |
| replace_obj | 58.2% |
| swap_att | 51.5% |
| replace_att | 50.0% |
| swap_obj | 50.0% |

The evidence is strong exactly where the paper's contribution is strongest — object-level binding — and at chance for categories further from it, which is precisely the boundary the reviewer asked us to state explicitly. We will revise the abstract, introduction, and conclusion accordingly: object-level binding and shared-content interference in 2×2 cosine-based retrieval. Critically, the paper's central empirical result is unaffected by this narrowing — mean-erasure's recovery of Winoground Group Score (9.0%→31.0%, Table 2), replicated on ColorSwap, an independently constructed second 2×2 benchmark, holds at exactly this scope, with its own controls, regardless of probe-based evidence elsewhere in the paper.

---

Taken together: a restructured presentation with explicit per-hypothesis scope, a second independent confirmation of the paper's core instance-specificity claim via the matching-function axis, a probe-free measurement that is both more rigorous and more precisely targeted than the reviewer's original concern, a clarified two-mechanism account in place of an implied single one, and a conclusion scoped exactly to what the evidence supports. The paper's central empirical contribution — mean-erasure's recovery of compositional Group Score, replicated across two independent benchmarks and five models — is untouched by any of these revisions.
