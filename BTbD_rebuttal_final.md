We thank the reviewer for a detailed, specific review. We addressed every point directly, including running a new experiment that goes further than requested. We respond in the order raised.

## Missing related works

We have added all seven references. Kang et al. [5] and Miranda et al. [7] engage closely enough with our framing that we address them substantively below.

## "Survival of information" claims, tested on more datasets as requested

We ran the paper's exact held-out probe protocol (pair-wise 80/20 split, seed 42, identical MLP configuration across all runs) independently on **all seven SugarCrepe categories**, not just the one reported in the submission.

| Category | n (pairs) | Held-out accuracy |
|---|---|---|
| add_obj | 1000 | **80.5%** |
| add_att | 692 | **70.5%** |
| replace_rel | 1000 | 61.0% |
| replace_obj | 1000 | 58.2% |
| swap_att | 666 | 51.5% |
| replace_att | 788 | 50.0% |
| swap_obj | 245 | 50.0% |

The reviewer's concern here was well-founded. The probe reliably distinguishes true from false captions for object-*presence* categories (add_obj, add_att) and, more weakly, for the replace-type categories. For swap_obj, swap_att, and replace_att — the categories closest to genuine compositional rearrangement — held-out accuracy is at chance. This confirms the reviewer's instinct precisely: the survival-of-information claim as stated was narrower than the abstract suggests. We will revise the abstract, introduction, and Table 1 caption to state the claim at this precise scope, and report the full seven-category table above in the revision rather than a single number.

While this correction appropriately narrows the scope of our probe-based claim, our core empirical findings remain untouched: the retrieval-geometry mechanism, the formal 1×2/2×2 result, and mean-erasure's recovery of Winoground Group Score (Table 2: 9.0%→31.0%, replicated across five models) all rest on independent evidence — the erasure result, in particular, is a closed-form projection with no probe or classifier involved anywhere in its computation, and is unaffected by the probe's category-specific behavior.

## Relating Line 40 to [5, 6, 7]

These three papers, together with ours, reflect a genuine and useful convergence: multiple groups are now independently concluding that CLIP's compositional failures are better explained by *how matching happens* than by what is encoded. We differentiate as follows:

- **Kang et al. [5]** prove an impossibility result for a *specific* representational construction — embeddings built by additive superposition of object-level parts provably cannot disambiguate attribute bindings. This is a conditional result about an idealized encoding scheme, not an empirical claim about trained CLIP models, and it does not address singular-value magnitude. Their fix (DCSM) operates on dense per-patch, per-token similarity maps — a different granularity of intervention than ours, which operates entirely on the single global pooled embedding.
- **Koishigarina et al. [6]** demonstrate empirically that CLIP's text encoder linearly encodes attribute-object binding that cross-modal matching fails to use, and propose a fixed, globally-trained linear correction (LABCLIP). We provide the geometric mechanism they do not: binding-relevant dimensions are systematically discounted by cosine similarity because cosine implicitly weights by singular-value magnitude (Section 3-4). We also find that a global correction in the same structural class as LABCLIP does not transfer across compositional axes in our setting — motivating our instance-specific approach as a genuine alternative, not a redundant one.
- **Miranda et al. [7]** is the closest in thesis (representation vs. inference), already cited in our submission. Their fix requires fine-grained region-segment alignment via a trained transformer operating on frozen patch and token embeddings. Ours requires no patch-level access at all — a single projection of the final pooled embedding, with a formal proof (Proposition 1) of exactly when it can and cannot succeed, which has no counterpart in [5], [6], or [7].

We will add a paragraph making this three-way differentiation explicit in Related Work, along with the sentence at Line 40 connecting our question directly to [5,6].

## Term definitions

**Equation (1).** We will state explicitly: "Let $C_0, C_1 \in \mathbb{R}^{d_\text{out}}$ denote the L2-normalized text embeddings — not the raw captions or token sequences — of the two candidate captions. These vectors are produced entirely after tokenization and after the text encoder and projection; addition and subtraction in Eq. (1) are standard vector operations in this final embedding space, never on tokens or strings. Define $C_\text{mean} = \tfrac{1}{2}(C_0+C_1)$ and $\Delta = \tfrac{1}{2}(C_0-C_1)$." This resolves the ambiguity and is consistent with Equation (2) and Proposition 1's normalization assumption throughout.

**Section 4.3.** We will add: "All mean-erasure operations are applied at evaluation time to the frozen, post-projection image embeddings (Eq. 2); no model weights are modified and no training occurs — the projection is a deterministic geometric transform of the encoder's output embeddings."

**Probe protocol.** For the SugarCrepe headline number, we will state explicitly: pair-wise 80/20 held-out split (both samples of a pair on the same side), random seed 42, `MLPClassifier(hidden_layer_sizes=(512,256), max_iter=1000, early_stopping=True)` with Adam defaults, reported on held-out test pairs only. We will confirm the ARO figure is reported under an equivalently rigorous held-out protocol in the revision.

## Validating the instance-specific claim on another 2×2 evaluation

We ran the paper's full protocol, unchanged, on **ColorSwap** (Burapacheep et al., 2024) — a second, independently constructed 2×2 benchmark testing color-attribute binding rather than spatial arrangement — across all five models with the complete control suite.

| Model | Baseline GS | Mean-Erasure GS | Relative gain | p |
|---|---|---|---|---|
| CLIP-B/32 | 12.0% | 35.7% | 2.97× | <0.0001 |
| CLIP-L/14 | 7.3% | 32.7% | 4.45× | <0.0001 |
| LAION-CLIP-B/32 | 23.3% | 56.3% | 2.41× | <0.0001 |
| SigLIP-B/16 | 30.3% | 61.3% | 2.02× | <0.0001 |
| SigLIP-SO400M | 37.0% | 67.3% | 1.82× | <0.0001 |

Random-pair erasure shows no effect on any model (p = 0.30–0.85), and single-caption-only erasure reduces Group Score below baseline for every model — the same specificity signature as Table 2. Following the reviewer's suggestion to validate beyond Winoground, this confirms the intervention's necessity is not an artifact of one benchmark's construction: only the matched, instance-specific direction recovers performance, on an independently built benchmark testing a different compositional axis. This also speaks directly to the reviewer's broader point: the goal is understanding compositionality, not just succeeding on Winoground. With formal support (Proposition 1) and validation across two independent benchmarks, we establish that instance-specificity is a necessary condition for recovery — a structural finding we believe should guide the design of future methods in this space.

## Minor formatting

Fixed: `\citet{}` → `\citep{}` throughout for parenthetical citations (e.g., Line 21), and the Line 26 spacing/quotation bug.

---

We believe this response — a more precisely scoped probe claim tested across all seven categories; a new benchmark validating the core intervention; and a substantive three-way differentiation from concurrent work — directly addresses every weakness raised.
