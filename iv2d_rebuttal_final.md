We thank the reviewer for a precise and constructive set of concerns. We address each in turn.

## Weakness 1 / Q2: Generalization beyond Winoground

We agree this is the central open question the paper needed to answer, and we address it with a new experiment on **ColorSwap** (Burapacheep et al., 2024), a second genuinely 2×2 compositional benchmark: two images × two captions per group, testing color-attribute binding rather than spatial arrangement. We ran the paper's exact protocol unchanged — identical scoring, identical control suite (single-caption erasure, random-pair erasure, image-side symmetry) — on 300 test pairs across all five models.

| Model | Baseline GS | Mean-Erasure GS | Relative gain | p (one-sided) |
|---|---|---|---|---|
| CLIP-B/32 | 12.0% | 35.7% | 2.97× | <0.0001 |
| CLIP-L/14 | 7.3% | 32.7% | 4.45× | <0.0001 |
| LAION-CLIP-B/32 | 23.3% | 56.3% | 2.41× | <0.0001 |
| SigLIP-B/16 | 30.3% | 61.3% | 2.02× | <0.0001 |
| SigLIP-SO400M | 37.0% | 67.3% | 1.82× | <0.0001 |

As on Winoground, the effect is specific to the matched instance-level direction: `random_pair` erasure shows no significant effect on any model (p = 0.30–0.85), and single-caption-only erasure (`c0_only`/`c1_only`) *reduces* Group Score below baseline for every model — exactly the specificity signature reported in Table 2. This demonstrates the phenomenon is not a Winoground-specific artifact but generalizes across compositional benchmarks testing a different swap axis (color-attribute binding vs. spatial arrangement).

We also formally clarify why this evidence could not come from SugarCrepe/ARO instead: Proposition 1 proves mean-erasure is a mathematical no-op on any 1×2 evaluation, because the shared direction $\hat{C}_\text{mean}$ contributes identically to both candidate scores and cancels in their difference. This is not a limitation of our method but a structural property of the 1×2 comparison itself — SugarCrepe, ARO, VALSE, and SugarCrepe++ all use this format, so no 1×2 benchmark could ever show a mean-erasure gain regardless of whether the underlying geometric account is correct. ColorSwap was therefore the necessary test, and it is the one we ran.

## Weakness 2 / Q1: Reversal beyond a single model

We first note the premise is not quite "one data point": Table 1 already reports SigLIP-**base** as magnitude-masked ($\rho=-0.223$, $-0.303$) and SigLIP-**SO400M** as magnitude-aligned ($\rho=+0.394$, $+0.420$) — two checkpoints in the same architecture family, same sigmoid training objective, opposite signs. This is suggestive that scale, not the objective itself, may be the relevant factor, since the objective is held fixed across the two.

**[EXPERIMENT RUNNING — fill in once results return]** We are extending this to a scale ladder within and across model families to test this more decisively: [OpenCLIP LAION checkpoints at multiple scales / additional SigLIP checkpoints / regression of $\rho$ on {params, data size, objective}]. We report here only what has been verified; we will not claim results we have not run.

We also note, in the interest of full transparency, that our own robustness checks (conducted in response to Reviewer v8Kz) complicate a simple "scale explains it" story: under scale-free permutation importance, SigLIP-base's signal is not significant in either direction (null, not masked), and under an architecture-agnostic PCA basis, SigLIP-base's access ratio is highly basis-dependent (0.63 under the projection basis vs. 5.99 under PCA). We do not think this undermines the SO400M reversal itself, which is robust across every measure we have tried — but we do not think the current evidence supports a fully resolved causal account of *why* it reverses, and we will state this as an open question in the revision rather than overclaim.

## Weakness 3 / Q3: Downstream / generative VLM consumption

**[NOT YET RUN — decide before submitting: commit as future work, or attempt]** This is a well-posed and testable extension: if compositional information survives the projection but is inaccessible to cosine similarity specifically, a downstream LLM consuming the same frozen features via cross-attention (e.g., LLaVA-style architectures) should not be bound by the same limitation, and might recover compositional signal that zero-shot cosine retrieval cannot. We agree this is a high-value follow-up. [If run: results here.] [If not run in time: state as a concrete, well-specified direction for future work — e.g., "evaluate a frozen-CLIP-backbone MLLM (LLaVA-1.5) on Winoground's 400 pairs as a binary caption-choice task, testing whether the LLM's cross-attention pathway extracts the compositional signal our diagnostic shows survives the projection."]
