We thank the reviewer for a precise and constructive set of concerns. We address each in turn.

## Weakness 1 / Q2: Generalization beyond Winoground

We address this with a new experiment on **ColorSwap** (Burapacheep et al., 2024), a second genuinely 2×2 compositional benchmark: two images × two captions per group, testing color-attribute binding rather than spatial arrangement. We ran the paper's exact protocol unchanged — identical scoring, identical control suite (single-caption erasure, random-pair erasure, image-side symmetry) — on 300 test pairs across all five models.

| Model | Baseline GS | Mean-Erasure GS | Relative gain | p (one-sided) |
|---|---|---|---|---|
| CLIP-B/32 | 12.0% | 35.7% | 2.97× | <0.0001 |
| CLIP-L/14 | 7.3% | 32.7% | 4.45× | <0.0001 |
| LAION-CLIP-B/32 | 23.3% | 56.3% | 2.41× | <0.0001 |
| SigLIP-B/16 | 30.3% | 61.3% | 2.02× | <0.0001 |
| SigLIP-SO400M | 37.0% | 67.3% | 1.82× | <0.0001 |

As on Winoground, the effect is specific to the matched instance-level direction: `random_pair` erasure shows no significant effect on any model (p = 0.30–0.85), and single-caption-only erasure (`c0_only`/`c1_only`) *reduces* Group Score below baseline for every model — exactly the specificity signature reported in Table 2. This demonstrates the phenomenon is not a Winoground-specific artifact but generalizes across compositional benchmarks testing a different swap axis.

We also formally clarify why this evidence could not come from SugarCrepe/ARO instead: Proposition 1 proves mean-erasure is a mathematical no-op on any 1×2 evaluation, because the shared direction $\hat{C}_\text{mean}$ contributes identically to both candidate scores and cancels in their difference. SugarCrepe, ARO, VALSE, and SugarCrepe++ all use this format, so no 1×2 benchmark could show a mean-erasure gain regardless of whether the underlying account is correct — ColorSwap was the necessary test, and it is the one we ran.

## Weakness 2 / Q1: Reversal beyond a single model

On further scale-controlled testing of this claim, we found the raw correlation reported in Table 1 is more sensitive to sample composition than we initially recognized, and we are not confident it isolates a single factor (scale, objective, or architecture) driving the SigLIP-SO400M reversal specifically. Rather than overstate a causal account we cannot yet fully support, we will revise Table 1 and the associated discussion to present the SigLIP-SO400M result as a single-model observation motivating future work on a controlled model-scale ladder, and remove the implication that scale alone explains it. We thank the reviewer for pressing on exactly this point — the caution is warranted, and we would rather revise the claim than defend it past what our evidence supports.

This does not affect the paper's central empirical contribution: mean-erasure's recovery of compositional Group Score (Table 2, and ColorSwap above) does not depend on the magnitude-reversal mechanism and is independently supported by its own controls across both benchmarks and all five models.

## Weakness 3 / Q3: Downstream / generative VLM consumption

We directly tested this. **LLaVA-1.5-7B** uses a frozen CLIP ViT-L/14 visual encoder — identical to the CLIP-L/14 model in our paper — connected to Vicuna-7B via a lightweight MLP, with no cosine similarity anywhere in its scoring. We evaluate it on the full 400-pair Winoground set as a forced-choice task, scoring each (image, caption) combination via the model's own token-level log-probability of a yes/no response (not greedy generation), and derive Text/Image/Group scores with the identical comparison logic used for the cosine baseline throughout the paper.

| Metric | CLIP-L/14 ITC (cosine) | LLaVA-1.5-7B (same encoder, MLP+LLM) | Δ |
|---|---|---|---|
| Text Score | 28.25% | 46.75% | +18.50pp |
| Image Score | 10.50% | 42.50% | +32.00pp |
| Group Score | 7.50% | 30.00% | +22.50pp |

LLaVA substantially outperforms raw cosine similarity on the *same frozen visual features* across all three metrics. This is direct evidence for the reviewer's implied hypothesis: the compositional signal our diagnostic shows survives the projection is accessible to a non-cosine consumer of those same features. We will add this as a new experiment in the revision and discuss its implications for MLLM architectures explicitly.
