# Rebuttal Strategy for "Where Compositionality Hides"

## Internal Summary

This work addresses the central challenge of why contrastive vision-language models fail at compositional reasoning. It shifts the blame from "representation collapse" to "retrieval geometry." The claims of the paper are supported by formal mathematical proofs regarding benchmark structures and rigorous diagnostic evaluations using the mean-erasure technique. Overall, I find this submission to be structurally sound and fully defensible against the reviewers' critiques. 

Below is the concrete, section-by-section rebuttal strategy answering the Area Chair's meta-review mandates. 

## Actionable Rebuttal Points

### 1. Generalizability & The 1x2 vs 2x2 Evaluation Formats (iv2d, BTbD)
**Critique:** The evaluation is only applied to Winoground's 2x2 structure. No gains demonstrated for SugarCrepe/ARO. The rebuttal must establish gains or prove why 1x2 benchmarks nullify the geometric properties.
**Rebuttal Response:** We will politely point out that **Proposition 1 in Section 4.2 formally proves exactly this**. The theorem mathematically proves that projecting away from $C_{mean}$ is an algebraic no-op on 1x2 evaluations because $C_{mean}$ and the syntactic difference $\Delta$ are strictly orthogonal in the joint text space. As a result, mean-erasure *cannot* change rankings on SugarCrepe or ARO, because the bag-of-words component cancels itself out during the single-image dot-product comparison. Thus, the 2x2 Winoground format is structurally unique in exposing this bag-of-words geometry, making our theoretical proof a full resolution to the AC's request.

### 2. SVD Correlation Magnitude & Significance (g6iB)
**Critique:** The SVD correlation explains merely 1.7% of the variance. Justify why this is structurally significant.
**Rebuttal Response:** A Spearman $\rho$ of -0.13 (approx. 1.7% explained variance) across $d=512$ dimensions represents a systematic and global rank shift rather than isolated noise. In a high-dimensional dot-product regime, cosine similarity is dominated by the top few principal components. When the compositional signal is systematically pushed out of the top $\sim 20\%$ of components and diffused across the bottom $80\%$, the *cumulative* magnitude of the bag-of-words features overwhelms the compositional signal. We will clarify that this small but reliable rank correlation leads to massive misalignment when aggregated in the zero-shot sum.

### 3. Gap between Low-Magnitude SVD and Cmean Placement (v8Kz)
**Critique:** Test whether the per-example erasure gains depend on spectral placement of $C_{mean}$ and relational difference.
**Rebuttal Response:** We will clarify that the global SVD masking and the local $C_{mean}$ erasure are two sides of the same geometric coin. Because standard contrastive training forces object-presence features (which $C_{mean}$ captures) to dominate the representation, $C_{mean}$ inherently projects heavily onto the top singular dimensions. Erasing $C_{mean}$ is functionally equivalent to aggressively attenuating the highest-magnitude object components on a per-instance basis, exposing the low-magnitude $\Delta$ (the relational difference) previously drowned out by the object signals. 

### 4. Real-world Applicability & Inference Mechanics (g6iB, iv2d)
**Critique:** Applicability of mean erasure to practical zero-shot retrieval scenarios.
**Rebuttal Response:** As explicitly stated in Section 6, mean-erasure is a *diagnostic tool* to prove the representation vs. retrieval geometry hypothesis, not a deployable zero-shot method. However, to bridge the gap to real-world applicability, we will propose an immediate extension: at query time, one could use an image captioner to estimate $C_{mean}$ from a single image and project the visual embedding away from this proxy, eliminating the need for the foil caption. Furthermore, the insight holds massive value for *pre-training objectives*: it proves that future VLMs don't need entirely new architectures, but rather losses that elevate structural features out of the low-magnitude noise floor (as seen in SigLIP-SO400M).

### 5. SigLIP Reversal Claim (iv2d)
**Critique:** Only a single data point to support the reversal claim.
**Rebuttal Response:** We will emphasize that SigLIP-base also exhibits masking ($\rho = -0.223$), establishing that the Sigmoid loss alone does not cause the reversal. The positive alignment in SigLIP-SO400M ($\rho = +0.394$) is an emergent property of scale (both data and parameter count). This provides strong evidence that scaling contrastive models naturally shifts spatial/compositional features into high-magnitude dimensions, solving the masking problem geometrically.

### 6. Missing Related Works & Connection to LLMs (BTbD, iv2d)
**Critique:** Include related works [1]-[7] and explain connection to non-cosine MLLMs.
**Rebuttal Response:** We will thank Reviewer BTbD and integrate the missing citations, specifically distinguishing our instance-specific geometric analysis from Kang et al.'s token alignment. For MLLMs, we will note that our findings perfectly align with modern practice: LLMs attached to frozen CLIP encoders via MLPs or cross-attention (e.g., LLaVA) *do not* rely on global cosine similarity, allowing their non-linear networks to access the exact same low-magnitude compositional features that our MLP probe found!

### 7. Clarification on Probing & Definitions (v8Kz, BTbD)
**Critique:** Robustness of probe and clarification of Eq. 1.
**Rebuttal Response:** We will clarify that Eq. 1 addition operates on the final $L_2$-normalized text embeddings, not tokens. For probing, our split was 80/20 train/test, and we will offer to run a simple linear probe on the element-wise interaction $(v_i \odot t_i)$ to satisfy v8Kz's robustness check.