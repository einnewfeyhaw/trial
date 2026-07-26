# Where Compositionality Hides — NeurIPS 2026 Code Release

Minimal, reproducible code release accompanying the NeurIPS 2026 paper *"Where Compositionality Hides: Bag-of-Words Magnitude Masking in Contrastive Vision-Language Models."*

This repository contains only the code needed to reproduce every numerical result in the paper. No cached results are included — running the scripts below regenerates every number reported in the paper.

---

## Headline result

| Finding | Reproduces with |
|---|---|
| Compositional info **survives** the projection (MLP probe: 80.3% on SugarCrepe, 77.0% on ARO) | [01_probing_tests/corrected_probe_eval.py](01_probing_tests/corrected_probe_eval.py) |
| Compositional features sit in **low-magnitude** SVD dims (CLIP: ρ = −0.129 on SugarCrepe, −0.132 on ARO, p < 0.005) | [02_feature_importance/feature_importance.py](02_feature_importance/feature_importance.py), [02_feature_importance/aro_svd_correlation.py](02_feature_importance/aro_svd_correlation.py) |
| Pattern **reverses** in compositionally strong models (SigLIP-SO400M: ρ = +0.394 / +0.420) | [02_feature_importance/strong_model_correlation.py](02_feature_importance/strong_model_correlation.py) |
| Mean-erasure on Winoground (Group Score 9.0% → **31.0%**) | [04_concept_erasure/concept_erasure_eval.py](04_concept_erasure/concept_erasure_eval.py) |
| Generic linear interventions all fail null controls (p > 0.2) | [03_metric_replacements/](03_metric_replacements/) |

---

## Section ↔ script map

| Paper section | Script |
|---|---|
| §3 MLP probe accuracy | [01_probing_tests/corrected_probe_eval.py](01_probing_tests/corrected_probe_eval.py) |
| §4.1 Magnitude masking on CLIP-B/32 (SugarCrepe) | [02_feature_importance/feature_importance.py](02_feature_importance/feature_importance.py) |
| §4.1 Magnitude masking on CLIP-B/32 (ARO) | [02_feature_importance/aro_svd_correlation.py](02_feature_importance/aro_svd_correlation.py) |
| §4.2 Cross-model SVD correlation (SugarCrepe) | [02_feature_importance/multi_model_correlation.py](02_feature_importance/multi_model_correlation.py) |
| §4.2 Cross-model SVD correlation (ARO) | [02_feature_importance/multi_model_aro_correlation.py](02_feature_importance/multi_model_aro_correlation.py) |
| §4.2 SigLIP-SO400M reversal | [02_feature_importance/strong_model_correlation.py](02_feature_importance/strong_model_correlation.py) |
| §5 Mean-erasure on Winoground (CLIP-B/32) | [04_concept_erasure/concept_erasure_eval.py](04_concept_erasure/concept_erasure_eval.py) |
| §5 Random-caption control | [04_concept_erasure/erasure_random_control.py](04_concept_erasure/erasure_random_control.py) |
| §5 Per-category + single-caption ablations | [04_concept_erasure/winoground_ablations.py](04_concept_erasure/winoground_ablations.py) |
| §5 Cross-model erasure + symmetric (image-side) erasure | [04_concept_erasure/symmetric_erasure_cross_model.py](04_concept_erasure/symmetric_erasure_cross_model.py) |
| §6 Generic interventions (SVD / Mahalanobis / random subspace / random pre-proj) | [03_metric_replacements/unified_interventions.py](03_metric_replacements/unified_interventions.py) |
| §6 CAV intervention | [03_metric_replacements/cav_steering.py](03_metric_replacements/cav_steering.py) |
| §6 Mahalanobis (standalone run) | [03_metric_replacements/mahalanobis_eval.py](03_metric_replacements/mahalanobis_eval.py) |
| Appendix Sensitivity to erasure strength | [04_concept_erasure/erasure_sensitivity.py](04_concept_erasure/erasure_sensitivity.py) |
| Appendix Per-pair pass overlap (text-side vs image-side) | [04_concept_erasure/erasure_pass_overlap.py](04_concept_erasure/erasure_pass_overlap.py) |

---

## Reproducing every result from scratch

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch ≥ 2.0. A CUDA-capable GPU is recommended for the cross-model sweeps (SigLIP-SO400M is the slowest model); everything runs on CPU but is correspondingly slower.

### 2. Authenticate with HuggingFace

Winoground is gated; request access at <https://huggingface.co/datasets/facebook/winoground> and then export your access token:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

(The other datasets — SugarCrepe and ARO — are public and require no token.)

### 3. Create the output directory

A few scripts write JSON output to `./05_results/`; create it once before running:

```bash
mkdir -p 05_results
```

Other scripts write their JSON to the script's working directory; running everything from the repo root is the recommended workflow.

### 4. Run the scripts

Each script prints its summary to stdout and saves a JSON to its working directory or to `05_results/`. Run from the repo root:

```bash
# §3 — probe accuracy
python 01_probing_tests/corrected_probe_eval.py

# §4 — magnitude masking (single model + cross-model)
python 02_feature_importance/feature_importance.py
python 02_feature_importance/aro_svd_correlation.py
python 02_feature_importance/multi_model_correlation.py
python 02_feature_importance/multi_model_aro_correlation.py
python 02_feature_importance/strong_model_correlation.py

# §5 — mean-erasure + controls + per-category + sensitivity
python 04_concept_erasure/concept_erasure_eval.py
python 04_concept_erasure/erasure_random_control.py
python 04_concept_erasure/winoground_ablations.py
python 04_concept_erasure/erasure_sensitivity.py
python 04_concept_erasure/symmetric_erasure_cross_model.py

# Appendix — per-pair pass overlap
python 04_concept_erasure/erasure_pass_overlap.py

# §6 — generic intervention null results
python 03_metric_replacements/unified_interventions.py
python 03_metric_replacements/cav_steering.py
python 03_metric_replacements/mahalanobis_eval.py
```

Full pipeline takes roughly **80–90 minutes** on a single GPU; the cross-model sweeps (lines starting with `multi_model_*` and `symmetric_erasure_cross_model`) account for most of the runtime.

---

## Repository layout

```
.
├── 01_probing_tests/        # MLP probe — info survives the projection
├── 02_feature_importance/   # SVD feature importance vs. singular value
├── 03_metric_replacements/  # Generic linear interventions (negative result)
└── 04_concept_erasure/      # Diagnostic mean-erasure on Winoground
```

---

## Datasets

- [SugarCrepe](https://huggingface.co/datasets/HuggingFaceM4/SugarCrepe) — public
- [ARO (VG-Relations)](https://github.com/mertyg/vision-language-models-are-bows) — public
- [Winoground](https://huggingface.co/datasets/facebook/winoground) — **gated**, requires `HF_TOKEN`

## Models

- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `google/siglip-base-patch16-224`
- `google/siglip-so400m-patch14-384`

---

## Notes for reviewers

- All MLP probes use sklearn's `MLPClassifier(random_state=42)`; some scripts also set `torch.manual_seed(42)`. Re-running the scripts should reproduce the headline numbers within ±0.5 percentage points.
- HuggingFace models and datasets are not pinned to specific revisions. We do not expect the published weights or splits to change, but for strict long-tail reproducibility, pin via `revision=<sha>` arguments to `from_pretrained()` and `load_dataset()`.

---

## Citation

The paper is currently under review at NeurIPS 2026. A BibTeX entry will be added once the proceedings are public.
