We thank the reviewer for pressing on both points — they identify a real gap we hadn't fully closed.

## On the 80.3% → 50% discrepancy

The 80.3% figure in the submission was computed by a script that draws examples from the dataset without filtering by category. That sample is dominated by `add_obj` (and, to a lesser extent, `add_att`), not `swap_obj` as the table caption states — it contains no genuine `swap_obj` examples. When we filter explicitly to `swap_obj`, held-out accuracy is 50%, chance level. The 80.3% number should not have been reported as a swap-object result; we withdraw it and will replace it with the corrected seven-category table in the revision.

## On the length-bias concern for add_obj/add_att

We tested this directly rather than relying on the literature alone. For each of the seven categories, we computed the accuracy of a trivial heuristic — predict the true caption is whichever of the two has fewer words, using no image, no model, no embeddings:

| Category | Length-heuristic accuracy | MLP probe accuracy | Same-length ties |
|---|---|---|---|
| add_obj | **98.8%** | 80.5% | 20/1000 |
| add_att | **99.1%** | 70.5% | 8/692 |
| replace_rel | 54.2% | 61.0% | 505/1000 |
| replace_obj | 55.0% | 58.2% | 775/1000 |
| swap_att | 51.1% | 51.5% | 569/666 |
| replace_att | 51.0% | 50.0% | 660/788 |
| swap_obj | 52.4% | 50.0% | 221/245 |

Two things follow from this, and we want to be precise rather than overstate either.

First, `add_obj`/`add_att` are confirmed as confounded, and more severely than the reviewer's citation alone would suggest: a heuristic using zero visual or semantic information achieves 98.8%/99.1%, exceeding our probe's own accuracy. The probe sits *below* this ceiling, so we cannot say it learned pure word-counting — but the categories are dominated by a confound strong enough that probe accuracy there is uninformative regardless of exactly what the probe learned. We withdraw `add_obj`/`add_att` as evidence for compositional information survival.

Second, the other five categories — including the two closest to genuine compositional rearrangement, `swap_obj` and `swap_att` — are not length-confounded: the heuristic sits at chance, and the large majority of pairs (up to 775/1000) have *identical* word counts, since swaps and single-word replacements preserve caption length by construction. This rules out length bias as an explanation for those categories one way or the other; it does not by itself establish that genuine signal is present.

This means the honest conclusion is stronger than our previous response stated: **held-out MLP probe accuracy does not reliably support the claim that compositional information survives the projection, for any of the seven categories** — `add_obj`/`add_att` are confounded, and the rest are at or near chance for the probe regardless of confound status. We will revise the abstract and introduction to remove probe accuracy as evidence for this claim entirely, rather than merely narrowing its scope.

We do, however, have a second and different kind of evidence for `swap_obj` specifically, and we think it directly speaks to the reviewer's concern rather than around it. Separately from any trained classifier, we measured — as an exact identity, not a fitted model — each embedding dimension's literal contribution to cosine similarity and its discriminability (AUC) between matched and mismatched pairs, on `swap_obj` (n=245, held out) [B]. Because `swap_obj` negatives are the same words as the positive, merely reordered, there is no length or content-count signal available to exploit; any discriminability this measure detects can only come from genuine word-order sensitivity in the embeddings. The result: the most discriminative 5% of dimensions receive only 0.484× their proportional share of cosine similarity (95% CI [0.448, 0.530], cross-validated by a second, independently implemented script at 0.47). For this to be measurable and this precise, there must be real, non-random discriminative signal present for `swap_obj` — the question the length-heuristic control cannot bear on either way. This is the evidence we now rely on for the survival claim on `swap_obj`, in place of probe accuracy.

This changes what the paper's survival-of-information claim rests on overall. It now rests on: (1) the identity-based discriminability measure above, for `swap_obj` specifically, and (2) mean-erasure's empirical recovery (Table 2: 9.0%→31.0%, replicated on ColorSwap and across five models) — a closed-form geometric projection requiring no classifier or training. Neither depends on probe accuracy, and neither is affected by the length-bias confound. We will state this explicitly in the revision and retain the probe results only as a documented negative finding.

## On the necessity of instance-specific intervention

This is a fair challenge, and we should state our claim more precisely than the submission does. What we actually demonstrate is narrower than "any method must be instance-specific": we show that a *specific class* of interventions — fixed, globally-applied corrections to the embedding geometry or matching function (SVD steering, CAVs, Mahalanobis reweighting, and, in response to another reviewer, a learned bilinear matching function) — cannot recover the shared bag-of-words confound that mean-erasure removes, because that confound ($C_\text{mean}$) is defined per caption-pair by construction. A fixed transformation cannot track a quantity that varies pair to pair; this is closer to a structural observation about *this specific confound* than a claim about all possible future methods.

We agree this does not establish that every method improving compositionality must construct negatives at inference time. Approaches operating on a different representational level — e.g., dense patch-token alignment (Kang et al. [5]) or region-segment structural matching (Miranda et al. [7]) — may address compositionality without needing per-instance negative examples at all, since they are not attempting to correct this specific global-vs-instance-specific confound in the first place. We will revise Section 5's framing and the sentence at lines 94-95 to state the claim at this precise scope: fixed corrections to the bag-of-words confound specifically require instance-specific information, not that all compositional methods do. We will also note this as an open question for future work rather than a settled implication of our results.

On the "cat chasing a dog" example specifically: our diagnostic does not offer a deployment recipe for this query as stated, and we do not want to overstate its practical bearing. We separately tested an approximation using the query's own top-K retrieved captions in place of the foil (in response to another reviewer), which recovers roughly a quarter of the gap to the diagnostic's ceiling without requiring negative examples to be constructed by hand — but this remains a partial, diagnostic-adjacent result, not a general solution, and we will describe it with this caveat rather than as evidence that instance-specific correction is practically necessary.

[A] Udandarao et al. A Good CREPE Needs More Than Just Sugar: Investigating Biases in Compositional Vision-Language Benchmarks. 2025.
