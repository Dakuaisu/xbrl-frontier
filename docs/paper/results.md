# Results

## Numeric fact extraction on the development split

The final Qwen3-1.7B adapter achieved **91.47% micro-F1** on the deduplicated internal development split, compared with 86.60% at the 10,000-instance checkpoint and 88.13% at the 20,000-instance checkpoint. We evaluated all three checkpoints on the same split using the numeric identification task (FinNI) described by [Wang et al., *FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information*, arXiv:2505.20650v5](https://arxiv.org/abs/2505.20650v5). The local scorer matches sets of (normalized numeric value, type) pairs within each instance and pools true positives (TP), false positives (FP), and false negatives (FN) before computing precision, recall, and F1. This split is not the public FinNI test set.

**Table 1.** Development-set micro-precision, recall, and F1 (%). Deduplicated results are primary. Unparseable responses count as empty predictions.

| Checkpoint | Scoring set | Instances | TP | FP | FN | Parse failures | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | Raw | 1,933 | 10,564 | 2,411 | 861 | 3 | 81.42 | 92.46 | 86.59 |
| 20,000 | Raw | 1,933 | 11,135 | 2,712 | 290 | 0 | 80.41 | 97.46 | 88.12 |
| Final | Raw | 1,933 | 10,937 | 1,552 | 488 | 1 | 87.57 | 95.73 | 91.47 |
| 10,000 | Deduplicated | 1,920 | 10,564 | 2,408 | 861 | 3 | 81.44 | 92.46 | 86.60 |
| 20,000 | Deduplicated | 1,920 | 11,135 | 2,710 | 290 | 0 | 80.43 | 97.46 | 88.13 |
| Final | Deduplicated | 1,920 | 10,937 | 1,551 | 488 | 1 | 87.58 | 95.73 | 91.47 |

Recall peaked at the 20,000-instance checkpoint. The final checkpoint's higher F1 came from higher precision: false positives fell from 2,710 to 1,551 on the deduplicated split, while false negatives rose from 290 to 488. Retaining only the first occurrence of each exact context–answer pair removed 13 evaluation rows and did not change the checkpoint ranking. Three outputs at 10,000 instances hit the generation limit; none did at the later checkpoints.

**Figure 1.** [Development-set micro-F1 by checkpoint](./checkpoint-f1.svg). The first two points are nested snapshots from epoch 1, at steps 625 and 1,250. The final adapter was saved at step 4,716 after **two total epochs** over the same 37,721 distinct training instances (2,358 steps per epoch); 1,067 of the 38,788 available instances exceeded the 4,096-token limit and were skipped. The end-of-epoch-1 checkpoint is not scored in this comparison, so the gain cannot be attributed separately to seeing more distinct examples in epoch 1 or revisiting them in epoch 2. These snapshots are not independent training-size experiments.

**Paired uncertainty.** The final checkpoint improved on the 10,000-instance checkpoint by **+4.87 F1 points (95% CI: +3.25 to +7.09)** on the deduplicated split. We computed the percentile interval from 1,000 paired bootstrap resamples of the 17 companies (SEC Central Index Keys, or CIKs). Each replicate drew the same companies with replacement for both checkpoints, pooled their TP, FP, and FN counts, and recomputed the difference in micro-F1. The interval reflects variation across the sampled companies for this training seed, not variation across training runs.

**Score artifacts.** The [checkpoint summary](../../data/scores/d3-10k-20k-final-dev.json) consolidates the saved [10,000-instance](../../data/scores/d3-prefix-10000-dev.json), [20,000-instance](../../data/scores/d3-prefix-20000-dev.json), and [final](../../data/scores/d3-final-dev-pilot.json) reports. The paired interval is computed from the per-instance counts in the 10,000-instance and final reports.
