# Standalone Insertion Quality Encoder V1

## Outcome

**Acceptance gate: FAIL**. TAVLA integration is therefore not allowed yet.

- Encoder pooled Macro-F1: `0.551`
- Misaligned pooled F1: `0.328`
- Aligned-progress pooled F1: `0.774`
- Pooled AUROC: `0.486`
- Quality-evaluable folds: `7/10` (both classes present in `3/10`)
- Terminal pooled F1: `0.620`
- Handcrafted Logistic pooled Macro-F1: `0.589`
- Encoder gain over Logistic: `-0.037`
- Boundary median error: `8.00` frames
- Paired Real/Sim probability median difference: `0.129`
- Active paired windows: `27` across `4/10` folds
- Latent domain classifier accuracy (diagnostic): `0.867`

## Interpretation

The encoder does not meet the insertion-quality objective. Its pooled AUROC is close to chance, it predicts aligned-progress for most misaligned frames, and it underperforms both the handcrafted and full-11D Logistic baselines. The no-motion-anchor ablation is also better than the full encoder, so the current future-motion auxiliary loss is not helping cross-trajectory quality recognition.

Quality probabilities outside high-confidence misaligned/aligned intervals are diagnostic only: `free` and `uncertain` were intentionally masked. The current entropy-derived confidence is not calibrated and must not be interpreted as physical certainty. Visual inspection shows overconfident aligned predictions in these masked phases, which is another reason not to expose this 4D output to TAVLA.

The phase audit quantifies this collapse: `p_aligned >= 0.5` on 76.6% of free frames, 81.4% of uncertain frames, and 74.4% of manually labeled misaligned frames. Median entropy-confidence is 0.933/0.949/0.970 respectively, so the confidence output is overconfident rather than uncertainty-aware. See `eval_outputs/insertion_quality_v1/phase_output_summary.csv`.

Only 7/10 held-out trajectories contain any quality interval and only 3/10 contain both classes. Active cross-domain pairs exist in only 4/10 folds. These coverage limitations make the apparent paired-probability result insufficient evidence of Sim/Real generalization.

## Ablations

- Force direction only pooled Macro-F1: `0.501`
- Magnitude/trend only pooled Macro-F1: `0.548`
- Full 11D Logistic pooled Macro-F1: `0.611`
- Encoder without motion-anchor pooled Macro-F1: `0.625`

## Leakage controls

Each fold holds out one complete paired trajectory. Weak-label thresholds, activity gates, empirical CDFs and motion normalization are fit only on train trajectories. The held-out Real and Sim replay are never split across train/test. Pair consistency is applied to output probabilities only; latent equality is not enforced.
Every fold stores `preprocess.json`, train-only empirical CDF arrays, and `window_manifest.csv` with label sources/masks under `eval_outputs/insertion_quality_v1/folds/`.

## Structured output

The deployment-facing output is `[p_misaligned, p_aligned_progress, p_terminal, confidence]`. Confidence is `1 - normalized binary entropy`. The 16D latent remains an audit value and is not a default TAVLA observation.
Because the acceptance gate failed, no deployment/frozen checkpoint is produced. Fold checkpoints are retained for audit only, and TAVLA training logic remains unchanged.
