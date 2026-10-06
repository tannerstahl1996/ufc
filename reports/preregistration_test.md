# Pre-registration: final test-set evaluation

Written 2026-10-06, committed BEFORE the test set (fights dated 2024-01-01 onward) is accessed.
The commit hash of this file is recorded in `reports/test_results.md`.

## Primary analysis (decided 2026-10-06 in Phase 1-2, unchanged)

- Model: position-split feature set (`features.FEATURES`), ordered five-outcome round model, ridge 1e-3.
- Fit on train + tune (2017-01-01 to 2023-12-31). Feature scaling from that same data.
- Evaluated once on test (2024-01-01 to 2026-10-03) with the locked metrics in `evaluate.py`:
  card-winner accuracy and exact round-count accuracy, 95% fight-bootstrap CIs.
- Compared against the three one-stat baselines and the most-common-card ceiling on the same fights.
- Also reported: calibration (10 bins, ECE) and exact-card accuracy.
- For continuity: the Phase 2 model (fit on train only) evaluated on test as well.

No change to features, model, or metrics will be made after seeing test results. Any later
change gets reported as post-test exploration, not as the result.

## Secondary hypotheses (formed 2026-10-06 from the confident-miss review of 2017-23)

H1. Judges value a landed leg strike less than a landed head strike. Test: fit the target-split
    variant (`features.FEATURES_TARGET`) on test-period fights only; prediction is that the
    leg/head weight ratio is below 1 with the 95% bootstrap interval excluding 1.

H2. When the model is confidently wrong (>= 80% on a fight where all three judges disagree),
    the favoured fighter's significant strikes lean more on leg strikes than when it is
    confidently right. Test on test-period predictions from the primary model; prediction is a
    positive difference in mean leg share (misses minus hits).

H3 (descriptive, no prediction). The target-split variant's test accuracy versus the primary model.

## Also after the test run

The judge-style analysis (Phase 3, section 2) is re-run on all 2017-2026 fights, since it is
descriptive and was held back only to keep 2024+ unseen until this point.
