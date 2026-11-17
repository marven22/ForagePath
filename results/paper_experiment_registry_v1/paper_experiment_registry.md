# Paper Experiment Registry v1

This registry is the manuscript source of truth for experimental cohort definitions and claim boundaries. It was generated from the cited result artifacts on 2026-09-14.

| ID | Paper role | Status | Cohort and boundary |
|---|---|---|---|
| F1 | Internal flowering development benchmark | READY WITH QUALIFIER | 203,012 plant-week rows and 587,240 observation checks |
| F2 | Cross-source external flowering evaluation | READY WITH QUALIFIER | 42,294 weekly rows from 42 sites and 342 plants |
| B1 | Future-year native bee seasonal activity | READY | 281 training-eligible species appearing later and 163,616 scored later occurrence records. Training eligible pool equals 288 species. |
| L1 | Plant-bee link ranking | READY WITH PROTOCOL NOTE | All four output groups contain the same future_year, held_event, unseen_bee, unseen_pair, unseen_plant core splits. Example held-event split contains 8,669 candidate rows and 359 events. |
| M1 | Decision-level plant-mixture backtest | READY WITH SCOPE LIMIT | 23 site keys. Rolling-origin years 2002, 2003, 2009, 2010, 2012, 2014, 2015, 2016, 2017. Leave-one-study-out studies 11. Mixture sizes 3, 5, and 7. |

## Required before final manuscript submission

- Use only the five common V16 splits when directly comparing all plant-bee link model families.
- Preserve the stated qualifiers for the internal flowering benchmark, partial-covariate NEON evaluation, and retrospective mixture outcome.
- Do not use the older 348-species bee confidence-interval file. The verified 281-species file listed below is authoritative.

## Artifact locations

- **F1**  results\flowering_history_window_sensitivity_v1\experiment_manifest.json | results\flowering_history_window_sensitivity_v1\train_2015_2022\final_test_metrics.csv | results\paper_uncertainty_intervals_v1\flowering_bootstrap_ci.csv
- **F2**  results\neon_external_flowering_evaluation_v2\summary.json | results\neon_external_flowering_evaluation_v2\metrics.csv | results\paper_uncertainty_intervals_v1\neon_site_bootstrap_ci_v1.csv
- **B1**  results\national_bee_history_window_sensitivity_v1\train_2015_2022\summary.json | results\national_bee_history_window_sensitivity_v1\train_2015_2022\final_test_metrics_summary.csv | results\national_bee_history_window_sensitivity_v1\train_2015_2022\final_test_metrics_by_bee.csv | results\paper_uncertainty_intervals_v1\bee_activity_verified_281_bootstrap_ci.csv
- **L1**  results\phase1_v16_appalachian_all\baselines\metrics.csv | results\phase1_v16_appalachian_all\two_tower\factorized_ai_metrics.csv | results\phase1_v16_appalachian_all\compact_gnns\metrics.csv | results\phase1_v16_appalachian_all\rich_gnns\metrics.csv | results\paper_uncertainty_intervals_v1\phase1_link_bootstrap_ci.csv
- **M1**  results\phase2_v16_appalachian_robust_validation\summary.json | results\phase2_v16_appalachian_robust_validation\robust_validation_metrics.csv | results\phase2_v16_appalachian_robust_validation\paired_bootstrap_comparisons.csv
