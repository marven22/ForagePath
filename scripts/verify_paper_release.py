"""Verify compact frozen artifacts supporting the ForagePath manuscript.

This script is intentionally an artifact-integrity check rather than a model
trainer. It makes the quantities reported in the paper inspectable without
redistributing bulk third-party occurrence and phenology records.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 5e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, found {actual}")


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise AssertionError(f"{label} is missing columns: {sorted(missing)}")


def main() -> int:
    plant_candidates = pd.read_csv(ROOT / "data/reproducibility/candidate_native_flowering_plants.csv")
    bee_cohorts = pd.read_csv(ROOT / "data/reproducibility/national_bee_activity_cohorts.csv")
    strict_edges = pd.read_csv(ROOT / "data/reproducibility/strict_admitted_plant_bee_edges.csv")
    with (ROOT / "data/reproducibility/strict_network_summary.json").open(encoding="utf-8") as handle:
        strict_summary = json.load(handle)

    assert len(plant_candidates) == 1138, len(plant_candidates)
    assert len(bee_cohorts) == 436, len(bee_cohorts)
    assert len(strict_edges) == 5236, len(strict_edges)
    assert strict_summary["strict_admitted_pairs_A_or_B"] == 5236
    assert strict_summary["plants_with_strict_admitted_edge"] == 571
    assert strict_summary["bees_with_strict_admitted_edge"] == 436

    flowering = pd.read_csv(
        ROOT / "results/flowering_history_window_sensitivity_v1/train_2015_2022/final_test_metrics.csv"
    )
    require_columns(flowering, {"model", "rows", "checks", "weighted_brier"}, "flowering metrics")
    best_flowering = flowering.loc[flowering.model.eq("hierarchical_plant_calibrated_ensemble")].iloc[0]
    assert int(best_flowering.rows) == 203_012
    assert int(best_flowering.checks) == 587_240
    close(float(best_flowering.weighted_brier), 0.039829)

    neon = pd.read_csv(ROOT / "results/neon_external_flowering_evaluation_v2/metrics.csv")
    best_neon = neon.loc[neon.model.eq("hierarchical_ensemble")].iloc[0]
    assert int(best_neon.rows) == 42_294
    assert int(best_neon.checks) == 225_436
    close(float(best_neon.weighted_brier), 0.023255)

    bee = pd.read_csv(
        ROOT / "results/national_bee_history_window_sensitivity_v1/train_2015_2022/final_test_metrics_summary.csv"
    )
    spatial = bee.loc[bee.model.eq("spatial_partial_pooling")].iloc[0]
    assert int(spatial.bees_evaluated) == 281
    assert int(spatial.test_records) == 163_616
    close(float(spatial.weighted_top10_mass), 0.624780)

    mixture = pd.read_csv(ROOT / "results/phase2_v16_appalachian_robust_validation/robust_validation_summary.csv")
    held_catboost = mixture.loc[
        mixture.validation_design.eq("leave_one_study_out")
        & mixture.mixture_size.eq(3)
        & mixture.method.eq("catboost_supported_links")
    ].iloc[0]
    held_strict = mixture.loc[
        mixture.validation_design.eq("leave_one_study_out")
        & mixture.mixture_size.eq(3)
        & mixture.method.eq("strict_documented_training_links")
    ].iloc[0]
    close(float(held_catboost.bee_week_coverage), 0.352409)
    close(float(held_strict.bee_week_coverage), 0.270984)

    print("ForagePath paper-release verification passed.")
    print("  candidate plants: 1,138")
    print("  target native bees: 436")
    print("  strict Grade A/B edges: 5,236")
    print("  flowering future rows: 203,012")
    print("  NEON external rows: 42,294")
    print("  future bee records: 163,616")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
