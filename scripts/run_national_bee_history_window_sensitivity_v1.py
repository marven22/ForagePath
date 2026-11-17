"""Compare bee seasonal-support models across predeclared training-history windows.

Each window ends in 2022, tunes only its pooling prior on 2023, and evaluates
on 2024--2025.  Bee eligibility is rebuilt from that window's training rows
only, preventing later records from influencing cohort admission.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "snapshots" / "gbif-us-native-bees-2009-2025-v1" / "derived" / "clean_species_occurrences.parquet"
TRAITS = ROOT / "results" / "national_bee_activity_cohorts_v1" / "national_bee_activity_cohorts.csv"
OUT = ROOT / "results" / "national_bee_history_window_sensitivity_v1"
WINDOW_STARTS = (2009, 2011, 2013, 2015)
TRAIN_END = 2022
VALIDATION_YEAR = 2023
TEST_YEARS = (2024, 2025)
WEEKS = np.arange(1, 54)
TAUS = (25.0, 50.0, 100.0, 200.0)


def normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else np.full(53, 1.0 / 53)


def circular_smooth(counts: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = int(math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.r_[counts[-radius:], counts, counts[:radius]]
    return normalize(np.convolve(padded, kernel, mode="valid"))


def spline_curve(counts: np.ndarray) -> np.ndarray:
    weeks = WEEKS.reshape(-1, 1)
    basis = SplineTransformer(n_knots=8, degree=3, extrapolation="periodic", include_bias=True).fit_transform(weeks)
    model = Ridge(alpha=2.0, fit_intercept=True).fit(basis, np.log1p(counts))
    return normalize(np.exp(model.predict(basis)) - 1 + 1e-6)


def counts_by_bee(frame: pd.DataFrame, bees: list[str]) -> dict[str, np.ndarray]:
    grouped = frame.groupby(["bee_key", "iso_week"]).size()
    return {bee: np.array([grouped.get((bee, week), 0) for week in WEEKS], dtype=float) for bee in bees}


def assign_cohort(frame: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    coverage = frame.groupby("bee_key", as_index=False).agg(
        occurrences=("bee_key", "size"),
        years=("year", "nunique"),
        weeks=("iso_week", "nunique"),
        cells=("cell_key", "nunique"),
    )
    cohort = traits.merge(coverage, on="bee_key", how="left")
    for column in ("occurrences", "years", "weeks", "cells"):
        cohort[column] = cohort[column].fillna(0).astype(int)
    individual = (
        cohort.occurrences.ge(100)
        & cohort.years.ge(5)
        & cohort.weeks.ge(20)
        & cohort.cells.ge(10)
    )
    partial = (
        ~individual
        & cohort.occurrences.ge(25)
        & cohort.years.ge(3)
        & cohort.weeks.ge(10)
        & cohort.cells.ge(3)
    )
    cohort["activity_support_cohort"] = np.select(
        [individual, partial, cohort.occurrences.gt(0)],
        ["individual_supported", "partially_pooled", "group_only"],
        default="no_activity_evidence",
    )
    return cohort


def group_profiles(curves: dict[str, np.ndarray], cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in cohort.loc[cohort.activity_support_cohort.eq("individual_supported")].itertuples(index=False):
        grouped.setdefault(f"{row.beegap_family}|{row.sociality}", []).append(curves[row.bee_key])
    return {group: np.mean(values, axis=0) for group, values in grouped.items()}


def evaluate_global(predictions: dict[str, np.ndarray], test_counts: dict[str, np.ndarray], cohort: pd.DataFrame, model: str) -> pd.DataFrame:
    cohort_lookup = cohort.set_index("bee_key")
    rows = []
    for bee, observed in test_counts.items():
        if observed.sum() == 0:
            continue
        probability = predictions[bee]
        weights = observed / observed.sum()
        rows.append({
            "model": model,
            "bee_key": bee,
            "cohort": cohort_lookup.loc[bee, "activity_support_cohort"],
            "test_records": int(observed.sum()),
            "cross_entropy": float(-(weights * np.log(np.maximum(probability, 1e-12))).sum()),
            "top10_mass": float(weights[np.argsort(probability)[-10:]].sum()),
        })
    return pd.DataFrame(rows)


def spatial_evaluate(train: pd.DataFrame, test: pd.DataFrame, global_profiles: dict[str, np.ndarray], cohort: pd.DataFrame) -> pd.DataFrame:
    weekly = train.groupby(["bee_key", "grid_lat_0p5", "grid_lon_0p5", "iso_week"]).size()
    totals = train.groupby(["bee_key", "grid_lat_0p5", "grid_lon_0p5"]).size().to_dict()
    local_profiles: dict[tuple[str, float, float], np.ndarray] = {}
    for (bee, lat, lon), count in totals.items():
        if count >= 10 and bee in global_profiles:
            counts = np.array([weekly.get((bee, lat, lon, week), 0) for week in WEEKS], dtype=float)
            alpha = count / (count + 25.0)
            local_profiles[(bee, lat, lon)] = alpha * circular_smooth(counts) + (1 - alpha) * global_profiles[bee]
    observed = test.groupby(["bee_key", "grid_lat_0p5", "grid_lon_0p5", "iso_week"]).size()
    by_cell: dict[tuple[str, float, float], np.ndarray] = {}
    for (bee, lat, lon, week), count in observed.items():
        if bee in global_profiles:
            by_cell.setdefault((bee, lat, lon), np.zeros(53))[week - 1] = count
    cohort_lookup = cohort.set_index("bee_key")
    rows = []
    for (bee, lat, lon), counts in by_cell.items():
        probability = local_profiles.get((bee, lat, lon), global_profiles[bee])
        weights = counts / counts.sum()
        rows.append({
            "model": "spatial_partial_pooling",
            "bee_key": bee,
            "cohort": cohort_lookup.loc[bee, "activity_support_cohort"],
            "test_records": int(counts.sum()),
            "cross_entropy": float(-(weights * np.log(np.maximum(probability, 1e-12))).sum()),
            "top10_mass": float(weights[np.argsort(probability)[-10:]].sum()),
            "used_local_cell_curve": (bee, lat, lon) in local_profiles,
        })
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("model", as_index=False).apply(
        lambda group: pd.Series({
            "bees_evaluated": group.bee_key.nunique(),
            "test_records": int(group.test_records.sum()),
            "weighted_cross_entropy": np.average(group.cross_entropy, weights=group.test_records),
            "macro_cross_entropy": group.groupby("bee_key").cross_entropy.mean().mean(),
            "weighted_top10_mass": np.average(group.top10_mass, weights=group.test_records),
        }),
        include_groups=False,
    ).reset_index(drop=True)


def run_window(raw: pd.DataFrame, traits: pd.DataFrame, start_year: int, write: bool) -> tuple[pd.DataFrame, dict]:
    train = raw.loc[raw.year.between(start_year, TRAIN_END)].copy()
    validation = raw.loc[raw.year.eq(VALIDATION_YEAR)].copy()
    test = raw.loc[raw.year.between(*TEST_YEARS)].copy()
    cohort = assign_cohort(train, traits)
    model_cohort = cohort.loc[cohort.activity_support_cohort.isin(["individual_supported", "partially_pooled"])].copy()
    bees = model_cohort.bee_key.tolist()
    train = train.loc[train.bee_key.isin(bees)]
    validation = validation.loc[validation.bee_key.isin(bees)]
    test = test.loc[test.bee_key.isin(bees)]
    train_counts = counts_by_bee(train, bees)
    validation_counts = counts_by_bee(validation, bees)
    test_counts = counts_by_bee(test, bees)
    empirical = {bee: normalize(train_counts[bee] + 0.25) for bee in bees}
    smooth = {bee: circular_smooth(train_counts[bee]) for bee in bees}
    spline = {bee: spline_curve(train_counts[bee]) for bee in bees}
    groups = group_profiles(smooth, model_cohort)
    global_group = np.mean(list(groups.values()), axis=0)
    trials = []
    for tau in TAUS:
        pooled = {}
        for row in model_cohort.itertuples(index=False):
            group = groups.get(f"{row.beegap_family}|{row.sociality}", global_group)
            count = train_counts[row.bee_key].sum()
            pooled[row.bee_key] = count / (count + tau) * smooth[row.bee_key] + tau / (count + tau) * group
        scores = evaluate_global(pooled, validation_counts, model_cohort, f"pooled_tau_{int(tau)}")
        sparse = scores.loc[scores.cohort.eq("partially_pooled")]
        trials.append({"tau": tau, "partial_validation_cross_entropy": np.average(sparse.cross_entropy, weights=sparse.test_records)})
    tuning = pd.DataFrame(trials)
    best_tau = float(tuning.sort_values(["partial_validation_cross_entropy", "tau"]).iloc[0].tau)
    pooled = {}
    for row in model_cohort.itertuples(index=False):
        group = groups.get(f"{row.beegap_family}|{row.sociality}", global_group)
        count = train_counts[row.bee_key].sum()
        pooled[row.bee_key] = count / (count + best_tau) * smooth[row.bee_key] + best_tau / (count + best_tau) * group
    scores = pd.concat([
        evaluate_global(empirical, test_counts, model_cohort, "raw_weekly_histogram"),
        evaluate_global(smooth, test_counts, model_cohort, "circular_kernel"),
        evaluate_global(spline, test_counts, model_cohort, "cyclic_spline_ridge"),
        evaluate_global(pooled, test_counts, model_cohort, "trait_family_partial_pooling"),
        spatial_evaluate(train, test, smooth, model_cohort),
    ], ignore_index=True)
    summary = aggregate(scores)
    summary.insert(0, "train_start_year", start_year)
    metadata = {
        "train": f"{start_year}-{TRAIN_END}",
        "validation": str(VALIDATION_YEAR),
        "final_test": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
        "model_eligible_bees_from_training_only": int(len(bees)),
        "individual_supported_bees": int(model_cohort.activity_support_cohort.eq("individual_supported").sum()),
        "partially_pooled_bees": int(model_cohort.activity_support_cohort.eq("partially_pooled").sum()),
        "selected_partial_pooling_prior_records": best_tau,
        "cohort_policy": "Eligibility was derived using only training-window records.",
    }
    if write:
        target = OUT / f"train_{start_year}_{TRAIN_END}"
        target.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(target / "training_window_cohorts.csv", index=False)
        tuning.to_csv(target / "pooling_tau_validation.csv", index=False)
        scores.to_csv(target / "final_test_metrics_by_bee.csv", index=False)
        summary.to_csv(target / "final_test_metrics_summary.csv", index=False)
        (target / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned windows without running or writing results.")
    args = parser.parse_args()
    traits = pd.read_csv(TRAITS)[["bee_key", "beegap_family", "sociality"]].drop_duplicates("bee_key")
    raw = pd.read_parquet(RAW, columns=["bee_key", "event_date", "iso_week", "grid_lat_0p5", "grid_lon_0p5"])
    raw["year"] = pd.to_datetime(raw.event_date, utc=True).dt.year
    raw["cell_key"] = raw.grid_lat_0p5.astype(str) + "|" + raw.grid_lon_0p5.astype(str)
    raw = raw.loc[raw.bee_key.isin(traits.bee_key)].copy()
    if args.dry_run:
        print(json.dumps({
            "windows": [f"{start}-{TRAIN_END}" for start in WINDOW_STARTS],
            "validation": VALIDATION_YEAR,
            "final_test": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
            "final_model_comparisons": len(WINDOW_STARTS) * 5,
            "validation_pooling_candidates": len(WINDOW_STARTS) * len(TAUS),
            "eligibility_policy": "Recomputed from training rows for every window; no future cohort leakage.",
        }, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    combined = []
    manifests = []
    for start in WINDOW_STARTS:
        summary, metadata = run_window(raw, traits, start, write=True)
        combined.append(summary)
        manifests.append(metadata)
    pd.concat(combined, ignore_index=True).to_csv(OUT / "history_window_model_summary.csv", index=False)
    (OUT / "experiment_manifest.json").write_text(json.dumps({"windows": manifests}, indent=2), encoding="utf-8")
    print((OUT / "history_window_model_summary.csv").as_posix())


if __name__ == "__main__":
    main()
