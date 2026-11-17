"""Extend validation grids when a selected pooling parameter sits at an edge.

The study design is unchanged.  Both checks use records through 2022 for
fitting, 2023 for selection, and retain 2024--2025 as untouched final test
years.  No test-year score is consulted to select a candidate parameter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_flowering_v3_ecoregion_hierarchical_ensemble_v1 as flowering
import run_national_bee_history_window_sensitivity_v1 as bees


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pooling_boundary_sensitivity_v1"
FLOWERING_TAUS = (
    0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50,
    2.00, 2.50, 3.00, 4.00, 5.00, 7.50, 10.00, 15.00, 20.00,
    50.00, 100.00,
)
BEE_TAUS = (5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0, 200.0)


def flowering_sensitivity() -> dict[str, object]:
    flowering.seed()
    raw = pd.read_parquet(flowering.DATA)
    data = flowering.enhanced_weather(raw.loc[raw.weather_available].copy())
    full_train = data.loc[data.data_split.eq("train")].copy()
    train = flowering.screening_training_frame(full_train)
    validation = data.loc[data.data_split.eq("validation")].copy().reset_index(drop=True)
    test = data.loc[data.data_split.eq("test")].copy().reset_index(drop=True)
    cat_validation, cat_test = flowering.cat_predict(train, validation, test)
    ft_validation, ft_test = flowering.ft_predict(train, validation, test)
    weights = np.linspace(0.0, 1.0, 101)
    ft_weight = min(
        weights,
        key=lambda value: flowering.brier(validation, value * ft_validation + (1.0 - value) * cat_validation),
    )
    blended_validation = ft_weight * ft_validation + (1.0 - ft_weight) * cat_validation
    blended_test = ft_weight * ft_test + (1.0 - ft_weight) * cat_test
    x = np.r_[blended_validation, blended_validation]
    y = np.r_[np.ones(len(blended_validation)), np.zeros(len(blended_validation))]
    sample_weight = np.r_[validation.positive_checks.to_numpy(), validation.negative_checks.to_numpy()]
    calibration = LogisticRegression(C=1, solver="lbfgs", random_state=flowering.SEED).fit(
        flowering.logit(x).reshape(-1, 1), y, sample_weight=sample_weight
    )
    global_validation = calibration.predict_proba(flowering.logit(blended_validation).reshape(-1, 1))[:, 1]
    global_test = calibration.predict_proba(flowering.logit(blended_test).reshape(-1, 1))[:, 1]
    rows = []
    for tau in FLOWERING_TAUS:
        offsets = flowering.fit_offsets(validation, global_validation, tau)
        predicted = flowering.apply_offsets(validation, global_validation, offsets)
        rows.append({"tau": tau, "validation_brier": flowering.brier(validation, predicted), "offset_count": len(offsets)})
    table = pd.DataFrame(rows).sort_values("tau")
    selected_tau = float(table.sort_values(["validation_brier", "tau"]).iloc[0].tau)
    selected_offsets = flowering.fit_offsets(validation, global_validation, selected_tau)
    selected_test = flowering.apply_offsets(test, global_test, selected_offsets)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "flowering_extended_tau_validation.csv", index=False)
    pd.DataFrame(
        [{
            "selected_tau": selected_tau,
            "validation_brier": flowering.brier(validation, flowering.apply_offsets(validation, global_validation, selected_offsets)),
            "final_test_brier_not_used_for_selection": flowering.brier(test, selected_test),
            "ft_weight": float(ft_weight),
            "selection_year": 2023,
            "test_years": "2024-2025",
        }]
    ).to_csv(OUT / "flowering_extended_tau_summary.csv", index=False)
    return {"selected_tau": selected_tau, "table": table.to_dict(orient="records")}


def bee_sensitivity() -> dict[str, object]:
    traits = pd.read_csv(bees.TRAITS)[["bee_key", "beegap_family", "sociality"]].drop_duplicates("bee_key")
    raw = pd.read_parquet(bees.RAW, columns=["bee_key", "event_date", "iso_week", "grid_lat_0p5", "grid_lon_0p5"])
    raw["year"] = pd.to_datetime(raw.event_date, utc=True).dt.year
    raw["cell_key"] = raw.grid_lat_0p5.astype(str) + "|" + raw.grid_lon_0p5.astype(str)
    train = raw.loc[raw.year.between(2015, 2022)].copy()
    validation = raw.loc[raw.year.eq(2023)].copy()
    cohort = bees.assign_cohort(train, traits)
    model_cohort = cohort.loc[cohort.activity_support_cohort.isin(["individual_supported", "partially_pooled"])].copy()
    keys = model_cohort.bee_key.tolist()
    train = train.loc[train.bee_key.isin(keys)]
    validation = validation.loc[validation.bee_key.isin(keys)]
    train_counts = bees.counts_by_bee(train, keys)
    validation_counts = bees.counts_by_bee(validation, keys)
    smooth = {key: bees.circular_smooth(train_counts[key]) for key in keys}
    group_profiles = bees.group_profiles(smooth, model_cohort)
    global_profile = np.mean(list(group_profiles.values()), axis=0)
    rows = []
    for tau in BEE_TAUS:
        pooled = {}
        for row in model_cohort.itertuples(index=False):
            group = group_profiles.get(f"{row.beegap_family}|{row.sociality}", global_profile)
            count = train_counts[row.bee_key].sum()
            pooled[row.bee_key] = count / (count + tau) * smooth[row.bee_key] + tau / (count + tau) * group
        scores = bees.evaluate_global(pooled, validation_counts, model_cohort, f"pooled_tau_{tau:g}")
        partial = scores.loc[scores.cohort.eq("partially_pooled")]
        rows.append({"tau": tau, "partial_validation_cross_entropy": float(np.average(partial.cross_entropy, weights=partial.test_records))})
    table = pd.DataFrame(rows).sort_values("tau")
    selected_tau = float(table.sort_values(["partial_validation_cross_entropy", "tau"]).iloc[0].tau)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "bee_extended_tau_validation.csv", index=False)
    pd.DataFrame(
        [{
            "selected_tau": selected_tau,
            "selection_year": 2023,
            "test_years_reserved": "2024-2025",
            "partial_species_in_validation": int(model_cohort.activity_support_cohort.eq("partially_pooled").sum()),
        }]
    ).to_csv(OUT / "bee_extended_tau_summary.csv", index=False)
    return {"selected_tau": selected_tau, "table": table.to_dict(orient="records")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"flowering": flowering_sensitivity(), "bee": bee_sensitivity()}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
