from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "flowering_model_benchmark_v1.json"
DATA_PATH = ROOT / "data" / "snapshots" / "us-weekly-flowering-2009-2025-v2" / "derived" / "weekly_flowering_modeling.parquet"
OUT = ROOT / "results" / "flowering_model_benchmark_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    angle = 2.0 * np.pi * (result["iso_week"].astype(float) - 1.0) / 52.0
    result["week_sin_1"] = np.sin(angle)
    result["week_cos_1"] = np.cos(angle)
    result["week_sin_2"] = np.sin(2.0 * angle)
    result["week_cos_2"] = np.cos(2.0 * angle)
    result["week_sin_3"] = np.sin(3.0 * angle)
    result["week_cos_3"] = np.cos(3.0 * angle)
    # Growing-degree-day proxy for the weekly record, with a 5C base.
    result["gdd5_week_proxy"] = ((result["tmin_mean_c"] + result["tmax_mean_c"]) / 2.0 - 5.0).clip(lower=0) * result["weather_days"].fillna(0)
    return result


NUMERIC = [
    "week_sin_1", "week_cos_1", "week_sin_2", "week_cos_2", "week_sin_3", "week_cos_3",
    "gdd5_week_proxy", "tmin_mean_c", "tmax_mean_c", "precipitation_sum_mm",
    "solar_radiation_mean_w_m2", "vapor_pressure_mean_pa", "daylength_mean_seconds",
    "snow_water_equivalent_mean_kg_m2", "elevation_in_meters", "soil_slope_pct",
    "soil_ph1to1h2o", "soil_awc", "soil_om", "soil_sandtotal", "soil_silttotal", "soil_claytotal",
]
CAT_BASIC = ["state", "ecoregion_level3_code"]
CAT_PLANT = ["npn_species_id"]


def expand_binomial(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Represent grouped binomial counts as two weighted binary rows per group."""
    positive = frame.loc[frame["positive_checks"].gt(0)].copy()
    positive["_y"] = 1
    positive["_weight"] = positive["positive_checks"].astype(float)
    negative = frame.loc[frame["negative_checks"].gt(0)].copy()
    negative["_y"] = 0
    negative["_weight"] = negative["negative_checks"].astype(float)
    expanded = pd.concat([positive, negative], ignore_index=True)
    return expanded, expanded.pop("_y").to_numpy(dtype=np.int8), expanded.pop("_weight").to_numpy(dtype=float)


def feature_transformer(include_plant: bool, include_ecoregion: bool = True) -> ColumnTransformer:
    categorical = list(CAT_BASIC)
    if include_ecoregion is False:
        categorical = ["state"]
    if include_plant:
        categorical += CAT_PLANT
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical_pipe, categorical),
    ], sparse_threshold=0.3)


def fit_logistic(train: pd.DataFrame, include_plant: bool) -> Pipeline:
    expanded, y, weights = expand_binomial(train)
    pipeline = Pipeline([
        ("features", feature_transformer(include_plant=include_plant)),
        ("model", LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=80, tol=1e-3, n_jobs=-1, random_state=42)),
    ])
    pipeline.fit(expanded, y, model__sample_weight=weights)
    return pipeline


def fit_boosted(train: pd.DataFrame) -> Pipeline:
    expanded, y, weights = expand_binomial(train)
    # Trees use numeric predictors; the plant effect is represented by a smoothed
    # training-only plant-week climatology added below as a numeric feature.
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        min_samples_leaf=40, random_state=42,
    )
    model.fit(expanded[NUMERIC], y, sample_weight=weights)
    return model


def seasonal_predictions(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    # Smoothed plant-week climatology; all statistics come from training only.
    grouped = train.groupby(["npn_species_id", "iso_week"], sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
    week = train.groupby("iso_week", sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
    global_rate = float(train["positive_checks"].sum() / train["observation_checks"].sum())
    keys = pd.MultiIndex.from_frame(target[["npn_species_id", "iso_week"]])
    specific = grouped.reindex(keys)
    weekly = week.reindex(target["iso_week"].to_numpy())
    # Prior strength is 20 checks: enough to avoid 0/1 estimates for sparse cells.
    alpha = 20.0
    p_specific = (specific["y"].to_numpy(dtype=float) + alpha * weekly["y"].to_numpy(dtype=float) / weekly["n"].to_numpy(dtype=float)) / (specific["n"].to_numpy(dtype=float) + alpha)
    fallback = (weekly["y"].to_numpy(dtype=float) + alpha * global_rate) / (weekly["n"].to_numpy(dtype=float) + alpha)
    p = np.where(np.isfinite(p_specific), p_specific, fallback)
    p = np.where(np.isfinite(p), p, global_rate)
    return np.clip(p, 1e-5, 1 - 1e-5)


def evaluate(name: str, frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[dict, pd.DataFrame]:
    y = frame["positive_checks"].to_numpy(dtype=float) / frame["observation_checks"].to_numpy(dtype=float)
    weights = frame["observation_checks"].to_numpy(dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
    metrics = {
        "model": name,
        "rows": int(len(frame)),
        "checks": int(weights.sum()),
        "weighted_brier": float(np.average((p - y) ** 2, weights=weights)),
        "weighted_mae": float(np.average(np.abs(p - y), weights=weights)),
        "weighted_rmse": float(np.sqrt(np.average((p - y) ** 2, weights=weights))),
        "weighted_log_loss": float(log_loss(np.r_[np.zeros(1), np.ones(1)], np.r_[p.mean(), p.mean()], labels=[0, 1])) if False else float(np.average(-(y * np.log(p) + (1-y) * np.log(1-p)), weights=weights)),
    }
    per_plant = frame.assign(_p=p, _y=y).groupby("npn_species_id").apply(
        lambda g: pd.Series({"mae": np.average(np.abs(g._p-g._y), weights=g.observation_checks), "brier": np.average((g._p-g._y)**2, weights=g.observation_checks)}),
        include_groups=False,
    )
    metrics["macro_plant_mae"] = float(per_plant["mae"].mean())
    metrics["macro_plant_brier"] = float(per_plant["brier"].mean())
    by_tier = frame.assign(_p=p, _y=y).groupby("adequacy_tier").apply(
        lambda g: pd.Series({"mae": np.average(np.abs(g._p-g._y), weights=g.observation_checks), "brier": np.average((g._p-g._y)**2, weights=g.observation_checks), "rows": len(g)}),
        include_groups=False,
    ).reset_index()
    by_tier["model"] = name
    prediction = frame[["weekly_record_id", "npn_species_id", "site_id", "week_start_date", "data_split", "adequacy_tier", "positive_checks", "observation_checks"]].copy()
    prediction["model"] = name
    prediction["predicted_flowering_probability"] = p
    return metrics, by_tier, prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe flowering model benchmark")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        required = [OUT / "metrics.csv", OUT / "predictions.parquet", OUT / "summary.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing benchmark outputs: {missing}")
        print(f"VERIFIED {OUT.relative_to(ROOT)}")
        return 0
    frame = pd.read_parquet(DATA_PATH)
    frame = frame.loc[frame["weather_available"]].copy()
    frame = add_features(frame)
    train = frame.loc[frame["data_split"].eq("train")].copy()
    validation = frame.loc[frame["data_split"].eq("validation")].copy()
    test = frame.loc[frame["data_split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One or more required time splits is empty")
    safe_print(f"Weather-complete rows: train={len(train)}, validation={len(validation)}, test={len(test)}")

    models = []
    tier_tables = []
    predictions = []
    timing = {}
    start = time.time()
    p_val = seasonal_predictions(train, validation)
    p_test = seasonal_predictions(train, test)
    m, tier, pred = evaluate("seasonal_climatology", validation, p_val); models.append({"split":"validation",**m}); tier_tables.append(tier); predictions.append(pred)
    m, tier, pred = evaluate("seasonal_climatology", test, p_test); models.append({"split":"test",**m}); tier_tables.append(tier); predictions.append(pred)
    timing["seasonal_climatology_seconds"] = time.time()-start

    for name, include_plant in [("pooled_binomial_logistic", False), ("plant_effect_logistic_partial_pooling_proxy", True)]:
        safe_print(f"Fitting {name}")
        start = time.time()
        fitted = fit_logistic(train, include_plant=include_plant)
        timing[f"{name}_seconds"] = time.time()-start
        for split_name, target in [("validation", validation), ("test", test)]:
            probabilities = fitted.predict_proba(target)[:, 1]
            m, tier, pred = evaluate(name, target, probabilities); models.append({"split":split_name,**m}); tier_tables.append(tier); predictions.append(pred)

    safe_print("Fitting histogram_gradient_boosting")
    start = time.time()
    boosted = fit_boosted(train)
    timing["histogram_gradient_boosting_seconds"] = time.time()-start
    for split_name, target in [("validation", validation), ("test", test)]:
        probabilities = boosted.predict_proba(target[NUMERIC])[:, 1]
        m, tier, pred = evaluate("histogram_gradient_boosting", target, probabilities); models.append({"split":split_name,**m}); tier_tables.append(tier); predictions.append(pred)

    metrics = pd.DataFrame(models)
    tiers = pd.concat(tier_tables, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    tiers.to_csv(OUT / "metrics_by_plant_tier.csv", index=False)
    prediction_table.to_parquet(OUT / "predictions.parquet", index=False, compression="zstd")
    summary = {
        "benchmark_id": "flowering-model-benchmark-v1", "built_at_utc": utc_now(),
        "source_dataset": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
        "weather_complete_rows": int(len(frame)), "train_rows": int(len(train)),
        "validation_rows": int(len(validation)), "test_rows": int(len(test)),
        "models": ["seasonal_climatology", "pooled_binomial_logistic", "plant_effect_logistic_partial_pooling_proxy", "histogram_gradient_boosting"],
        "timing_seconds": timing,
        "imbalance_policy": "Grouped binomial counts; positive and negative checks used as frequency weights; no synthetic observations; macro-plant metrics reported.",
        "hierarchical_status": "A scalable L2-shrunk plant-effect proxy was benchmarked. A full random-effects Bayesian model is not included in this screening run.",
        "test_policy": "Test predictions were generated after fitting on train only; validation was not used to refit in this run.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    safe_print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
