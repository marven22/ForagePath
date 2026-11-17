from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from run_flowering_model_benchmark_v1 import (
    DATA_PATH, OUT as BASE_OUT, add_features, evaluate, expand_binomial, seasonal_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flowering_model_tuning_v2"


def seasonal_tuned(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    scores = []
    for alpha in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        # Reuse the same estimator with an explicit prior-strength implementation.
        grouped = train.groupby(["npn_species_id", "iso_week"], sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
        week = train.groupby("iso_week", sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
        global_rate = float(train.positive_checks.sum() / train.observation_checks.sum())
        def predict(target):
            keys = pd.MultiIndex.from_frame(target[["npn_species_id", "iso_week"]])
            specific = grouped.reindex(keys)
            weekly = week.reindex(target.iso_week.to_numpy())
            week_rate = weekly.y.to_numpy(float) / weekly.n.to_numpy(float)
            p = (specific.y.to_numpy(float) + alpha * week_rate) / (specific.n.to_numpy(float) + alpha)
            fallback = (weekly.y.to_numpy(float) + alpha * global_rate) / (weekly.n.to_numpy(float) + alpha)
            p = np.where(np.isfinite(p), p, fallback)
            return np.clip(np.where(np.isfinite(p), p, global_rate), 1e-5, 1-1e-5)
        p = predict(validation)
        y = validation.positive_checks.to_numpy(float) / validation.observation_checks.to_numpy(float)
        w = validation.observation_checks.to_numpy(float)
        scores.append({"alpha": alpha, "validation_weighted_brier": float(np.average((p-y)**2, weights=w))})
    chosen = min(scores, key=lambda row: row["validation_weighted_brier"])["alpha"]
    # Recompute predictions with the selected prior strength.
    grouped = train.groupby(["npn_species_id", "iso_week"], sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
    week = train.groupby("iso_week", sort=False).agg(y=("positive_checks", "sum"), n=("observation_checks", "sum"))
    global_rate = float(train.positive_checks.sum() / train.observation_checks.sum())
    def predict(target):
        keys = pd.MultiIndex.from_frame(target[["npn_species_id", "iso_week"]])
        specific = grouped.reindex(keys); weekly = week.reindex(target.iso_week.to_numpy())
        week_rate = weekly.y.to_numpy(float) / weekly.n.to_numpy(float)
        p = (specific.y.to_numpy(float) + chosen * week_rate) / (specific.n.to_numpy(float) + chosen)
        fallback = (weekly.y.to_numpy(float) + chosen * global_rate) / (weekly.n.to_numpy(float) + chosen)
        return np.clip(np.where(np.isfinite(p), p, np.where(np.isfinite(fallback), fallback, global_rate)), 1e-5, 1-1e-5)
    return chosen, scores, predict(validation), predict(test)


def enhanced_weather(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_features(frame)
    result = result.sort_values(["site_id", "year", "week_start_date"]).copy()
    result["mean_temp_c"] = (result.tmin_mean_c + result.tmax_mean_c) / 2.0
    result["gdd5_cumulative"] = result.groupby(["site_id", "year"], sort=False).gdd5_week_proxy.cumsum()
    result["precipitation_4week_mm"] = result.groupby("site_id", sort=False).precipitation_sum_mm.transform(lambda s: s.rolling(4, min_periods=1).sum())
    result["mean_temp_4week_c"] = result.groupby("site_id", sort=False).mean_temp_c.transform(lambda s: s.rolling(4, min_periods=1).mean())
    result["week_of_year"] = result.iso_week.astype(float)
    return result.sort_index()


ENH_NUMERIC = [
    "week_sin_1", "week_cos_1", "week_sin_2", "week_cos_2", "week_sin_3", "week_cos_3",
    "gdd5_week_proxy", "gdd5_cumulative", "precipitation_sum_mm", "precipitation_4week_mm",
    "mean_temp_c", "mean_temp_4week_c", "tmin_mean_c", "tmax_mean_c", "solar_radiation_mean_w_m2",
    "vapor_pressure_mean_pa", "daylength_mean_seconds", "snow_water_equivalent_mean_kg_m2",
    "elevation_in_meters", "soil_slope_pct", "soil_ph1to1h2o", "soil_awc", "soil_om",
    "soil_sandtotal", "soil_silttotal", "soil_claytotal",
]


def fit_fourier_plant_model(train: pd.DataFrame):
    expanded, y, weights = expand_binomial(train)
    numeric_imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler(with_mean=False)
    numeric = scaler.fit_transform(numeric_imputer.fit_transform(expanded[ENH_NUMERIC]))
    categories = expanded[["state", "ecoregion_level3_code"]].astype(str)
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat = cat_encoder.fit_transform(categories)
    plant_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    plant = plant_encoder.fit_transform(expanded[["npn_species_id"]].astype(str))
    harmonics = [expanded["week_sin_1"].to_numpy(), expanded["week_cos_1"].to_numpy(), expanded["week_sin_2"].to_numpy(), expanded["week_cos_2"].to_numpy()]
    interactions = [plant.multiply(value[:, None]) for value in harmonics]
    x = hstack([numeric, cat, plant, *interactions], format="csr")
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, max_iter=100, tol=1e-3, average=True, random_state=42)
    start = time.time(); model.fit(x, y, sample_weight=weights); elapsed = time.time() - start
    def predict(target):
        numeric_t = scaler.transform(numeric_imputer.transform(target[ENH_NUMERIC]))
        cat_t = cat_encoder.transform(target[["state", "ecoregion_level3_code"]].astype(str))
        plant_t = plant_encoder.transform(target[["npn_species_id"]].astype(str))
        h = [target["week_sin_1"].to_numpy(), target["week_cos_1"].to_numpy(), target["week_sin_2"].to_numpy(), target["week_cos_2"].to_numpy()]
        return model.predict_proba(hstack([numeric_t, cat_t, plant_t, *[plant_t.multiply(v[:, None]) for v in h]], format="csr"))[:, 1]
    return predict, elapsed, x.shape[1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA_PATH)
    frame = frame.loc[frame.weather_available].copy()
    frame = enhanced_weather(frame)
    train = frame.loc[frame.data_split.eq("train")].copy()
    validation = frame.loc[frame.data_split.eq("validation")].copy()
    test = frame.loc[frame.data_split.eq("test")].copy()
    chosen_alpha, alpha_scores, p_val_base, p_test_base = seasonal_tuned(train, validation, test)
    rows = []; tiers = []; preds = []
    for split, target, p in [("validation", validation, p_val_base), ("test", test, p_test_base)]:
        m, t, pr = evaluate("tuned_seasonal_climatology", target, p); rows.append({"split": split, **m}); tiers.append(t); preds.append(pr)
    predict_fourier, fit_seconds, feature_count = fit_fourier_plant_model(train)
    p_fourier_validation = predict_fourier(validation)
    p_fourier_test = predict_fourier(test)
    for split, target in [("validation", validation), ("test", test)]:
        p = p_fourier_validation if split == "validation" else p_fourier_test
        m, t, pr = evaluate("fourier_plant_weather_logistic_sgd", target, p); rows.append({"split": split, **m}); tiers.append(t); preds.append(pr)
    # Calibrate only after fitting the flowering model. Validation is used to
    # learn a one-dimensional logit calibration; the final score is on test.
    calibration_frame = pd.DataFrame({"logit": np.log(np.clip(p_fourier_validation, 1e-6, 1-1e-6) / np.clip(1-p_fourier_validation, 1e-6, 1)), "y": validation.positive_checks / validation.observation_checks})
    cal_positive = calibration_frame.loc[validation.positive_checks.gt(0)].copy(); cal_positive["target"] = 1; cal_positive["weight"] = validation.loc[validation.positive_checks.gt(0), "positive_checks"].to_numpy(float)
    cal_negative = calibration_frame.loc[validation.negative_checks.gt(0)].copy(); cal_negative["target"] = 0; cal_negative["weight"] = validation.loc[validation.negative_checks.gt(0), "negative_checks"].to_numpy(float)
    calibration_rows = pd.concat([cal_positive, cal_negative], ignore_index=True)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(calibration_rows[["logit"]], calibration_rows["target"], sample_weight=calibration_rows["weight"])
    p_cal_test = calibrator.predict_proba(pd.DataFrame({"logit": np.log(np.clip(p_fourier_test, 1e-6, 1-1e-6) / np.clip(1-p_fourier_test, 1e-6, 1))}))[:, 1]
    m, t, pr = evaluate("fourier_plant_weather_logistic_calibrated", test, p_cal_test); rows.append({"split": "test", **m}); tiers.append(t); preds.append(pr)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(tiers, ignore_index=True).to_csv(OUT / "metrics_by_plant_tier.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False, compression="zstd")
    summary = {
        "benchmark_id": "flowering-model-tuning-v2", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str((BASE_OUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "weather_complete_rows": int(len(frame)), "train_rows": int(len(train)), "validation_rows": int(len(validation)), "test_rows": int(len(test)),
        "seasonal_prior_strength_candidates": alpha_scores, "selected_alpha": chosen_alpha,
        "fourier_model_feature_count": int(feature_count), "fourier_model_fit_seconds": fit_seconds,
        "calibration_policy": "One-dimensional logit calibration fitted on validation predictions and applied once to the held-out test set.",
        "data_policy": "All weather-complete records used; grouped binomial checks represented by frequency weights; no synthetic rows; test not used for tuning.",
        "interpretation": "The Fourier model allows every plant to have its own seasonal curve and weather response, while L2 shrinkage stabilizes sparse plants."
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    from datetime import datetime, timezone
    raise SystemExit(main())
