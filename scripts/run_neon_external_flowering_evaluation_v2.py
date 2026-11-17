"""One external NEON evaluation of the frozen V3 flowering candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from build_us_environment_reference_v1 import load_shapes, shape_contains
from run_flowering_established_model_benchmark_v1 import screening_training_frame
from run_flowering_model_tuning_v2 import enhanced_weather, seasonal_tuned
from run_flowering_v3_ecoregion_hierarchical_ensemble_v1 import (
    brier, cat_predict, fit_offsets, ft_predict, seed,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "snapshots" / "us-weekly-flowering-2009-2025-v3" / "derived" / "weekly_flowering_modeling.parquet"
NEON = ROOT / "results" / "neon_external_validation_preparation_v1" / "neon_weekly_open_flower_matched.parquet"
WEATHER = ROOT / "results" / "neon_external_validation_weather_v1" / "neon_weekly_daymet.parquet"
ECO_DIR = ROOT / "data" / "snapshots" / "us-environment-reference-2026-08-19-v2" / "extracted" / "epa_level3_ecoregions"
OUT = ROOT / "results" / "neon_external_flowering_evaluation_v2"


def haversine(lat1, lon1, lat2, lon2):
    a = np.sin(np.radians(lat2-lat1)/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_ecoregion(frame: pd.DataFrame) -> pd.DataFrame:
    fields, shapes = load_shapes(ECO_DIR)
    prj = next(ECO_DIR.rglob("*.prj"))
    transform = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_wkt(prj.read_text(encoding="utf-8")), always_xy=True)
    records = []
    for lat, lon in frame[["latitude", "longitude"]].drop_duplicates().itertuples(index=False):
        x, y = transform.transform(lon, lat)
        code = "unknown"
        for record, shape in shapes:
            if shape_contains(shape, x, y):
                code = str(record.get("NA_L3CODE") or record.get("US_L3CODE") or "unknown")
                break
        records.append({"latitude": lat, "longitude": lon, "ecoregion_level3_code": code})
    return frame.merge(pd.DataFrame(records), on=["latitude", "longitude"], how="left", validate="many_to_one")


def discrimination(frame, p):
    positive = frame.loc[frame.positive_checks.gt(0), ["positive_checks"]].assign(y=1, p=p[frame.positive_checks.gt(0)], w=lambda x: x.positive_checks)
    negative = frame.loc[frame.negative_checks.gt(0), ["negative_checks"]].assign(y=0, p=p[frame.negative_checks.gt(0)], w=lambda x: x.negative_checks)
    x = pd.concat([positive, negative], ignore_index=True)
    return float(roc_auc_score(x.y, x.p, sample_weight=x.w)), float(average_precision_score(x.y, x.p, sample_weight=x.w))


def score(frame, name, p):
    observed = (frame.positive_checks / frame.observation_checks).to_numpy()
    roc, pr = discrimination(frame, p)
    return {"model": name, "rows": len(frame), "checks": int(frame.observation_checks.sum()),
            "weighted_brier": brier(frame, p),
            "weighted_mae": float(np.average(abs(p-observed), weights=frame.observation_checks)),
            "roc_auc": roc, "pr_auc": pr}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = enhanced_weather(pd.read_parquet(SOURCE).loc[lambda x: x.weather_available].copy())
    neon = pd.read_parquet(NEON).merge(pd.read_parquet(WEATHER), on=["siteID", "week_start_date"], how="left", validate="many_to_one")
    neon = neon.rename(columns={"siteID":"site_id", "decimalLatitude":"latitude", "decimalLongitude":"longitude", "elevation":"elevation_in_meters"})
    neon["data_split"] = "neon_external"; neon["adequacy_tier"] = "external_neon"; neon["state"] = "external"; neon["weekly_record_id"] = "neon_" + neon.index.astype(str)
    for col in ["soil_slope_pct","soil_ph1to1h2o","soil_awc","soil_om","soil_sandtotal","soil_silttotal","soil_claytotal"]:
        neon[col] = np.nan
    neon = add_ecoregion(neon)
    neon = enhanced_weather(neon).loc[lambda x: x.year.ge(2024)].copy().reset_index(drop=True)

    train_pool = source.loc[source.data_split.eq("train")].copy()
    coordinates = neon[["latitude", "longitude"]].drop_duplicates().to_numpy()
    source_coordinates = train_pool[["latitude", "longitude"]].drop_duplicates()
    source_coordinates["keep"] = [not np.any(haversine(row.latitude, row.longitude, coordinates[:,0], coordinates[:,1]) < 1) for row in source_coordinates.itertuples(index=False)]
    train_pool = train_pool.merge(source_coordinates.loc[source_coordinates.keep, ["latitude", "longitude"]], on=["latitude", "longitude"], how="inner")
    validation = source.loc[source.data_split.eq("validation")].copy().reset_index(drop=True)
    train = screening_training_frame(train_pool)

    seed(); cat_val, cat_neon = cat_predict(train, validation, neon); ft_val, ft_neon = ft_predict(train, validation, neon)
    weight = min(np.linspace(0, 1, 101), key=lambda w: brier(validation, w*ft_val + (1-w)*cat_val))
    raw_val, raw_neon = weight*ft_val + (1-weight)*cat_val, weight*ft_neon + (1-weight)*cat_neon
    # Grouped-binomial Platt scaling trained only on USA-NPN 2023 validation.
    x = np.r_[raw_val, raw_val]; y = np.r_[np.ones(len(raw_val)), np.zeros(len(raw_val))]
    w = np.r_[validation.positive_checks.to_numpy(), validation.negative_checks.to_numpy()]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42).fit(logit(np.clip(x,1e-5,1-1e-5)).reshape(-1,1), y, sample_weight=w)
    global_neon = calibrator.predict_proba(logit(np.clip(raw_neon,1e-5,1-1e-5)).reshape(-1,1))[:,1]
    global_val = calibrator.predict_proba(logit(np.clip(raw_val,1e-5,1-1e-5)).reshape(-1,1))[:,1]
    tau_grid = [.10,.20,.35,.50,.75,1.0]
    tau, offsets = min(((tau, fit_offsets(validation, global_val, tau)) for tau in tau_grid), key=lambda item: brier(validation, expit(logit(np.clip(global_val,1e-5,1-1e-5)) + validation.npn_species_id.map(item[1]).fillna(0).to_numpy())))
    hierarchical_neon = expit(logit(np.clip(global_neon,1e-5,1-1e-5)) + neon.npn_species_id.map(offsets).fillna(0).to_numpy())
    _, _, _, seasonal_neon = seasonal_tuned(train_pool, validation, neon)

    predictions = {"seasonal": seasonal_neon, "catboost_ecoregion": cat_neon, "ft_transformer_ecoregion": ft_neon, "global_ensemble": global_neon, "hierarchical_ensemble": hierarchical_neon}
    pd.DataFrame([score(neon, name, p) for name, p in predictions.items()]).to_csv(OUT / "metrics.csv", index=False)
    saved = neon[["weekly_record_id","site_id","npn_species_id","scientific_name","week_start_date","positive_checks","observation_checks","ecoregion_level3_code"]].copy()
    for name, p in predictions.items(): saved[name] = p
    saved.to_parquet(OUT / "predictions.parquet", index=False, compression="zstd")
    summary = {"external_years":[2024,2025], "sites":int(neon.site_id.nunique()), "plants":int(neon.npn_species_id.nunique()), "ecoregion_codes":int(neon.ecoregion_level3_code.nunique()), "ft_weight":float(weight), "hierarchical_tau":float(tau), "weather_coverage":float(neon.weather_days.notna().mean()), "nearby_usnpn_training_site_coordinates_excluded":int((~source_coordinates.keep).sum()), "soil_policy":"NEON soil covariates are unavailable, so full models use USA-NPN training-median imputation. This is a partial-covariate external test, not a full deployment test."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "metrics.csv").to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
