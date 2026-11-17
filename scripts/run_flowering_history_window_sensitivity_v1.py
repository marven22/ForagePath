"""Compare finalist flowering models across fixed historical training windows.

Every window ends in 2022.  The 2023 records select ensemble and calibration
settings.  The 2024--2025 records remain a common future test set.  This
mirrors the national bee history-window experiment without re-running models
that the primary flowering benchmark had already eliminated.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import run_flowering_established_model_benchmark_v1 as base
import run_flowering_v3_ecoregion_hierarchical_ensemble_v1 as hierarchy
from run_flowering_model_tuning_v2 import enhanced_weather, seasonal_tuned


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "snapshots" / "us-weekly-flowering-2009-2025-v3" / "derived" / "weekly_flowering_modeling.parquet"
OUT = ROOT / "results" / "flowering_history_window_sensitivity_v1"
WINDOW_STARTS = (2009, 2011, 2013, 2015)
TRAIN_END = 2022
VALIDATION_YEAR = 2023
TEST_YEARS = (2024, 2025)
THRESHOLDS = (0.10, 0.25, 0.50)
SEED = 42


def seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    hierarchy.seed()


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(probability / (1 - probability))


def brier(frame: pd.DataFrame, probability: np.ndarray) -> float:
    observed = (frame.positive_checks / frame.observation_checks).to_numpy(float)
    return float(np.average((probability - observed) ** 2, weights=frame.observation_checks))


def ranking_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    positive = frame.positive_checks.to_numpy(float)
    negative = frame.negative_checks.to_numpy(float)
    scores = np.r_[probability, probability]
    target = np.r_[np.ones(len(frame), dtype=np.int8), np.zeros(len(frame), dtype=np.int8)]
    weights = np.r_[positive, negative]
    keep = weights > 0
    return {
        "weighted_roc_auc": float(roc_auc_score(target[keep], scores[keep], sample_weight=weights[keep])),
        "weighted_pr_auc": float(average_precision_score(target[keep], scores[keep], sample_weight=weights[keep])),
    }


def threshold_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    positive = frame.positive_checks.to_numpy(float)
    negative = frame.negative_checks.to_numpy(float)
    rows: dict[str, float] = {}
    for threshold in THRESHOLDS:
        selected = probability >= threshold
        tp = positive[selected].sum(); fp = negative[selected].sum()
        fn = positive[~selected].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        suffix = f"{threshold:.2f}".replace(".", "_")
        rows.update({f"precision_{suffix}": float(precision), f"recall_{suffix}": float(recall), f"f1_{suffix}": float(f1)})
    return rows


def calibration_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    """Diagnostic calibration fit on test outcomes.  It never changes predictions."""
    x = np.r_[logit(probability), logit(probability)].reshape(-1, 1)
    target = np.r_[np.ones(len(frame)), np.zeros(len(frame))]
    weights = np.r_[frame.positive_checks.to_numpy(float), frame.negative_checks.to_numpy(float)]
    keep = weights > 0
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
    model.fit(x[keep], target[keep], sample_weight=weights[keep])
    return {"calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}


def metrics(label: str, frame: pd.DataFrame, probability: np.ndarray, start_year: int) -> dict[str, float | int | str]:
    result, _, _ = base.evaluate(label, frame, probability)
    result.update(ranking_metrics(frame, probability))
    result.update(threshold_metrics(frame, probability))
    result.update(calibration_metrics(frame, probability))
    result["train_start_year"] = start_year
    result["train_end_year"] = TRAIN_END
    result["validation_year"] = VALIDATION_YEAR
    result["test_years"] = "2024-2025"
    return result


def fit_window(frame: pd.DataFrame, start_year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed()
    train = frame.loc[frame.year.between(start_year, TRAIN_END)].copy().reset_index(drop=True)
    validation = frame.loc[frame.year.eq(VALIDATION_YEAR)].copy().reset_index(drop=True)
    test = frame.loc[frame.year.between(*TEST_YEARS)].copy().reset_index(drop=True)
    screen = base.screening_training_frame(train)

    _, _, _, seasonal = seasonal_tuned(train, validation, test)
    cat_val, cat_test = hierarchy.cat_predict(screen, validation, test)
    ft_val, ft_test = hierarchy.ft_predict(screen, validation, test)

    grid = np.linspace(0, 1, 101)
    ft_weight = float(min(grid, key=lambda w: brier(validation, w * ft_val + (1 - w) * cat_val)))
    raw_val = ft_weight * ft_val + (1 - ft_weight) * cat_val
    raw_test = ft_weight * ft_test + (1 - ft_weight) * cat_test

    calibrator = LogisticRegression(C=1, solver="lbfgs", random_state=SEED)
    calibration_x = np.r_[logit(raw_val), logit(raw_val)].reshape(-1, 1)
    calibration_y = np.r_[np.ones(len(validation)), np.zeros(len(validation))]
    calibration_w = np.r_[validation.positive_checks.to_numpy(float), validation.negative_checks.to_numpy(float)]
    calibrator.fit(calibration_x, calibration_y, sample_weight=calibration_w)
    global_val = calibrator.predict_proba(logit(raw_val).reshape(-1, 1))[:, 1]
    global_test = calibrator.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1]

    candidates = []
    for tau in (.10, .20, .35, .50, .75, 1.0):
        offsets = hierarchy.fit_offsets(validation, global_val, tau)
        candidates.append((tau, brier(validation, hierarchy.apply_offsets(validation, global_val, offsets)), offsets))
    tau, _, offsets = min(candidates, key=lambda value: value[1])
    hierarchical_test = hierarchy.apply_offsets(test, global_test, offsets)

    probabilities = {
        "seasonal_climatology": seasonal,
        "catboost_ecoregion": cat_test,
        "hierarchical_plant_calibrated_ensemble": hierarchical_test,
    }
    rows = [metrics(name, test, probability, start_year) for name, probability in probabilities.items()]
    predictions = pd.DataFrame({
        "weekly_record_id": test.weekly_record_id,
        "site_id": test.site_id,
        "npn_species_id": test.npn_species_id,
        "year": test.year,
        "positive_checks": test.positive_checks,
        "negative_checks": test.negative_checks,
        "observation_checks": test.observation_checks,
        "adequacy_tier": test.adequacy_tier,
        "seasonal_climatology": seasonal,
        "catboost_ecoregion": cat_test,
        "hierarchical_plant_calibrated_ensemble": hierarchical_test,
    })
    metadata = {
        "train": f"{start_year}-{TRAIN_END}", "validation": str(VALIDATION_YEAR), "test": "2024-2025",
        "train_rows": int(len(train)), "screening_train_rows": int(len(screen)), "validation_rows": int(len(validation)), "test_rows": int(len(test)),
        "ft_transformer_weight": ft_weight, "selected_random_effect_prior_sd_tau": float(tau),
        "policy": "Each model sees only its stated historical window. Ensemble weight, global calibration, and plant random-effect prior are selected from 2023 only.",
    }
    return pd.DataFrame(rows), predictions, metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA)
    data = enhanced_weather(data.loc[data.weather_available].copy())
    all_rows = []
    manifests = []
    for start_year in WINDOW_STARTS:
        target = OUT / f"train_{start_year}_{TRAIN_END}"
        target.mkdir(parents=True, exist_ok=True)
        started = time.time()
        summary, predictions, metadata = fit_window(data, start_year)
        summary["fit_seconds"] = round(time.time() - started, 2)
        summary.to_csv(target / "final_test_metrics.csv", index=False)
        predictions.to_parquet(target / "final_test_predictions.parquet", index=False, compression="zstd")
        (target / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        all_rows.append(summary)
        manifests.append(metadata)
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / "history_window_model_summary_in_progress.csv", index=False)
        print(f"completed train {start_year}-{TRAIN_END}", flush=True)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT / "history_window_model_summary.csv", index=False)
    (OUT / "experiment_manifest.json").write_text(json.dumps({"built_at_utc": datetime.now(timezone.utc).isoformat(), "windows": manifests}, indent=2), encoding="utf-8")
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
