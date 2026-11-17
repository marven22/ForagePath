"""Paper-grade flowering-model benchmark.

Every learned model is fitted twice: (1) plant + season, and (2) plant +
season + measured weather, soil, and location.  Splits are fixed by the
locked V2 dataset: 2015--22 training, 2023 validation, 2024--25 test.

This deliberately treats grouped observations as binomial evidence.  Neural
models use the fraction flowering with the number of checks as its weight;
CatBoost/TabNet receive an equivalent positive/negative two-row expansion.
No synthetic ecological observations are created.

Once these scores have been inspected, 2024--25 is a development benchmark,
not a final publication lockbox. The selected configuration needs one final
external or locked evaluation (for example, on separate NEON records).
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from run_flowering_model_benchmark_v1 import DATA_PATH, evaluate
from run_flowering_model_tuning_v2 import enhanced_weather, seasonal_tuned


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flowering_established_model_benchmark_v1"
SEED = 42
MAX_WEEKLY_ROWS_PER_PLANT_FOR_SCREENING = 500

SEASON = ["iso_week"]
FULL_NUMERIC = [
    "iso_week", "latitude", "longitude", "elevation_in_meters",
    "gdd5_week_proxy", "gdd5_cumulative", "precipitation_sum_mm",
    "precipitation_4week_mm", "mean_temp_c", "mean_temp_4week_c",
    "tmin_mean_c", "tmax_mean_c", "solar_radiation_mean_w_m2",
    "vapor_pressure_mean_pa", "daylength_mean_seconds",
    "snow_water_equivalent_mean_kg_m2", "soil_slope_pct", "soil_ph1to1h2o",
    "soil_awc", "soil_om", "soil_sandtotal", "soil_silttotal", "soil_claytotal",
]
CAT = ["npn_species_id"]


def set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def make_expanded(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """At most two rows per weekly record, each weighted by real checks."""
    pos = frame.loc[frame.positive_checks.gt(0)].copy(); pos["target"] = 1
    pos["sample_weight"] = pos.positive_checks.astype(float)
    neg = frame.loc[frame.negative_checks.gt(0)].copy(); neg["target"] = 0
    neg["sample_weight"] = neg.negative_checks.astype(float)
    out = pd.concat([pos, neg], ignore_index=True)
    return out, out.target.to_numpy(np.int8), out.sample_weight.to_numpy(float)


def screening_training_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Retain every sparse plant and cap only very abundant plants.

    This is a reproducible model-screening budget, not a source-data deletion.
    All validation and test rows remain untouched; the winning model will later
    be refitted on the full training history.
    """
    return (train.groupby("npn_species_id", group_keys=False, sort=False)
            .apply(lambda x: x if len(x) <= MAX_WEEKLY_ROWS_PER_PLANT_FOR_SCREENING
                   else x.sample(MAX_WEEKLY_ROWS_PER_PLANT_FOR_SCREENING, random_state=SEED), include_groups=True)
            .reset_index(drop=True))


def probabilistic_metrics(name: str, split: str, target: pd.DataFrame, p: np.ndarray) -> dict:
    row, _, _ = evaluate(name, target, p)
    return {"model": name, "split": split, **row}


def fit_logistic_or_gam(train: pd.DataFrame, target: pd.DataFrame, full: bool, gam: bool) -> np.ndarray:
    expanded, y, w = make_expanded(train)
    numeric = FULL_NUMERIC if full else SEASON
    # GAM-like model: smooth spline bases, then a penalized binomial GLM.
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("spline", SplineTransformer(n_knots=8 if full else 12, degree=3, include_bias=False) if gam else StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("plant", OneHotEncoder(handle_unknown="ignore"), CAT),
    ], sparse_threshold=0.25)
    x = prep.fit_transform(expanded)
    # A deliberately conservative penalty stabilizes rare-plant coefficients.
    # SGD is the scalable penalized-binomial logistic solver for this million-check
    # benchmark.  The earlier exact SAGA fit was impractically slow at this scale.
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=2e-5, max_iter=80,
                          tol=1e-3, average=True, random_state=SEED)
    model.fit(x, y, sample_weight=w)
    return np.clip(model.predict_proba(prep.transform(target))[:, 1], 1e-5, 1 - 1e-5)


def fit_catboost(train: pd.DataFrame, validation: pd.DataFrame, target: pd.DataFrame, full: bool) -> np.ndarray:
    expanded, y, w = make_expanded(train)
    cols = (FULL_NUMERIC if full else SEASON) + CAT
    x = expanded[cols].copy(); xv = validation[cols].copy(); xt = target[cols].copy()
    # CatBoost requires categoricals as strings and no missing categorical values.
    for c in CAT:
        x[c] = x[c].astype(str); xv[c] = xv[c].astype(str); xt[c] = xt[c].astype(str)
    for c in FULL_NUMERIC if full else SEASON:
        median = float(x[c].median())
        x[c] = x[c].fillna(median); xv[c] = xv[c].fillna(median); xt[c] = xt[c].fillna(median)
    val_expanded, vy, vw = make_expanded(validation)
    valx = val_expanded[cols].copy()
    for c in CAT: valx[c] = valx[c].astype(str)
    for c in FULL_NUMERIC if full else SEASON: valx[c] = valx[c].fillna(float(x[c].median()))
    params = dict(loss_function="Logloss", eval_metric="Logloss", iterations=220, depth=7,
                  learning_rate=0.10, l2_leaf_reg=6.0, random_seed=SEED, verbose=False,
                  od_type="Iter", od_wait=35, allow_writing_files=False)
    if torch.cuda.is_available(): params.update(task_type="GPU", devices="0")
    model = CatBoostClassifier(**params)
    validation_pool = Pool(valx, label=vy, weight=vw, cat_features=CAT)
    model.fit(x, y, sample_weight=w, cat_features=CAT, eval_set=validation_pool, verbose=False)
    return np.clip(model.predict_proba(xt)[:, 1], 1e-5, 1 - 1e-5)


class TabularNet(torch.nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], mode: str):
        super().__init__()
        emb_dim = min(32, max(8, int(np.ceil(np.sqrt(max(cardinalities))))))
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(n, emb_dim) for n in cardinalities])
        width = n_numeric + emb_dim * len(cardinalities)
        if mode == "mlp":
            self.net = torch.nn.Sequential(torch.nn.Linear(width, 192), torch.nn.ReLU(), torch.nn.Dropout(.12),
                                            torch.nn.Linear(192, 96), torch.nn.ReLU(), torch.nn.Dropout(.08), torch.nn.Linear(96, 1))
        else:
            # FT-Transformer feature tokenizer: every numeric feature has its
            # own learned affine token, categoricals have learned embeddings,
            # and the [CLS] token carries the prediction after attention.
            self.numeric_weight = torch.nn.Parameter(torch.empty(n_numeric, 64))
            self.numeric_bias = torch.nn.Parameter(torch.empty(n_numeric, 64))
            self.cat_token = torch.nn.ModuleList([torch.nn.Embedding(n, 64) for n in cardinalities])
            self.cls_token = torch.nn.Parameter(torch.empty(1, 1, 64))
            layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=192, dropout=.10, batch_first=True, norm_first=True)
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64, 1))
            torch.nn.init.normal_(self.numeric_weight, std=.02)
            torch.nn.init.normal_(self.numeric_bias, std=.02)
            torch.nn.init.normal_(self.cls_token, std=.02)
            self.mode = "transformer"
        self.mode = mode

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            items = [numeric] + [e(categorical[:, i]) for i, e in enumerate(self.embeddings)]
            return self.net(torch.cat(items, dim=1)).squeeze(1)
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
        cat_tokens = [e(categorical[:, i]) for i, e in enumerate(self.cat_token)]
        tokens = torch.cat([self.cls_token.expand(numeric.shape[0], -1, -1), numeric_tokens,
                            *[token.unsqueeze(1) for token in cat_tokens]], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(1)


def encode_neural(train: pd.DataFrame, *targets: pd.DataFrame, full: bool):
    cols = FULL_NUMERIC if full else SEASON
    medians = train[cols].median(numeric_only=True)
    means = train[cols].fillna(medians).mean(); stds = train[cols].fillna(medians).std().replace(0, 1)
    levels = {c: {v: i + 1 for i, v in enumerate(train[c].astype(str).drop_duplicates())} for c in CAT}
    def transform(frame: pd.DataFrame):
        numeric = ((frame[cols].fillna(medians) - means) / stds).to_numpy(np.float32)
        categorical = np.column_stack([frame[c].astype(str).map(levels[c]).fillna(0).astype("int64") for c in CAT])
        return numeric, categorical
    return transform(train), [transform(x) for x in targets], [len(levels[c]) + 1 for c in CAT]


def fit_torch(train: pd.DataFrame, validation: pd.DataFrame, target: pd.DataFrame, full: bool, mode: str) -> np.ndarray:
    (xn, xc), [(vn, vc), (tn, tc)], cards = encode_neural(train, validation, target, full=full)
    y = (train.positive_checks / train.observation_checks).to_numpy(np.float32)
    weight = train.observation_checks.to_numpy(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularNet(xn.shape[1], cards, mode).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)

    def predict_in_batches(numeric: np.ndarray, categorical: np.ndarray, size: int = 512 if mode == "transformer" else 32768) -> np.ndarray:
        """Avoid a single giant Transformer attention operation at validation time."""
        outputs = []
        with torch.no_grad():
            for start in range(0, len(numeric), size):
                end = start + size
                outputs.append(torch.sigmoid(model(torch.from_numpy(numeric[start:end]).to(device),
                                                  torch.from_numpy(categorical[start:end]).to(device))).cpu().numpy())
        return np.concatenate(outputs)
    best, best_state, stale = float("inf"), None, 0
    batch_size = 512 if mode == "transformer" else 8192
    max_epochs = 16 if mode == "transformer" else 22
    for epoch in range(max_epochs):
        model.train(); order = np.random.permutation(len(train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            a = torch.from_numpy(xn[idx]).to(device); b = torch.from_numpy(xc[idx]).to(device)
            yy = torch.from_numpy(y[idx]).to(device); ww = torch.from_numpy(weight[idx]).to(device)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(a, b), yy, reduction="none") * ww).sum() / ww.sum()
            optimiser.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
        model.eval()
        pv = predict_in_batches(vn, vc)
        score = float(np.average((pv - (validation.positive_checks / validation.observation_checks).to_numpy()) ** 2, weights=validation.observation_checks))
        if score < best - 1e-6:
            best, best_state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 5: break
    model.load_state_dict(best_state); model.eval()
    return np.clip(predict_in_batches(tn, tc), 1e-5, 1 - 1e-5)


def fit_tabnet(train: pd.DataFrame, validation: pd.DataFrame, target: pd.DataFrame, full: bool) -> np.ndarray:
    expanded, y, weights = make_expanded(train); val, vy, vw = make_expanded(validation)
    (xn, xc), [(vn, vc), (tn, tc)], cards = encode_neural(expanded, val, target, full=full)
    features = np.hstack([xn, xc.astype(np.float32)]); vf = np.hstack([vn, vc.astype(np.float32)]); tf = np.hstack([tn, tc.astype(np.float32)])
    cat_idxs = list(range(xn.shape[1], features.shape[1]))
    model = TabNetClassifier(n_d=24, n_a=24, n_steps=4, gamma=1.4, lambda_sparse=1e-5,
                             cat_idxs=cat_idxs, cat_dims=cards, cat_emb_dim=12, optimizer_fn=torch.optim.AdamW,
                             optimizer_params=dict(lr=1e-3, weight_decay=2e-5), seed=SEED, verbose=0,
                             device_name="cuda" if torch.cuda.is_available() else "cpu")
    model.fit(features, y, weights=weights, eval_set=[(vf, vy)], eval_name=["validation"], eval_metric=["logloss"],
              max_epochs=45, patience=8, batch_size=8192, virtual_batch_size=1024, num_workers=0, drop_last=False)
    return np.clip(model.predict_proba(tf)[:, 1], 1e-5, 1 - 1e-5)


def main() -> int:
    set_seed(); OUT.mkdir(parents=True, exist_ok=True)
    frame = enhanced_weather(pd.read_parquet(DATA_PATH).loc[lambda d: d.weather_available].copy())
    train = frame.loc[frame.data_split.eq("train")].copy(); validation = frame.loc[frame.data_split.eq("validation")].copy(); test = frame.loc[frame.data_split.eq("test")].copy()
    model_train = screening_training_frame(train)
    results, predictions, failures = [], [], []
    alpha, _, _, base_test = seasonal_tuned(train, validation, test)
    results.append(probabilistic_metrics("seasonal_climatology", "test", test, base_test)); predictions.append(pd.DataFrame({"model":"seasonal_climatology", "weekly_record_id":test.weekly_record_id, "probability":base_test}))
    # GPU-capable nonlinear models first: results are written incrementally to
    # the console, rather than being held behind the slower sparse GLM stages.
    jobs = [
        ("catboost", fit_catboost), ("tabnet", fit_tabnet),
        ("residual_mlp", lambda a,b,c,d: fit_torch(a,b,c,d,"mlp")),
        ("ft_transformer", lambda a,b,c,d: fit_torch(a,b,c,d,"transformer")),
        ("logistic", fit_logistic_or_gam), ("gam_spline", fit_logistic_or_gam),
    ]
    requested = {x.strip() for x in os.environ.get("FLOWERING_BENCHMARK_ONLY", "").split(",") if x.strip()}
    if requested:
        jobs = [item for item in jobs if item[0] in requested]
    for full in (False, True):
        variant = "full_features" if full else "season_only"
        for name, fn in jobs:
            label = f"{name}_{variant}"; start = time.time()
            try:
                if name == "logistic": p = fn(model_train, test, full, False)
                elif name == "gam_spline": p = fn(model_train, test, full, True)
                else: p = fn(model_train, validation, test, full)
                result = probabilistic_metrics(label, "test", test, p); result["fit_seconds"] = round(time.time()-start, 2)
                results.append(result); predictions.append(pd.DataFrame({"model":label, "weekly_record_id":test.weekly_record_id, "probability":p}))
                pd.DataFrame(results).sort_values("weighted_brier").to_csv(OUT / "metrics_in_progress.csv", index=False)
                print(label, result, flush=True)
            except Exception as exc:
                failures.append({"model":label, "error":repr(exc)}); print("FAILED", label, repr(exc), flush=True)
    # Simple equal-weight ensemble of the two best *validated candidates* would require
    # validation predictions.  It is intentionally deferred rather than tuned on test.
    new_results = pd.DataFrame(results)
    previous_path = OUT / "metrics.csv"
    if requested and previous_path.exists():
        previous = pd.read_csv(previous_path)
        replace_prefixes = tuple(f"{name}_" for name in requested)
        # The seasonal baseline is recomputed identically; retain a single copy.
        previous = previous.loc[~previous.model.fillna("").str.startswith(replace_prefixes) & previous.model.ne("seasonal_climatology")]
        new_results = pd.concat([previous, new_results], ignore_index=True)
    new_results.sort_values("weighted_brier").to_csv(OUT / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / "test_predictions.parquet", index=False, compression="zstd")
    summary = {"benchmark_id":"flowering-established-model-benchmark-v1", "built_at_utc":datetime.now(timezone.utc).isoformat(),
               "source_dataset":str(DATA_PATH.relative_to(ROOT)).replace('\\\\','/'), "train_rows":len(train), "model_screening_train_rows":len(model_train), "validation_rows":len(validation), "test_rows":len(test),
               "seasonal_prior_strength":alpha, "split":"Train 2015-2022; validation 2023; future-year development benchmark 2024-2025.",
               "evaluation_warning":"Model comparisons have inspected 2024-2025. Before publication, evaluate the frozen candidate once on a separately locked external dataset, such as NEON.",
               "feature_variants":{"season_only":"plant identity plus week of year", "full_features":"season plus Daymet weather, coordinates, elevation, and SSURGO soil fields"},
               "binomial_policy":"All models are trained from observed positive/negative checks; no synthetic observations or artificial class balancing. Screening caps only abundant plants at 500 weekly rows and retains all sparse plants.", "failures":failures}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(new_results.sort_values("weighted_brier").to_string(index=False))


if __name__ == "__main__":
    main()
