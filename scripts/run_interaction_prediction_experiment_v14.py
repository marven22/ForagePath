from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "interaction_benchmark_v14"
OUT = ROOT / "results" / "interaction_prediction_v14"
SEED = 20260815
SMOOTHING = 10.0


def bucket(value: str, modulus: int = 5) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % modulus


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def platt(calibration_y: np.ndarray, calibration_p: np.ndarray, test_p: np.ndarray) -> np.ndarray:
    if len(np.unique(calibration_y)) < 2:
        return test_p
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    model.fit(logit(calibration_p).reshape(-1, 1), calibration_y)
    return model.predict_proba(logit(test_p).reshape(-1, 1))[:, 1]


@dataclass
class RateTable:
    keys: list[str]
    values: pd.Series
    prior: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, keys: list[str], prior: float) -> "RateTable":
        grouped = frame.groupby(keys)["interaction_observed"].agg(["sum", "count"])
        values = (grouped["sum"] + SMOOTHING * prior) / (grouped["count"] + SMOOTHING)
        return cls(keys, values, prior)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if len(self.keys) == 1:
            return frame[self.keys[0]].map(self.values).fillna(self.prior).to_numpy(float)
        index = pd.MultiIndex.from_frame(frame[self.keys])
        return self.values.reindex(index, fill_value=self.prior).to_numpy(float)


class FeatureBuilder:
    def __init__(self, train: pd.DataFrame, external: pd.DataFrame):
        train = self.with_context(train)
        self.prior = float(train["interaction_observed"].mean())
        self.rates = {
            "bee_rate": RateTable.fit(train, ["bee"], self.prior),
            "plant_rate": RateTable.fit(train, ["plant"], self.prior),
            "pair_rate": RateTable.fit(train, ["bee", "plant"], self.prior),
            "family_pair_rate": RateTable.fit(train, ["bee_family", "plant_family"], self.prior),
            "tribe_family_rate": RateTable.fit(train, ["bee_tribe", "plant_family"], self.prior),
            "genus_pair_rate": RateTable.fit(train, ["bee_genus", "plant_genus"], self.prior),
            "bee_week_rate": RateTable.fit(train, ["bee", "week"], self.prior),
            "plant_week_rate": RateTable.fit(train, ["plant", "week"], self.prior),
            "study_rate": RateTable.fit(train, ["study"], self.prior),
            "study_bee_rate": RateTable.fit(train, ["study", "bee"], self.prior),
            "study_plant_rate": RateTable.fit(train, ["study", "plant"], self.prior),
            "study_pair_rate": RateTable.fit(train, ["study", "bee", "plant"], self.prior),
            "site_bee_rate": RateTable.fit(train, ["study_site", "bee"], self.prior),
            "site_plant_rate": RateTable.fit(train, ["study_site", "plant"], self.prior),
            "site_pair_rate": RateTable.fit(train, ["study_site", "bee", "plant"], self.prior),
            "site_bee_week_rate": RateTable.fit(train, ["study_site", "bee", "week"], self.prior),
            "site_plant_week_rate": RateTable.fit(train, ["study_site", "plant", "week"], self.prior),
        }
        self.external_pairs = set(zip(external["bee"], external["plant"]))
        self.external_bee_degree = external.groupby("bee")["plant"].nunique()
        self.external_plant_degree = external.groupby("plant")["bee"].nunique()
        self._fit_svd(train)

    @staticmethod
    def with_context(frame: pd.DataFrame) -> pd.DataFrame:
        if "study_site" in frame.columns:
            return frame
        result = frame.copy()
        result["study_site"] = result["study"].astype(str) + "|" + result["site"].astype(str)
        return result

    def _fit_svd(self, frame: pd.DataFrame) -> None:
        positive = frame[frame["interaction_observed"].eq(1)][["bee", "plant"]].drop_duplicates()
        self.positive_pairs = set(zip(positive["bee"], positive["plant"]))
        self.bees = sorted(positive["bee"].unique())
        self.plants = sorted(positive["plant"].unique())
        self.bee_index = {name: index for index, name in enumerate(self.bees)}
        self.plant_index = {name: index for index, name in enumerate(self.plants)}
        row = positive["bee"].map(self.bee_index).to_numpy()
        column = positive["plant"].map(self.plant_index).to_numpy()
        matrix = sparse.csr_matrix((np.ones(len(positive)), (row, column)), shape=(len(self.bees), len(self.plants)))
        components = max(1, min(16, min(matrix.shape) - 1))
        model = TruncatedSVD(n_components=components, n_iter=10, random_state=SEED)
        self.bee_embedding = model.fit_transform(matrix)
        self.plant_embedding = model.components_.T

    def svd_score(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for position, (bee, plant) in enumerate(zip(frame["bee"], frame["plant"])):
            i, j = self.bee_index.get(bee), self.plant_index.get(plant)
            if i is not None and j is not None:
                result[position] = float(self.bee_embedding[i] @ self.plant_embedding[j])
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = self.with_context(frame)
        result = pd.DataFrame(index=frame.index)
        for name, lookup in self.rates.items():
            result[name] = lookup.transform(frame)
        radians = 2 * np.pi * (frame["week"].to_numpy(float) - 1) / 52.1775
        result["week_sin"] = np.sin(radians)
        result["week_cos"] = np.cos(radians)
        result["svd_score"] = self.svd_score(frame)
        result["external_edge"] = [int(pair in self.external_pairs) for pair in zip(frame["bee"], frame["plant"])]
        result["external_bee_degree"] = np.log1p(frame["bee"].map(self.external_bee_degree).fillna(0).to_numpy(float))
        result["external_plant_degree"] = np.log1p(frame["plant"].map(self.external_plant_degree).fillna(0).to_numpy(float))
        return result.reset_index(drop=True)

    def evidence_stratum(self, frame: pd.DataFrame) -> np.ndarray:
        labels = []
        for bee, plant in zip(frame["bee"], frame["plant"]):
            bee_seen = bee in self.bee_index
            plant_seen = plant in self.plant_index
            if not bee_seen and not plant_seen:
                labels.append("unseen_both_taxa")
            elif not bee_seen:
                labels.append("unseen_bee")
            elif not plant_seen:
                labels.append("unseen_plant")
            elif (bee, plant) not in self.positive_pairs:
                labels.append("unseen_pair_known_taxa")
            else:
                labels.append("seen_pair")
        return np.asarray(labels)


FEATURES = {
    "taxonomy_season": [
        "bee_rate", "plant_rate", "family_pair_rate", "tribe_family_rate",
        "genus_pair_rate", "bee_week_rate", "plant_week_rate", "week_sin", "week_cos",
    ],
    "temporal_graph": [
        "bee_rate", "plant_rate", "pair_rate", "family_pair_rate", "tribe_family_rate",
        "genus_pair_rate", "bee_week_rate", "plant_week_rate", "week_sin", "week_cos", "svd_score",
    ],
    "hierarchical_context": [
        "bee_rate", "plant_rate", "pair_rate", "family_pair_rate", "tribe_family_rate",
        "genus_pair_rate", "bee_week_rate", "plant_week_rate", "week_sin", "week_cos", "svd_score",
        "study_rate", "study_bee_rate", "study_plant_rate", "study_pair_rate",
        "site_bee_rate", "site_plant_rate", "site_pair_rate", "site_bee_week_rate", "site_plant_week_rate",
    ],
    "archive_augmented": [
        "bee_rate", "plant_rate", "pair_rate", "family_pair_rate", "tribe_family_rate",
        "genus_pair_rate", "bee_week_rate", "plant_week_rate", "week_sin", "week_cos", "svd_score",
        "external_edge", "external_bee_degree", "external_plant_degree",
    ],
}


def expected_calibration_error(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 10).astype(int), 9)
    total = len(y)
    return float(sum(
        (bins == item).sum() / total * abs(y[bins == item].mean() - p[bins == item].mean())
        for item in range(10) if (bins == item).any()
    ))


def recall_at_fraction(frame: pd.DataFrame, probabilities: np.ndarray, fraction: float = 0.1) -> float:
    working = frame[["event_id", "interaction_observed"]].copy()
    working["probability"] = probabilities
    captured = positives = 0
    for _, event in working.groupby("event_id"):
        k = max(1, int(np.ceil(len(event) * fraction)))
        captured += int(event.nlargest(k, "probability")["interaction_observed"].sum())
        positives += int(event["interaction_observed"].sum())
    return float(captured / positives) if positives else float("nan")


def metrics(split: str, model: str, test: pd.DataFrame, probability: np.ndarray) -> dict:
    y = test["interaction_observed"].to_numpy(int)
    return {
        "split": split,
        "model": model,
        "rows": len(test),
        "events": int(test["event_id"].nunique()),
        "positive_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_error": expected_calibration_error(y, probability),
        "event_recall_at_top_10pct": recall_at_fraction(test, probability),
    }


def evaluate(
    split: str, train: pd.DataFrame, test: pd.DataFrame, external: pd.DataFrame
) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    internal_bucket = train["event_id"].map(lambda value: bucket(str(value), modulus=7))
    gate = train[internal_bucket.eq(0)].copy()
    calibration = train[internal_bucket.eq(1)].copy()
    fit = train[~internal_bucket.isin([0, 1])].copy()
    if any(part["interaction_observed"].nunique() < 2 for part in (fit, gate, calibration)):
        raise ValueError(f"Split {split} lacks both labels in fit, gate, or calibration data")
    builder = FeatureBuilder(fit, external)
    x_fit = builder.transform(fit)
    x_gate = builder.transform(gate)
    x_cal = builder.transform(calibration)
    x_test = builder.transform(test)
    y_fit = fit["interaction_observed"].to_numpy(int)
    y_cal = calibration["interaction_observed"].to_numpy(int)

    predictions: dict[str, np.ndarray] = {}
    gate_predictions: dict[str, np.ndarray] = {}
    prevalence = np.full(len(test), builder.prior)
    predictions["prevalence"] = prevalence
    gate_predictions["prevalence"] = np.full(len(gate), builder.prior)
    popularity_gate = np.sqrt(x_gate["bee_rate"].to_numpy() * x_gate["plant_rate"].to_numpy())
    popularity_cal = np.sqrt(x_cal["bee_rate"].to_numpy() * x_cal["plant_rate"].to_numpy())
    popularity_test = np.sqrt(x_test["bee_rate"].to_numpy() * x_test["plant_rate"].to_numpy())
    predictions["popularity"] = platt(y_cal, popularity_cal, popularity_test)
    gate_predictions["popularity"] = platt(y_cal, popularity_cal, popularity_gate)

    for model_name, columns in FEATURES.items():
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=5.0,
            random_state=SEED,
        )
        model.fit(x_fit[columns], y_fit)
        gate_probability = model.predict_proba(x_gate[columns])[:, 1]
        cal_probability = model.predict_proba(x_cal[columns])[:, 1]
        test_probability = model.predict_proba(x_test[columns])[:, 1]
        predictions[model_name] = platt(y_cal, cal_probability, test_probability)
        gate_predictions[model_name] = platt(y_cal, cal_probability, gate_probability)

    allowed = ["popularity", "taxonomy_season", "temporal_graph", "hierarchical_context"]
    gate_strata = builder.evidence_stratum(gate)
    test_strata = builder.evidence_stratum(test)
    gate_y = gate["interaction_observed"].to_numpy(int)
    choices: dict[str, str] = {}
    gate_rows = []
    for stratum in sorted(set(gate_strata) | set(test_strata)):
        mask = gate_strata == stratum
        eligible = mask.sum() >= 30 and len(np.unique(gate_y[mask])) == 2
        scores = {
            name: float(brier_score_loss(gate_y[mask], gate_predictions[name][mask]))
            for name in allowed
        } if eligible else {}
        selected = min(scores, key=scores.get) if scores else "taxonomy_season"
        choices[stratum] = selected
        gate_rows.append({
            "split": split,
            "evidence_stratum": stratum,
            "gate_rows": int(mask.sum()),
            "selected_model": selected,
            "selection_basis": "lowest Brier on an internal event-held gate set" if scores else "taxonomy fallback; insufficient gate data",
            **{f"gate_brier_{name}": scores.get(name, float("nan")) for name in allowed},
        })
    adaptive = np.empty(len(test), dtype=float)
    for stratum, selected in choices.items():
        mask = test_strata == stratum
        adaptive[mask] = predictions[selected][mask]
    predictions["adaptive_context_graph"] = adaptive

    rows = [metrics(split, name, test, probability) for name, probability in predictions.items()]
    output = test[[
        "event_id", "study", "site", "date", "year", "week", "bee", "plant",
        "interaction_observed", "observed_frequency",
    ]].copy()
    output.insert(0, "split", split)
    for name, probability in predictions.items():
        output[f"p_{name}"] = probability
    output["evidence_stratum"] = test_strata
    output["adaptive_selected_model"] = [choices[value] for value in test_strata]
    return rows, output, gate_rows


def split_definitions(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    definitions = []
    event_test = data["event_id"].map(lambda value: bucket(str(value))).eq(0)
    definitions.append(("held_event", data[~event_test].copy(), data[event_test].copy()))

    pair_test = (data["bee"] + "|" + data["plant"]).map(bucket).eq(0)
    definitions.append(("unseen_pair", data[~pair_test].copy(), data[pair_test].copy()))

    plant_test = data["plant"].map(bucket).eq(0)
    definitions.append(("unseen_plant", data[~plant_test].copy(), data[plant_test].copy()))

    bee_test = data["bee"].map(bucket).eq(0)
    definitions.append(("unseen_bee", data[~bee_test].copy(), data[bee_test].copy()))

    site_years = data.groupby(["study", "site"])["year"].nunique()
    eligible_sites = set(site_years[site_years.ge(2)].index)
    eligible = pd.MultiIndex.from_frame(data[["study", "site"]]).isin(eligible_sites)
    latest_year = data.groupby(["study", "site"])["year"].transform("max")
    future_test = eligible & data["year"].eq(latest_year)
    definitions.append(("future_year", data[~future_test].copy(), data[future_test].copy()))

    # Complete leave-study-out predictions: each study is tested exactly once.
    for study in sorted(data["study"].unique()):
        mask = data["study"].eq(study)
        definitions.append((f"unseen_study::{study}", data[~mask].copy(), data[mask].copy()))
    return definitions


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA / "temporal_joint_availability_candidates.csv.gz", parse_dates=["date"])
    external = pd.read_csv(DATA / "north_american_direct_pairs.csv.gz")
    metric_rows, prediction_parts, gate_rows = [], [], []
    for name, train, test in split_definitions(data):
        print(f"{name}: train={len(train):,}, test={len(test):,}", flush=True)
        rows, predictions, gates = evaluate(name, train, test, external)
        metric_rows.extend(rows)
        prediction_parts.append(predictions)
        gate_rows.extend(gates)

    all_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    # Aggregate the per-study results without allowing large studies to hide failures in small studies.
    study = all_metrics[all_metrics["split"].str.startswith("unseen_study::")].copy()
    aggregate_rows = []
    for model, subset in study.groupby("model"):
        aggregate_rows.append({
            "split": "unseen_study_macro_average",
            "model": model,
            "rows": int(subset["rows"].sum()),
            "events": int(subset["events"].sum()),
            "positive_rate": float(subset["positive_rate"].mean()),
            **{column: float(subset[column].mean()) for column in [
                "average_precision", "roc_auc", "brier", "log_loss",
                "calibration_error", "event_recall_at_top_10pct",
            ]},
        })
    all_metrics = pd.concat([all_metrics, pd.DataFrame(aggregate_rows)], ignore_index=True)
    all_metrics.to_csv(OUT / "interaction_prediction_metrics.csv", index=False)
    predictions.to_csv(OUT / "interaction_test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(gate_rows).to_csv(OUT / "adaptive_model_gates.csv", index=False)

    primary = all_metrics[all_metrics["split"].isin([
        "held_event", "unseen_pair", "unseen_plant", "unseen_bee", "future_year", "unseen_study_macro_average"
    ])]
    winners = primary.loc[primary.groupby("split")["average_precision"].idxmax(), ["split", "model", "average_precision"]]
    summary = {
        "candidate_rows": int(len(data)),
        "observed_rows": int(data["interaction_observed"].sum()),
        "studies": int(data["study"].nunique()),
        "models": sorted(all_metrics["model"].unique()),
        "primary_splits": sorted(primary["split"].unique()),
        "average_precision_winners": winners.to_dict(orient="records"),
        "primary_model": "adaptive_context_graph",
        "archive_augmented_status": (
            "sensitivity model only because the external archive may contain records from evaluation studies"
        ),
        "outcome_definition": (
            "Probability that an interaction is recorded conditional on both taxa appearing in the same study-site-date event"
        ),
        "zero_definition": "same-event non-detection, not biological incompatibility",
    }
    (OUT / "interaction_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
