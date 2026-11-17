from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import run_interaction_prediction_experiment_v14 as experiment


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "interaction_benchmark_v14"
OUT = ROOT / "results" / "decision_backtest_v14"
SEED = 20260815
MIXTURE_SIZE = 5
RANDOM_REPLICATES = 200


class Calibrator:
    def __init__(self, y: np.ndarray, probability: np.ndarray):
        self.model = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
        self.model.fit(experiment.logit(probability).reshape(-1, 1), y)

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(experiment.logit(probability).reshape(-1, 1))[:, 1]


class FuturePredictor:
    def __init__(self, train: pd.DataFrame, external: pd.DataFrame):
        internal = train["event_id"].map(lambda value: experiment.bucket(str(value), modulus=7))
        self.gate = train[internal.eq(0)].copy()
        calibration = train[internal.eq(1)].copy()
        fit = train[~internal.isin([0, 1])].copy()
        self.builder = experiment.FeatureBuilder(fit, external)
        x_fit = self.builder.transform(fit)
        x_cal = self.builder.transform(calibration)
        x_gate = self.builder.transform(self.gate)
        y_fit = fit["interaction_observed"].to_numpy(int)
        y_cal = calibration["interaction_observed"].to_numpy(int)
        self.models = {}
        self.calibrators = {}
        self.gate_probabilities = {}

        popularity_cal = np.sqrt(x_cal["bee_rate"] * x_cal["plant_rate"])
        self.calibrators["popularity"] = Calibrator(y_cal, popularity_cal.to_numpy())
        popularity_gate = np.sqrt(x_gate["bee_rate"] * x_gate["plant_rate"]).to_numpy()
        self.gate_probabilities["popularity"] = self.calibrators["popularity"].predict(popularity_gate)

        for name, columns in experiment.FEATURES.items():
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=5.0, random_state=SEED,
            ).fit(x_fit[columns], y_fit)
            self.models[name] = model
            cal_raw = model.predict_proba(x_cal[columns])[:, 1]
            self.calibrators[name] = Calibrator(y_cal, cal_raw)
            gate_raw = model.predict_proba(x_gate[columns])[:, 1]
            self.gate_probabilities[name] = self.calibrators[name].predict(gate_raw)

        self.choices = self._select_gates()

    def _select_gates(self) -> dict[str, str]:
        allowed = ["popularity", "taxonomy_season", "temporal_graph", "hierarchical_context"]
        strata = self.builder.evidence_stratum(self.gate)
        y = self.gate["interaction_observed"].to_numpy(int)
        choices = {}
        for stratum in sorted(set(strata)):
            mask = strata == stratum
            if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
                choices[stratum] = "taxonomy_season"
                continue
            brier = {
                name: float(np.mean((self.gate_probabilities[name][mask] - y[mask]) ** 2))
                for name in allowed
            }
            choices[stratum] = min(brier, key=brier.get)
        return choices

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        features = self.builder.transform(frame)
        result = {}
        raw_popularity = np.sqrt(features["bee_rate"] * features["plant_rate"]).to_numpy()
        result["popularity"] = self.calibrators["popularity"].predict(raw_popularity)
        for name, columns in experiment.FEATURES.items():
            raw = self.models[name].predict_proba(features[columns])[:, 1]
            result[name] = self.calibrators[name].predict(raw)
        strata = self.builder.evidence_stratum(frame)
        adaptive = np.empty(len(frame), dtype=float)
        for stratum in np.unique(strata):
            choice = self.choices.get(stratum, "taxonomy_season")
            mask = strata == stratum
            adaptive[mask] = result[choice][mask]
        result["adaptive_context_graph"] = adaptive
        return result


def greedy_mixture(scores: pd.DataFrame, plants: list[str]) -> list[str]:
    matrix = scores.pivot(index="node", columns="plant", values="probability").reindex(columns=plants).fillna(0)
    bee_by_node = pd.Series(matrix.index.str.rsplit("|", n=1).str[0], index=matrix.index)
    selected: list[str] = []
    current = np.zeros(len(matrix))
    for _ in range(min(MIXTURE_SIZE, len(plants))):
        candidates = []
        for plant in plants:
            if plant in selected:
                continue
            coverage = np.maximum(current, matrix[plant].to_numpy(float))
            bee_means = pd.Series(coverage, index=matrix.index).groupby(bee_by_node).mean()
            objective = 0.6 * coverage.mean() + 0.4 * bee_means.quantile(0.1)
            candidates.append((objective, plant, coverage))
        _, winner, current = max(candidates, key=lambda item: (item[0], item[1]))
        selected.append(winner)
    return sorted(selected)


def evaluate_mixture(test: pd.DataFrame, mixture: list[str]) -> dict[str, float]:
    observed = test[test["interaction_observed"].eq(1)].copy()
    covered = observed["plant"].isin(mixture)
    total_frequency = observed["observed_frequency"].sum()
    bee_total = observed["bee"].nunique()
    week_total = observed["week"].nunique()
    bee_week_total = observed[["bee", "week"]].drop_duplicates().shape[0]
    return {
        "frequency_coverage": float(observed.loc[covered, "observed_frequency"].sum() / total_frequency) if total_frequency else 0.0,
        "bee_coverage": float(observed.loc[covered, "bee"].nunique() / bee_total) if bee_total else 0.0,
        "week_coverage": float(observed.loc[covered, "week"].nunique() / week_total) if week_total else 0.0,
        "bee_week_coverage": float(
            observed.loc[covered, ["bee", "week"]].drop_duplicates().shape[0] / bee_week_total
        ) if bee_week_total else 0.0,
    }


def bootstrap_difference(frame: pd.DataFrame, model: str, baseline: str, metric: str) -> tuple[float, float, float]:
    pivot = frame[frame["method"].isin([model, baseline])].pivot(index="site_key", columns="method", values=metric).dropna()
    difference = pivot[model].to_numpy() - pivot[baseline].to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.asarray([
        rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(5000)
    ])
    return float(difference.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA / "temporal_joint_availability_candidates.csv.gz", parse_dates=["date"])
    external = pd.read_csv(DATA / "north_american_direct_pairs.csv.gz")
    site_years = data.groupby(["study", "site"])["year"].nunique()
    eligible_site_keys = set(site_years[site_years.ge(2)].index)
    eligible = pd.MultiIndex.from_frame(data[["study", "site"]]).isin(eligible_site_keys)
    latest = data.groupby(["study", "site"])["year"].transform("max")
    test_mask = eligible & data["year"].eq(latest)
    global_train = data[~test_mask].copy()
    predictor = FuturePredictor(global_train, external)
    taxonomy = data.drop_duplicates("bee").set_index("bee")[["bee_genus", "bee_tribe", "bee_family"]]
    plant_taxonomy = data.drop_duplicates("plant").set_index("plant")[["plant_genus", "plant_family"]]
    methods = [
        "popularity", "taxonomy_season", "temporal_graph", "hierarchical_context",
        "adaptive_context_graph", "archive_augmented",
    ]
    rows = []
    rng = np.random.default_rng(SEED)

    for study, site in sorted(eligible_site_keys):
        local = data[data["study"].eq(study) & data["site"].eq(site)]
        final_year = int(local["year"].max())
        train = local[local["year"].lt(final_year)]
        test = local[local["year"].eq(final_year)]
        positive_train = train[train["interaction_observed"].eq(1)]
        plants = sorted(positive_train["plant"].unique())
        bees = sorted(positive_train["bee"].unique())
        if len(plants) < 6 or len(bees) < 3 or train["event_id"].nunique() < 3 or test["event_id"].nunique() < 3:
            continue

        bee_weeks = train[["bee", "week"]].drop_duplicates()
        plant_weeks = set(zip(train["plant"], train["week"]))
        synthetic = bee_weeks.merge(pd.DataFrame({"plant": plants}), how="cross")
        synthetic["study"] = study
        synthetic["site"] = site
        synthetic["event_id"] = "decision|" + study + "|" + site
        synthetic["date"] = pd.Timestamp(f"{final_year}-01-01")
        synthetic["year"] = final_year
        synthetic["interaction_observed"] = 0
        synthetic["observed_frequency"] = 0.0
        synthetic = synthetic.merge(taxonomy, on="bee", how="left", validate="many_to_one")
        synthetic = synthetic.merge(plant_taxonomy, on="plant", how="left", validate="many_to_one")
        probabilities = predictor.predict(synthetic)
        node = synthetic["bee"] + "|" + synthetic["week"].astype(str)
        active = np.asarray([(plant, week) in plant_weeks for plant, week in zip(synthetic["plant"], synthetic["week"])])

        mixtures = {}
        for method in methods:
            score_frame = pd.DataFrame({
                "node": node,
                "plant": synthetic["plant"],
                "probability": probabilities[method] * active,
            })
            mixtures[method] = greedy_mixture(score_frame, plants)
        frequency = positive_train.groupby("plant")["observed_frequency"].sum().reindex(plants, fill_value=0)
        mixtures["training_frequency"] = sorted(frequency.nlargest(MIXTURE_SIZE).index)
        oracle_frequency = test[test["interaction_observed"].eq(1)].groupby("plant")["observed_frequency"].sum().reindex(plants, fill_value=0)
        mixtures["oracle_ceiling"] = sorted(oracle_frequency.nlargest(MIXTURE_SIZE).index)

        site_key = f"{study}|{site}"
        for method, mixture in mixtures.items():
            rows.append({
                "site_key": site_key, "study": study, "site": site,
                "test_year": final_year, "method": method, "mixture": ";".join(mixture),
                "candidate_plants": len(plants), "training_bees": len(bees),
                "training_events": int(train["event_id"].nunique()),
                "test_events": int(test["event_id"].nunique()),
                **evaluate_mixture(test, mixture),
            })

        random_metrics = []
        for _ in range(RANDOM_REPLICATES):
            mixture = sorted(rng.choice(plants, size=MIXTURE_SIZE, replace=False))
            random_metrics.append(evaluate_mixture(test, mixture))
        rows.append({
            "site_key": site_key, "study": study, "site": site,
            "test_year": final_year, "method": "random_mean", "mixture": "200 random mixtures",
            "candidate_plants": len(plants), "training_bees": len(bees),
            "training_events": int(train["event_id"].nunique()),
            "test_events": int(test["event_id"].nunique()),
            **{key: float(np.mean([item[key] for item in random_metrics])) for key in random_metrics[0]},
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "future_year_mixture_backtest.csv", index=False)
    metrics = ["frequency_coverage", "bee_coverage", "week_coverage", "bee_week_coverage"]
    aggregate = results.groupby("method")[metrics].agg(["mean", "median"]).reset_index()
    aggregate.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate.columns.tolist()[1:]]
    aggregate.to_csv(OUT / "future_year_mixture_metrics.csv", index=False)

    comparisons = []
    for baseline in [
        "random_mean", "training_frequency", "popularity", "taxonomy_season",
        "temporal_graph", "hierarchical_context",
    ]:
        for metric in metrics:
            mean, low, high = bootstrap_difference(results, "adaptive_context_graph", baseline, metric)
            pivot = results[results["method"].isin(["adaptive_context_graph", baseline])].pivot(
                index="site_key", columns="method", values=metric
            ).dropna()
            difference = pivot["adaptive_context_graph"] - pivot[baseline]
            statistic, pvalue = wilcoxon(difference) if not np.allclose(difference, 0) else (0.0, 1.0)
            comparisons.append({
                "model": "adaptive_context_graph", "baseline": baseline, "metric": metric,
                "paired_sites": len(difference), "mean_difference": mean,
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "wilcoxon_statistic": float(statistic), "wilcoxon_pvalue_exploratory": float(pvalue),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(OUT / "paired_method_comparisons.csv", index=False)

    ranking = aggregate.sort_values("bee_week_coverage_mean", ascending=False)
    summary = {
        "eligible_future_sites": int(results["site_key"].nunique()),
        "mixture_size": MIXTURE_SIZE,
        "methods": sorted(results["method"].unique()),
        "best_non_oracle_mean_bee_week_coverage_method": str(
            ranking[~ranking["method"].eq("oracle_ceiling")].iloc[0]["method"]
        ),
        "adaptive_context_graph_mean_metrics": aggregate[
            aggregate["method"].eq("adaptive_context_graph")
        ].iloc[0].to_dict(),
        "evaluation_boundary": (
            "Retrospective recovery of held-out observed interactions among plants documented in earlier local years; "
            "not a field intervention or health/yield outcome."
        ),
        "oracle_status": "uses held-out outcomes and is reported only as an unattainable ceiling",
        "archive_augmented_status": "sensitivity only; external archive can overlap evaluation studies",
    }
    (OUT / "decision_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nAggregate metrics:\n", aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
