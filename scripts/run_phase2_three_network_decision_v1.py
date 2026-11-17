"""Phase 2 historical mixture-decision backtest with three link layers.

The experiment holds out the last year at each eligible study-site.  A common
forage-selection routine then chooses mixtures using one of three frozen link
layers: direct training evidence, the Phase 1 CatBoost model, or the Phase 1
two-tower trait model.  Plant and bee timing are deliberately held constant
across the three alternatives, so a performance difference is attributable to
the link layer rather than a different candidate pool.

This evaluates recovery of later *recorded* interactions conditional on the
study's sampling design.  It does not establish bee health, abundance, crop
pollination, yield, or a causal planting benefit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import run_decision_backtest_v14 as decision
import run_factorized_ai_experiment_v15 as tower
import run_phase1_shared_baselines_v1 as catboost_base


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "factorized_model_v15" / "factorized_event_records.csv.gz"
OUT = ROOT / "results" / "phase2_three_network_decision_v1"
SEED = 20260819
MIXTURE_SIZES = (3, 5, 7)
RANDOM_REPLICATES = 200


def calibrate(y: np.ndarray, raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calibrate logits/probabilities using only the internal calibration events."""
    if len(np.unique(y)) < 2:
        return 1.0 / (1.0 + np.exp(-target))
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    model.fit(raw.reshape(-1, 1), y)
    return model.predict_proba(target.reshape(-1, 1))[:, 1]


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


class LinkLayers:
    """Fit Phase 1 winners once on the common historical training pool."""

    def __init__(self, train: pd.DataFrame):
        internal = train.event_id.map(lambda value: tower.bucket(str(value), 7))
        calibration = train.loc[internal.eq(1)].copy()
        fit = train.loc[~internal.isin([0, 1])].copy()
        self.strict_pairs = set(map(tuple, fit.loc[fit.interaction_observed.eq(1), ["bee", "plant"]].to_numpy()))

        # Phase 1 CatBoost feature specification and calibration protocol.
        self.catboost = CatBoostClassifier(
            iterations=300, depth=7, learning_rate=0.04, loss_function="Logloss",
            verbose=False, random_seed=SEED, l2_leaf_reg=8, allow_writing_files=False,
        )
        self.catboost.fit(
            catboost_base.feature(fit), fit.interaction_observed,
            cat_features=catboost_base.CATS,
        )
        cat_cal = self.catboost.predict_proba(catboost_base.feature(calibration))[:, 1]
        self.cat_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED).fit(
            logit(cat_cal).reshape(-1, 1), calibration.interaction_observed.to_numpy(int)
        )

        # The Phase 1 two-tower model used a two-member ensemble.  Preserve that
        # architecture and average member logits before calibration.
        self.tower_encoder = tower.Encoder(fit, use_traits=True)
        validation = train.loc[internal.eq(0)].copy()
        self.tower_models = [
            tower.train_one(self.tower_encoder, fit, validation, SEED + member)
            for member in range(tower.ENSEMBLE_SIZE)
        ]
        calibration_logits = np.mean(
            [tower.predict_model(model, self.tower_encoder, calibration)[0] for model in self.tower_models], axis=0
        )
        self.tower_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED).fit(
            calibration_logits.reshape(-1, 1), calibration.interaction_observed.to_numpy(int)
        )

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        strict = np.asarray([
            1.0 if (bee, plant) in self.strict_pairs else 0.0
            for bee, plant in zip(frame.bee, frame.plant)
        ])
        cat_raw = self.catboost.predict_proba(catboost_base.feature(frame))[:, 1]
        cat = self.cat_calibrator.predict_proba(logit(cat_raw).reshape(-1, 1))[:, 1]
        tower_logits = np.mean(
            [tower.predict_model(model, self.tower_encoder, frame)[0] for model in self.tower_models], axis=0
        )
        two_tower = self.tower_calibrator.predict_proba(tower_logits.reshape(-1, 1))[:, 1]
        return {
            "strict_documented_training_links": strict,
            "catboost_supported_links": cat,
            "two_tower_supported_links": two_tower,
        }


def make_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bee_cols = ["bee"] + tower.BEE_CATS + tower.BEE_FLAGS
    plant_cols = ["plant"] + tower.PLANT_CATS + tower.PLANT_TRAITS + tower.PLANT_FLAGS + [
        f"{column}_missing" for column in tower.PLANT_TRAITS
    ]
    bees = data[bee_cols].drop_duplicates("bee")
    plants = data[plant_cols].drop_duplicates("plant")
    return bees, plants


def synthetic_candidates(
    train: pd.DataFrame, study: str, site: str, year: int,
    bees: list[str], plants: list[str], bee_profiles: pd.DataFrame, plant_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create potential bee-week/plant records using only pre-holdout data."""
    bee_weeks = train[["bee", "week"]].drop_duplicates()
    frame = bee_weeks.merge(pd.DataFrame({"plant": plants}), how="cross")
    frame = frame.merge(bee_profiles, on="bee", how="left", validate="many_to_one")
    frame = frame.merge(plant_profiles, on="plant", how="left", validate="many_to_one")
    frame["study"], frame["site"] = study, site
    frame["event_id"] = f"phase2|{study}|{site}"
    frame["sampling_grain"] = train.sampling_grain.mode().iat[0]
    frame["date"] = pd.Timestamp(f"{year}-01-01")
    frame["year"] = year
    frame["interaction_observed"] = 0
    frame["observed_frequency"] = 0.0
    frame["candidate_pairs"] = len(bees) * len(plants)
    frame["bees"] = len(bees)
    frame["plants"] = len(plants)
    frame["study_events"] = train.event_id.nunique()
    frame["study_sites"] = train.site.nunique()
    frame["study_years"] = train.year.nunique()
    frame["week_sin"] = np.sin(2 * np.pi * frame.week / 52.0)
    frame["week_cos"] = np.cos(2 * np.pi * frame.week / 52.0)
    frame["log_event_candidate_pairs"] = np.log1p(len(bees) * len(plants))
    frame["log_study_events"] = np.log1p(train.event_id.nunique())
    frame["log_study_sites"] = np.log1p(train.site.nunique())
    # A plant must have been recorded as available at this site and week in the
    # training years before it can contribute to a selected mixture.
    plant_weeks = set(map(tuple, train[["plant", "week"]].drop_duplicates().to_numpy()))
    available = np.asarray([(plant, week) in plant_weeks for plant, week in zip(frame.plant, frame.week)])
    return frame, available


def choose_pareto_balanced(frame: pd.DataFrame, plants: list[str], k: int) -> tuple[list[str], dict[str, float]]:
    """Generate a small, reproducible approximate Pareto set, then pick balance."""
    matrix = frame.pivot(index="node", columns="plant", values="score").reindex(columns=plants).fillna(0.0)
    bee_for_node = pd.Series(matrix.index.str.rsplit("|", n=1).str[0], index=matrix.index)
    candidate_solutions: list[tuple[list[str], float, float]] = []
    for coverage_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        selected: list[str] = []
        current = np.zeros(len(matrix), dtype=float)
        for _ in range(min(k, len(plants))):
            choices = []
            for plant in plants:
                if plant in selected:
                    continue
                coverage = np.maximum(current, matrix[plant].to_numpy(float))
                per_bee = pd.Series(coverage, index=matrix.index).groupby(bee_for_node).mean()
                mean_support = float(coverage.mean())
                lower_bee_support = float(per_bee.quantile(0.10))
                objective = coverage_weight * mean_support + (1 - coverage_weight) * lower_bee_support
                choices.append((objective, plant, coverage, mean_support, lower_bee_support))
            _, chosen, current, _, _ = max(choices, key=lambda item: (item[0], item[1]))
            selected.append(chosen)
        per_bee = pd.Series(current, index=matrix.index).groupby(bee_for_node).mean()
        candidate_solutions.append((sorted(selected), float(current.mean()), float(per_bee.quantile(0.10))))

    # A mixture is dominated only if another generated mixture is at least as
    # good on average and for the lower-supported bee tail, and strictly better
    # on one.  This is an explicit approximate Pareto frontier, not an arbitrary
    # weighted-score-only choice.
    unique = {tuple(items): (items, mean, lower) for items, mean, lower in candidate_solutions}
    frontier = []
    for solution in unique.values():
        _, mean, lower = solution
        dominated = any(
            (other[1] >= mean and other[2] >= lower and (other[1] > mean or other[2] > lower))
            for other in unique.values()
        )
        if not dominated:
            frontier.append(solution)
    chosen = max(frontier, key=lambda item: ((item[1] + item[2]) / 2, item[0]))
    return chosen[0], {
        "pareto_frontier_size": len(frontier),
        "training_mean_predicted_support": chosen[1],
        "training_lower_decile_bee_support": chosen[2],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["date"])
    bee_profiles, plant_profiles = make_profiles(data)
    site_years = data.groupby(["study", "site"]).year.nunique()
    eligible = set(site_years[site_years.ge(2)].index)
    latest = data.groupby(["study", "site"]).year.transform("max")
    eligible_mask = pd.MultiIndex.from_frame(data[["study", "site"]]).isin(eligible)
    global_train = data.loc[~(eligible_mask & data.year.eq(latest))].copy()
    layers = LinkLayers(global_train)

    rows, selections = [], []
    rng = np.random.default_rng(SEED)
    for study, site in sorted(eligible):
        local = data.loc[data.study.eq(study) & data.site.eq(site)].copy()
        held_year = int(local.year.max())
        train, test = local.loc[local.year.lt(held_year)].copy(), local.loc[local.year.eq(held_year)].copy()
        prior_positive = train.loc[train.interaction_observed.eq(1)]
        plants, bees = sorted(prior_positive.plant.unique()), sorted(prior_positive.bee.unique())
        if len(plants) < 7 or len(bees) < 3 or train.event_id.nunique() < 3 or test.event_id.nunique() < 3:
            continue
        candidate, available = synthetic_candidates(
            train, study, site, held_year, bees, plants, bee_profiles, plant_profiles,
        )
        probabilities = layers.predict(candidate)
        node = candidate.bee + "|" + candidate.week.astype(str)
        site_key = f"{study}|{site}"
        for method, probability in probabilities.items():
            base = pd.DataFrame({"node": node, "plant": candidate.plant, "score": probability * available})
            for k in MIXTURE_SIZES:
                mixture, diagnostics = choose_pareto_balanced(base, plants, k)
                rows.append({
                    "site_key": site_key, "study": study, "site": site, "held_out_year": held_year,
                    "method": method, "mixture_size": k, "mixture": ";".join(mixture),
                    "candidate_plants": len(plants), "training_bees": len(bees),
                    **diagnostics, **decision.evaluate_mixture(test, mixture),
                })
                selections.extend({
                    "site_key": site_key, "held_out_year": held_year, "method": method,
                    "mixture_size": k, "rank": rank, "plant": plant,
                } for rank, plant in enumerate(mixture, 1))
        for k in MIXTURE_SIZES:
            random_scores = [decision.evaluate_mixture(test, sorted(rng.choice(plants, size=k, replace=False))) for _ in range(RANDOM_REPLICATES)]
            rows.append({
                "site_key": site_key, "study": study, "site": site, "held_out_year": held_year,
                "method": "random_mean", "mixture_size": k, "mixture": f"{RANDOM_REPLICATES} random mixtures",
                "candidate_plants": len(plants), "training_bees": len(bees),
                "pareto_frontier_size": np.nan, "training_mean_predicted_support": np.nan,
                "training_lower_decile_bee_support": np.nan,
                **{key: float(np.mean([score[key] for score in random_scores])) for key in random_scores[0]},
            })
        pd.DataFrame(rows).to_csv(OUT / "decision_metrics_partial.csv", index=False)

    result = pd.DataFrame(rows)
    metrics = ["frequency_coverage", "bee_coverage", "week_coverage", "bee_week_coverage"]
    aggregate = result.groupby(["mixture_size", "method"])[metrics].agg(["mean", "median"]).reset_index()
    aggregate.columns = ["mixture_size", "method"] + [f"{metric}_{stat}" for metric, stat in aggregate.columns.tolist()[2:]]
    result.to_csv(OUT / "future_year_decision_metrics.csv", index=False)
    aggregate.to_csv(OUT / "future_year_decision_summary.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_mixtures.csv", index=False)
    summary = {
        "phase": "2 decision-level integration",
        "link_layers": ["strict documented training links", "Phase 1 full-feature CatBoost", "Phase 1 trait two-tower neural network"],
        "timing_layer": "Only plant and bee weeks observed before each study-site's held-out final year are used to construct candidate availability.",
        "optimization": "Approximate Pareto frontier over mean predicted support and lower-decile bee support; balanced frontier solution is reported.",
        "eligible_study_sites": int(result.site_key.nunique()),
        "mixture_sizes": list(MIXTURE_SIZES),
        "evaluation": "Future-year recovery of observed interactions and their bee/week coverage among plants available in prior local years.",
        "not_claimed": "Bee health, bee abundance, nectar/pollen quantity, crop yield, or a causal benefit of planting.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
