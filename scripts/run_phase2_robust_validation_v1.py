"""Leakage-safe robustness validation for the Phase 2 mixture decision engine.

Two outer validation designs are run:
1. Rolling origin: for each modern test year, the link models use only records
   from earlier calendar years, then recommend mixtures for sites observed that
   year.
2. Leave-one-study-out: the learned link models are trained without any labels
   from one entire study. Earlier years at a held study site remain available
   only to construct the site's known plant/bee timing and strict local links.

Both designs retain the three Phase 2 link layers and the same Pareto-balanced
mixture optimizer. They measure future recorded-interaction coverage, not a
field, health, abundance, pollination, or yield outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_phase2_three_network_decision_v1 as phase2
import run_decision_backtest_v14 as decision


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "factorized_model_v15" / "factorized_event_records.csv.gz"
OUT = ROOT / "results" / "phase2_robust_validation_v1"
MIXTURE_SIZES = (3, 5, 7)
RANDOM_REPLICATES = 200
MODERN_FIRST_TEST_YEAR = 2000
SEED = 20260819


def valid_case(train: pd.DataFrame, test: pd.DataFrame, k: int) -> bool:
    positives = train.loc[train.interaction_observed.eq(1)]
    return (
        positives.plant.nunique() >= k
        and positives.bee.nunique() >= 3
        and train.event_id.nunique() >= 3
        and test.event_id.nunique() >= 3
    )


def evaluate_case(
    layers: phase2.LinkLayers, train: pd.DataFrame, test: pd.DataFrame,
    study: str, site: str, year: int, design: str, outer_fold: str,
    bee_profiles: pd.DataFrame, plant_profiles: pd.DataFrame, rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    positives = train.loc[train.interaction_observed.eq(1)]
    plants, bees = sorted(positives.plant.unique()), sorted(positives.bee.unique())
    if not valid_case(train, test, max(MIXTURE_SIZES)):
        return [], []
    candidate, available = phase2.synthetic_candidates(
        train, study, site, year, bees, plants, bee_profiles, plant_profiles,
    )
    probabilities = layers.predict(candidate)
    node = candidate.bee + "|" + candidate.week.astype(str)
    site_key = f"{study}|{site}"
    rows, selections = [], []
    for method, probability in probabilities.items():
        scores = pd.DataFrame({"node": node, "plant": candidate.plant, "score": probability * available})
        for k in MIXTURE_SIZES:
            mixture, diagnostics = phase2.choose_pareto_balanced(scores, plants, k)
            rows.append({
                "validation_design": design, "outer_fold": outer_fold,
                "site_key": site_key, "study": study, "site": site, "held_out_year": year,
                "method": method, "mixture_size": k, "mixture": ";".join(mixture),
                "candidate_plants": len(plants), "training_bees": len(bees),
                **diagnostics, **decision.evaluate_mixture(test, mixture),
            })
            selections.extend({
                "validation_design": design, "outer_fold": outer_fold,
                "site_key": site_key, "held_out_year": year, "method": method,
                "mixture_size": k, "rank": rank, "plant": plant,
            } for rank, plant in enumerate(mixture, 1))
    for k in MIXTURE_SIZES:
        random_scores = [
            decision.evaluate_mixture(test, sorted(rng.choice(plants, size=k, replace=False)))
            for _ in range(RANDOM_REPLICATES)
        ]
        rows.append({
            "validation_design": design, "outer_fold": outer_fold,
            "site_key": site_key, "study": study, "site": site, "held_out_year": year,
            "method": "random_mean", "mixture_size": k, "mixture": f"{RANDOM_REPLICATES} random mixtures",
            "candidate_plants": len(plants), "training_bees": len(bees),
            "pareto_frontier_size": np.nan, "training_mean_predicted_support": np.nan,
            "training_lower_decile_bee_support": np.nan,
            **{key: float(np.mean([score[key] for score in random_scores])) for key in random_scores[0]},
        })
    return rows, selections


def rolling_cases(data: pd.DataFrame) -> dict[int, list[tuple[str, str, pd.DataFrame, pd.DataFrame]]]:
    result: dict[int, list[tuple[str, str, pd.DataFrame, pd.DataFrame]]] = {}
    for (study, site), local in data.groupby(["study", "site"], sort=True):
        for year in sorted(local.year.unique()):
            if year < MODERN_FIRST_TEST_YEAR:
                continue
            train, test = local.loc[local.year.lt(year)].copy(), local.loc[local.year.eq(year)].copy()
            if valid_case(train, test, max(MIXTURE_SIZES)):
                result.setdefault(int(year), []).append((str(study), str(site), train, test))
    return result


def final_year_study_cases(data: pd.DataFrame, study: str) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame]]:
    cases = []
    for (_, site), local in data.loc[data.study.eq(study)].groupby(["study", "site"], sort=True):
        year = int(local.year.max())
        train, test = local.loc[local.year.lt(year)].copy(), local.loc[local.year.eq(year)].copy()
        if valid_case(train, test, max(MIXTURE_SIZES)):
            cases.append((str(study), str(site), train, test))
    return cases


def persist(rows: list[dict], selections: list[dict], completed: list[str]) -> None:
    pd.DataFrame(rows).to_csv(OUT / "metrics_partial.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_mixtures_partial.csv", index=False)
    (OUT / "progress.json").write_text(json.dumps({"completed_outer_folds": completed}, indent=2), encoding="utf-8")


def paired_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["frequency_coverage", "bee_coverage", "week_coverage", "bee_week_coverage"]
    methods = ["catboost_supported_links", "two_tower_supported_links"]
    baselines = ["strict_documented_training_links", "random_mean"]
    rng = np.random.default_rng(SEED)
    rows = []
    for (design, k), current in frame.groupby(["validation_design", "mixture_size"]):
        # A site can occur in several rolling-origin years; retain site-year as
        # the paired unit so each future decision has a matching baseline.
        unit = current.site_key + "|" + current.held_out_year.astype(str)
        current = current.assign(unit=unit)
        for method in methods:
            for baseline in baselines:
                for metric in metrics:
                    paired = current.loc[current.method.isin([method, baseline])].pivot(
                        index="unit", columns="method", values=metric
                    ).dropna()
                    delta = paired[method].to_numpy() - paired[baseline].to_numpy()
                    boot = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10_000)])
                    rows.append({
                        "validation_design": design, "mixture_size": int(k), "method": method,
                        "baseline": baseline, "metric": metric, "paired_decisions": int(len(delta)),
                        "mean_difference": float(delta.mean()),
                        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                        "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                        "losses": int((delta < 0).sum()),
                    })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["date"])
    bee_profiles, plant_profiles = phase2.make_profiles(data)
    rows: list[dict] = []
    selections: list[dict] = []
    completed: list[str] = []
    rng = np.random.default_rng(SEED)

    # Temporal rolling-origin evaluation.  Each model only sees older years.
    rolling = rolling_cases(data)
    for year in sorted(rolling):
        cases = rolling[year]
        outer = data.loc[data.year.lt(year)].copy()
        if outer.interaction_observed.sum() < 100 or outer.event_id.nunique() < 20:
            continue
        layers = phase2.LinkLayers(outer)
        for study, site, train, test in cases:
            values, chosen = evaluate_case(
                layers, train, test, study, site, year, "rolling_origin", f"year_{year}",
                bee_profiles, plant_profiles, rng,
            )
            rows.extend(values)
            selections.extend(chosen)
        completed.append(f"rolling_year_{year}")
        persist(rows, selections, completed)

    # Cross-study transfer.  Link learners do not see any label from the held
    # study, while the study's prior local timing remains a deployment-realistic
    # input to its next-year mixture decision.
    for study in sorted(data.study.unique()):
        cases = final_year_study_cases(data, str(study))
        if not cases:
            continue
        outer = data.loc[~data.study.eq(study)].copy()
        layers = phase2.LinkLayers(outer)
        for case_study, site, train, test in cases:
            values, chosen = evaluate_case(
                layers, train, test, case_study, site, int(test.year.iloc[0]),
                "leave_one_study_out", f"study_{study}", bee_profiles, plant_profiles, rng,
            )
            rows.extend(values)
            selections.extend(chosen)
        completed.append(f"leave_study_{study}")
        persist(rows, selections, completed)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "robust_validation_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_mixtures.csv", index=False)
    metrics = ["frequency_coverage", "bee_coverage", "week_coverage", "bee_week_coverage"]
    aggregate = result.groupby(["validation_design", "mixture_size", "method"])[metrics].mean().reset_index()
    aggregate.to_csv(OUT / "robust_validation_summary.csv", index=False)
    paired_summary(result).to_csv(OUT / "paired_bootstrap_comparisons.csv", index=False)
    summary = {
        "validation_designs": {
            "rolling_origin": "The model and all learned link evidence precede the held-out calendar year.",
            "leave_one_study_out": "No interaction labels from the held-out study enter CatBoost or two-tower training; earlier local timing supports remain available for the decision.",
        },
        "link_layers": ["strict documented training links", "full-feature CatBoost", "trait two-tower neural model"],
        "mixture_sizes": list(MIXTURE_SIZES),
        "not_claimed": "Field benefit, bee health, abundance, pollination, or crop yield.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
