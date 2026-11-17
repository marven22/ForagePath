"""Compute paper-ready paired 95% bootstrap intervals from frozen test outputs.

This script never changes a trained model.  It evaluates the four model
families selected for the manuscript using their appropriate independent unit:
site-year (flowering), bee species (bee activity), and study (link ranking).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_uncertainty_intervals_v1"
RNG = np.random.default_rng(20260912)
N_BOOT = 1000


def ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def weighted_brier_fraction(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Brier score for the flowering check-fraction outcome used in the benchmark."""
    return float(np.average((y - p) ** 2, weights=w))


def flowering() -> pd.DataFrame:
    data = pd.read_parquet(ROOT / "data" / "snapshots" / "us-weekly-flowering-2009-2025-v3" / "derived" / "weekly_flowering_modeling.parquet")
    data = data.loc[data.data_split.eq("test") & data.weather_available].copy()
    # The paper's flowering comparison is the selected final CatBoost model
    # against the pre-specified seasonal climatology baseline.
    keep = ["seasonal_climatology", "catboost_full_features_corrected_ecoregion"]
    pred = pd.read_parquet(ROOT / "results" / "flowering_established_model_benchmark_v3" / "test_predictions.parquet")
    pred = pred.loc[pred.model.isin(keep)]
    wide = pred.pivot(index="weekly_record_id", columns="model", values="probability").reset_index()
    frame = data[["weekly_record_id", "site_id", "year", "flowering_check_fraction", "observation_checks"]].merge(wide, on="weekly_record_id", validate="one_to_one")
    frame["block"] = frame.site_id.astype(str) + "|" + frame.year.astype(str)
    # Brier loss is additive.  Aggregate it once by site-year, then bootstrap
    # those blocks; this is mathematically equivalent to rebuilding row samples
    # on every replicate and is substantially faster.
    block = frame.groupby("block", sort=False)["observation_checks"].sum().to_frame("weight")
    for model in keep:
        block[model] = frame.assign(_loss=frame.observation_checks * (frame.flowering_check_fraction - frame[model]) ** 2).groupby("block", sort=False)['_loss'].sum()
    block_ids = np.arange(len(block))
    weights = block.weight.to_numpy()
    rows = []
    baseline = "seasonal_climatology"
    for model in keep:
        draws, deltas = [], []
        for _ in range(N_BOOT):
            sample = RNG.choice(block_ids, size=len(block_ids), replace=True)
            denominator = weights[sample].sum()
            score = float(block[model].to_numpy()[sample].sum() / denominator)
            draws.append(score)
            if model != baseline:
                base = float(block[baseline].to_numpy()[sample].sum() / denominator)
                deltas.append(score - base)
        point = weighted_brier_fraction(frame.flowering_check_fraction.to_numpy(), frame[model].to_numpy(), frame.observation_checks.to_numpy())
        low, high = ci(np.asarray(draws))
        rows.append({"family":"flowering", "unit":"site-year", "model":model, "baseline":None,
                     "metric":"weighted_brier", "point_estimate":point, "difference":None,
                     "ci95_low":low, "ci95_high":high, "n_units":len(block_ids), "bootstrap_replicates":N_BOOT})
        if model != baseline:
            point_delta = point - weighted_brier_fraction(frame.flowering_check_fraction.to_numpy(), frame[baseline].to_numpy(), frame.observation_checks.to_numpy())
            low, high = ci(np.asarray(deltas))
            rows.append({"family":"flowering", "unit":"site-year", "model":model, "baseline":baseline,
                         "metric":"weighted_brier_difference", "point_estimate":None, "difference":point_delta,
                         "ci95_low":low, "ci95_high":high, "n_units":len(block_ids), "bootstrap_replicates":N_BOOT})
    # Discrimination is secondary because the fitted target is a flowering
    # fraction, but report its uncertainty for reviewers who prefer binary
    # flowering status summaries.
    binary = frame.assign(binary_label=data.set_index("weekly_record_id").loc[frame.weekly_record_id, "flowering_observed"].to_numpy(int))
    by_block = {name: group for name, group in binary.groupby("block", sort=False)}
    for metric_name, scorer in (("roc_auc", roc_auc_score), ("average_precision", average_precision_score)):
        values = {model: [] for model in keep}
        for _ in range(N_BOOT):
            chosen = RNG.choice(block.index.to_numpy(), size=len(block), replace=True)
            boot = pd.concat([by_block[name] for name in chosen], ignore_index=True)
            y = boot.binary_label.to_numpy()
            for model in keep:
                values[model].append(float(scorer(y, boot[model].to_numpy(), sample_weight=boot.observation_checks.to_numpy())))
        for model in keep:
            low, high = ci(np.asarray(values[model]))
            point = float(scorer(binary.binary_label, binary[model], sample_weight=binary.observation_checks))
            rows.append({"family":"flowering", "unit":"site-year", "model":model, "baseline":None,
                         "metric":f"weighted_{metric_name}", "point_estimate":point, "difference":None,
                         "ci95_low":low, "ci95_high":high, "n_units":len(block_ids), "bootstrap_replicates":N_BOOT})
            if model != baseline:
                diff = np.asarray(values[model]) - np.asarray(values[baseline])
                low, high = ci(diff)
                rows.append({"family":"flowering", "unit":"site-year", "model":model, "baseline":baseline,
                             "metric":f"weighted_{metric_name}_difference", "point_estimate":None,
                             "difference":point - float(scorer(binary.binary_label, binary[baseline], sample_weight=binary.observation_checks)),
                             "ci95_low":low, "ci95_high":high, "n_units":len(block_ids), "bootstrap_replicates":N_BOOT})
    return pd.DataFrame(rows)


def bee_activity() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results" / "national_bee_activity_model_benchmark_v1" / "final_test_metrics_by_bee.csv")
    # Spatial pooling emits one row per bee--cell.  Collapse those rows back to
    # a bee-level score before species-level resampling.
    frame = frame.groupby(["model", "bee_key"], as_index=False).apply(
        lambda g: pd.Series({
            "cross_entropy": np.average(g.cross_entropy, weights=g.test_records),
            "top10_mass": np.average(g.top10_mass, weights=g.test_records),
        }), include_groups=False
    ).reset_index(drop=True)
    models = sorted(frame.model.unique())
    baseline = "raw_weekly_histogram"
    rows = []
    for metric, direction in [("cross_entropy", -1), ("top10_mass", 1)]:
        wide = frame.pivot(index="bee_key", columns="model", values=metric).dropna()
        bees = wide.index.to_numpy()
        for model in models:
            draws = []
            for _ in range(N_BOOT):
                chosen = RNG.choice(bees, size=len(bees), replace=True)
                draws.append(float(wide.loc[chosen, model].mean()))
            point = float(wide[model].mean())
            low, high = ci(np.asarray(draws))
            rows.append({"family":"bee_activity", "unit":"bee_species", "model":model, "baseline":None,
                         "metric":f"macro_{metric}", "point_estimate":point, "difference":None,
                         "ci95_low":low, "ci95_high":high, "n_units":len(bees), "bootstrap_replicates":N_BOOT})
            if model != baseline:
                deltas = []
                for _ in range(N_BOOT):
                    chosen = RNG.choice(bees, size=len(bees), replace=True)
                    deltas.append(float((wide.loc[chosen, model] - wide.loc[chosen, baseline]).mean()))
                point_delta = float((wide[model] - wide[baseline]).mean())
                low, high = ci(np.asarray(deltas))
                rows.append({"family":"bee_activity", "unit":"bee_species", "model":model, "baseline":baseline,
                             "metric":f"macro_{metric}_difference", "point_estimate":None, "difference":point_delta,
                             "ci95_low":low, "ci95_high":high, "n_units":len(bees), "bootstrap_replicates":N_BOOT,
                             "better_direction":"lower" if direction < 0 else "higher"})
    return pd.DataFrame(rows)


def history_windows() -> pd.DataFrame:
    base = ROOT / "results" / "national_bee_history_window_sensitivity_v1"
    frames = []
    for start in (2009, 2011, 2013, 2015):
        x = pd.read_csv(base / f"train_{start}_2022" / "final_test_metrics_by_bee.csv")
        x["training_start"] = start
        frames.append(x)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.groupby(["training_start", "model", "bee_key"], as_index=False).apply(
        lambda g: pd.Series({
            "cross_entropy": np.average(g.cross_entropy, weights=g.test_records),
            "top10_mass": np.average(g.top10_mass, weights=g.test_records),
        }), include_groups=False
    ).reset_index(drop=True)
    rows = []
    for model in sorted(frame.model.unique()):
        for metric in ("cross_entropy", "top10_mass"):
            pivot = frame.loc[frame.model.eq(model)].pivot(index="bee_key", columns="training_start", values=metric).dropna()
            bees = pivot.index.to_numpy()
            for start in (2009, 2011, 2013):
                deltas = []
                for _ in range(N_BOOT):
                    chosen = RNG.choice(bees, size=len(bees), replace=True)
                    deltas.append(float((pivot.loc[chosen, start] - pivot.loc[chosen, 2015]).mean()))
                point = float((pivot[start] - pivot[2015]).mean())
                low, high = ci(np.asarray(deltas))
                rows.append({"family":"bee_history", "unit":"bee_species", "model":model, "baseline":"train_2015_2022",
                             "metric":f"macro_{metric}_difference", "training_window":f"{start}-2022", "difference":point,
                             "ci95_low":low, "ci95_high":high, "n_units":len(bees), "bootstrap_replicates":N_BOOT})
    return pd.DataFrame(rows)


def phase1_links() -> pd.DataFrame:
    base = ROOT / "results" / "phase1_v16_appalachian_all"
    baseline_base = base / "baselines"
    base_keys = ["split", "event_id", "study", "interaction_observed"]
    keys = base_keys + ["candidate_rank"]
    a = pd.read_csv(baseline_base / "predictions.csv.gz")[base_keys + ["p_strict_training_evidence", "p_degree_product_heuristic", "p_full_feature_catboost"]]
    b = pd.read_csv(base / "two_tower" / "factorized_ai_predictions.csv.gz")[base_keys + ["p_factorized_traits"]]
    c = pd.read_csv(base / "rich_gnns" / "predictions.csv.gz")[base_keys + ["p_pu_graphsage", "p_pu_hgt"]]
    d = pd.read_csv(base / "compact_gnns" / "predictions.csv.gz")[base_keys + ["p_pu_graphsage", "p_pu_hgt"]].rename(columns={
        "p_pu_graphsage": "p_compact_graphsage",
        "p_pu_hgt": "p_compact_hgt",
    })
    # Rich GNN outputs intentionally omit taxon names.  Candidate rows are in
    # the shared deterministic event order, so rank repeated labels within an
    # event before joining rather than treating repeated positives as one row.
    for part in (a, b, c, d):
        part["candidate_rank"] = part.groupby(base_keys, sort=False).cumcount()
    frame = a.merge(b, on=keys, validate="one_to_one").merge(c, on=keys, validate="one_to_one").merge(d, on=keys, validate="one_to_one")
    frame = frame.loc[frame.split.str.startswith("unseen_study_fold_")].copy()
    models = {
        "strict_training_evidence":"p_strict_training_evidence",
        "degree_product_heuristic":"p_degree_product_heuristic",
        "full_feature_catboost":"p_full_feature_catboost",
        "two_tower_traits":"p_factorized_traits",
        "compact_graphsage":"p_compact_graphsage",
        "compact_hgt":"p_compact_hgt",
        "rich_graphsage":"p_pu_graphsage",
        "rich_hgt":"p_pu_hgt",
    }
    studies = frame.study.drop_duplicates().to_numpy()
    baseline = "strict_training_evidence"
    metric_names = ("average_precision", "roc_auc", "brier")
    def all_scores(x: pd.DataFrame) -> dict[tuple[str, str], float]:
        y = x.interaction_observed.to_numpy()
        values = {}
        for name, col in models.items():
            p = x[col].to_numpy()
            values[(name, "average_precision")] = float(average_precision_score(y, p))
            values[(name, "roc_auc")] = float(roc_auc_score(y, p))
            values[(name, "brier")] = float(brier_score_loss(y, p))
        return values
    point_scores = all_scores(frame)
    # Resample whole studies, not rows.  Each replicate evaluates every model
    # once, ensuring that all contrasts are paired on identical sampled data.
    by_study = {study: frame.loc[frame.study.eq(study)] for study in studies}
    draws = {key: [] for key in point_scores}
    for _ in range(N_BOOT):
        chosen = RNG.choice(studies, size=len(studies), replace=True)
        boot = pd.concat([by_study[study] for study in chosen], ignore_index=True)
        for key, value in all_scores(boot).items():
            draws[key].append(value)
    rows = []
    for metric in metric_names:
        for name in models:
            low, high = ci(np.asarray(draws[(name, metric)]))
            rows.append({"family":"phase1_links", "unit":"study", "model":name, "baseline":None, "metric":metric,
                         "point_estimate":point_scores[(name, metric)], "difference":None, "ci95_low":low, "ci95_high":high,
                         "n_units":len(studies), "bootstrap_replicates":N_BOOT})
            if name != baseline:
                difference_draws = np.asarray(draws[(name, metric)]) - np.asarray(draws[(baseline, metric)])
                low, high = ci(difference_draws)
                rows.append({"family":"phase1_links", "unit":"study", "model":name, "baseline":baseline,
                             "metric":f"{metric}_difference", "point_estimate":None,
                             "difference":point_scores[(name, metric)] - point_scores[(baseline, metric)], "ci95_low":low, "ci95_high":high,
                             "n_units":len(studies), "bootstrap_replicates":N_BOOT})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flower = flowering(); flower.to_csv(OUT / "flowering_bootstrap_ci.csv", index=False)
    bee = bee_activity(); bee.to_csv(OUT / "bee_activity_bootstrap_ci.csv", index=False)
    history = history_windows(); history.to_csv(OUT / "bee_history_bootstrap_ci.csv", index=False)
    links = phase1_links(); links.to_csv(OUT / "phase1_link_bootstrap_ci.csv", index=False)
    all_rows = pd.concat([flower, bee, history, links], ignore_index=True, sort=False)
    all_rows.to_csv(OUT / "all_paper_bootstrap_intervals.csv", index=False)
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()
