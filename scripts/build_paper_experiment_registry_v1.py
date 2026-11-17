"""Build an inspectable, source-backed registry of experiments considered for the paper.

The registry distinguishes raw source scale from the derived evaluation cohort,
records the exact artifacts behind each claim, and flags whether an experiment
is ready for manuscript use or still needs a defined follow-up.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\vmargapu\OneDrive - Villanova University\Documents\Research\pollinator_forage_project")
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_experiment_registry_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.relative_to(ROOT))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    flower_dir = RESULTS / "flowering_history_window_sensitivity_v1"
    flower_manifest = load_json(flower_dir / "experiment_manifest.json")
    flower_2015 = next(item for item in flower_manifest["windows"] if item["train"] == "2015-2022")
    neon_dir = RESULTS / "neon_external_flowering_evaluation_v2"
    neon_summary = load_json(neon_dir / "summary.json")

    bee_dir = RESULTS / "national_bee_history_window_sensitivity_v1" / "train_2015_2022"
    bee_summary = load_json(bee_dir / "summary.json")
    bee_metrics = read_csv(bee_dir / "final_test_metrics_summary.csv")
    bee_per_species = read_csv(bee_dir / "final_test_metrics_by_bee.csv")
    scored_species = len({row["bee_key"] for row in bee_per_species})
    raw_histogram_records = sum(
        int(row["test_records"])
        for row in bee_per_species
        if row["model"] == "raw_weekly_histogram"
    )
    if scored_species != 281 or raw_histogram_records != 163_616:
        raise RuntimeError(
            "Unexpected bee evaluation cohort. "
            f"distinct species={scored_species}, raw-histogram records={raw_histogram_records}"
        )

    phase1 = RESULTS / "phase1_v16_appalachian_all"
    primary_splits = {"future_year", "held_event", "unseen_pair", "unseen_plant", "unseen_bee"}
    phase1_paths = {
        "baseline": phase1 / "baselines" / "metrics.csv",
        "two_tower": phase1 / "two_tower" / "factorized_ai_metrics.csv",
        "compact_gnn": phase1 / "compact_gnns" / "metrics.csv",
        "rich_gnn": phase1 / "rich_gnns" / "metrics.csv",
    }
    phase1_tables = {name: read_csv(path) for name, path in phase1_paths.items()}
    phase1_split_sets = {name: {row["split"] for row in table} for name, table in phase1_tables.items()}
    shared_primary_splits = sorted(set.intersection(*(splits for splits in phase1_split_sets.values())) & primary_splits)
    rich_progress = load_json(phase1 / "rich_gnns" / "progress.json")

    phase2 = RESULTS / "phase2_v16_appalachian_robust_validation"
    phase2_summary = load_json(phase2 / "summary.json")
    phase2_progress = load_json(phase2 / "progress.json")
    phase2_metrics = read_csv(phase2 / "robust_validation_metrics.csv")
    phase2_sites = {row["site_key"] for row in phase2_metrics}
    phase2_years = sorted({row["held_out_year"] for row in phase2_metrics if row["validation_design"] == "rolling_origin"})
    phase2_studies = sorted({row["study"] for row in phase2_metrics if row["validation_design"] == "leave_one_study_out"})

    uncertainty = RESULTS / "paper_uncertainty_intervals_v1"
    paths = {
        "flowering_manifest": flower_dir / "experiment_manifest.json",
        "flowering_metrics": flower_dir / "train_2015_2022" / "final_test_metrics.csv",
        "flowering_ci": uncertainty / "flowering_bootstrap_ci.csv",
        "neon_summary": neon_dir / "summary.json",
        "neon_metrics": neon_dir / "metrics.csv",
        "neon_ci": uncertainty / "neon_site_bootstrap_ci_v1.csv",
        "bee_summary": bee_dir / "summary.json",
        "bee_metrics": bee_dir / "final_test_metrics_summary.csv",
        "bee_per_species": bee_dir / "final_test_metrics_by_bee.csv",
        "bee_verified_ci": uncertainty / "bee_activity_verified_281_bootstrap_ci.csv",
        "link_baselines": phase1_paths["baseline"],
        "link_two_tower": phase1_paths["two_tower"],
        "link_compact_gnn": phase1_paths["compact_gnn"],
        "link_rich_gnn": phase1_paths["rich_gnn"],
        "link_ci": uncertainty / "phase1_link_bootstrap_ci.csv",
        "mixture_summary": phase2 / "summary.json",
        "mixture_metrics": phase2 / "robust_validation_metrics.csv",
        "mixture_pairs": phase2 / "paired_bootstrap_comparisons.csv",
    }
    artifact_paths = {name: require(path) for name, path in paths.items()}

    rows = [
        {
            "experiment_id": "F1",
            "paper_role": "Internal flowering development benchmark",
            "question": "Which flowering model best estimates later-year weekly flowering probability?",
            "train_validation_test": "Train 2015-2022 | validation 2023 | inspected development benchmark 2024-2025",
            "evaluation_cohort": f"{flower_2015['test_rows']:,} plant-week rows and 587,240 observation checks",
            "model_scope": "Seasonal, generalized linear, boosted, neural, transformer, and calibrated ensemble candidates",
            "statistical_evidence": "Paired site-year bootstrap, 2,895 site-years, 1,000 replicates",
            "status": "READY WITH QUALIFIER",
            "manuscript_use": "Internal model comparison only. Do not describe 2024-2025 as an untouched final test.",
            "primary_artifacts": f"{artifact_paths['flowering_manifest']} | {artifact_paths['flowering_metrics']} | {artifact_paths['flowering_ci']}",
        },
        {
            "experiment_id": "F2",
            "paper_role": "Cross-source external flowering evaluation",
            "question": "Does the flowering approach retain value on an independent source?",
            "train_validation_test": "USA-NPN-trained models | NEON records from 2024-2025",
            "evaluation_cohort": f"42,294 weekly rows from {neon_summary['sites']} sites and {neon_summary['plants']} plants",
            "model_scope": "Seasonal reference, CatBoost, FT-Transformer, global ensemble, hierarchical ensemble",
            "statistical_evidence": "Site-clustered bootstrap confidence intervals use all 42 NEON sites and 2,000 replicates. Model uses training-median soil imputation because NEON soil covariates are unavailable.",
            "status": "READY WITH QUALIFIER",
            "manuscript_use": "External cross-source evaluation. Describe it as partial-covariate, not a full deployment simulation.",
            "primary_artifacts": f"{artifact_paths['neon_summary']} | {artifact_paths['neon_metrics']} | {artifact_paths['neon_ci']}",
        },
        {
            "experiment_id": "B1",
            "paper_role": "Future-year native bee seasonal activity",
            "question": "Which seasonal activity model best allocates probability to later observed bee weeks?",
            "train_validation_test": "Training eligibility and model fitting 2015-2022 | validation 2023 | test 2024-2025",
            "evaluation_cohort": f"{scored_species} training-eligible species appearing later and 163,616 scored later occurrence records. Training eligible pool equals {bee_summary['model_eligible_bees_from_training_only']} species.",
            "model_scope": "Raw weekly histogram, circular kernel, cyclic spline, spatial partial pooling, trait-family partial pooling",
            "statistical_evidence": "Paired species bootstrap confidence intervals use the verified 281-species cohort, 10,000 replicates, and a fixed random seed.",
            "status": "READY",
            "manuscript_use": "Use the corrected point metrics and verified 281-species confidence intervals.",
            "primary_artifacts": f"{artifact_paths['bee_summary']} | {artifact_paths['bee_metrics']} | {artifact_paths['bee_per_species']} | {artifact_paths['bee_verified_ci']}",
        },
        {
            "experiment_id": "L1",
            "paper_role": "Plant-bee link ranking",
            "question": "Can methods rank documented links under held-event and novelty conditions?",
            "train_validation_test": "Shared V16 split protocol. Common core splits are held event, unseen pair, unseen plant, unseen bee, and future year.",
            "evaluation_cohort": f"All four output groups contain the same {', '.join(shared_primary_splits)} core splits. Example held-event split contains 8,669 candidate rows and 359 events.",
            "model_scope": "Strict evidence, degree heuristic, CatBoost, two-tower neural, compact and rich GraphSAGE, compact and rich HGT",
            "statistical_evidence": "Study-unit bootstrap intervals are available for strict evidence, heuristic, CatBoost, two-tower, rich GraphSAGE, and rich HGT.",
            "status": "READY WITH PROTOCOL NOTE",
            "manuscript_use": "Use the common core splits for direct family comparison. Do not use the stale parent progress file as evidence of incompleteness because rich-GNN progress confirms all 10 splits completed.",
            "primary_artifacts": f"{artifact_paths['link_baselines']} | {artifact_paths['link_two_tower']} | {artifact_paths['link_compact_gnn']} | {artifact_paths['link_rich_gnn']} | {artifact_paths['link_ci']}",
        },
        {
            "experiment_id": "M1",
            "paper_role": "Decision-level plant-mixture backtest",
            "question": "Do integrated seasonal-support mixtures improve future observed plant-bee overlap over baselines?",
            "train_validation_test": "Rolling-origin years and leave-one-study-out evaluations. Each held year or study is excluded from learned link information.",
            "evaluation_cohort": f"{len(phase2_sites)} site keys. Rolling-origin years {', '.join(phase2_years)}. Leave-one-study-out studies {len(phase2_studies)}. Mixture sizes 3, 5, and 7.",
            "model_scope": "Random mean, strict documented network, CatBoost-supported links, two-tower-supported links",
            "statistical_evidence": "Paired decision bootstrap comparisons available. The result summary records 23 paired decisions for the reported comparisons.",
            "status": "READY WITH SCOPE LIMIT",
            "manuscript_use": "Supports retrospective observed seasonal overlap, not field benefit, bee health, abundance, crop pollination, or yield.",
            "primary_artifacts": f"{artifact_paths['mixture_summary']} | {artifact_paths['mixture_metrics']} | {artifact_paths['mixture_pairs']}",
        },
    ]
    write_csv(OUT / "paper_experiment_registry.csv", rows)

    audit = {
        "registry_version": "v1",
        "status_definitions": {
            "READY WITH QUALIFIER": "The result can be reported when the stated interpretation boundary is retained.",
            "READY WITH PROTOCOL NOTE": "Direct comparisons must use the common splits described in the registry.",
            "READY WITH SCOPE LIMIT": "The experiment is valid for its measured retrospective outcome only.",
        },
        "automated_checks": {
            "flower_primary_window": flower_2015,
            "bee_training_eligible": bee_summary["model_eligible_bees_from_training_only"],
            "bee_scored_later_species": scored_species,
            "link_common_core_splits": shared_primary_splits,
            "rich_gnn_completed_splits": rich_progress["completed_splits"],
            "rich_gnn_total_splits": rich_progress["total_splits"],
            "mixture_completed_outer_folds": len(phase2_progress["completed_outer_folds"]),
            "mixture_validation_designs": phase2_summary["validation_designs"],
        },
        "rows": rows,
    }
    (OUT / "paper_experiment_registry_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = [
        "# Paper Experiment Registry v1",
        "",
        "This registry is the manuscript source of truth for experimental cohort definitions and claim boundaries. It was generated from the cited result artifacts on 2026-09-14.",
        "",
        "| ID | Paper role | Status | Cohort and boundary |",
        "|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| {row['experiment_id']} | {row['paper_role']} | {row['status']} | {row['evaluation_cohort']} |")
    md.extend(["", "## Required before final manuscript submission", "", "- Use only the five common V16 splits when directly comparing all plant-bee link model families.", "- Preserve the stated qualifiers for the internal flowering benchmark, partial-covariate NEON evaluation, and retrospective mixture outcome.", "- Do not use the older 348-species bee confidence-interval file. The verified 281-species file listed below is authoritative.", "", "## Artifact locations", ""])
    for row in rows:
        md.append(f"- **{row['experiment_id']}**  {row['primary_artifacts']}")
    (OUT / "paper_experiment_registry.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote registry to {OUT}")


if __name__ == "__main__":
    main()
