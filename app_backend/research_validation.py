"""Read-only summary of the latest offline research validation.

This module deliberately does not turn a historical backtest into a live farm
recommendation model.  It supplies the interface with the provenance and
results of the latest completed experiment so users can distinguish the
published operational release from the research evidence that informs the next
release.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase2_v16_appalachian_robust_validation"
SUMMARY = RESULTS / "summary.json"
METRICS = RESULTS / "robust_validation_summary.csv"
COMPARISONS = RESULTS / "paired_bootstrap_comparisons.csv"

METHOD_NAMES = {
    "strict_documented_training_links": "Documented links only",
    "catboost_supported_links": "CatBoost-supported links",
    "two_tower_supported_links": "Two-tower neural links",
    "random_mean": "Random reference",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def latest_validation() -> dict[str, Any]:
    """Return the completed V16 decision-level validation in UI-safe form."""
    required = (SUMMARY, METRICS, COMPARISONS)
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise LookupError(f"The latest research-validation files are unavailable: {', '.join(missing)}")

    manifest = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = _rows(METRICS)
    comparisons = _rows(COMPARISONS)
    views = []
    for design in ("leave_one_study_out", "rolling_origin"):
        metric_rows = [
            row for row in metrics
            if row["validation_design"] == design and int(row["mixture_size"]) == 5
        ]
        method_rows = []
        for row in metric_rows:
            method = row["method"]
            comparison = next((item for item in comparisons if (
                item["validation_design"] == design
                and int(item["mixture_size"]) == 5
                and item["method"] == method
                and item["baseline"] == "strict_documented_training_links"
                and item["metric"] == "bee_week_coverage"
            )), None)
            method_rows.append({
                "method": method,
                "label": METHOD_NAMES.get(method, method.replace("_", " ").title()),
                "bee_week_coverage": float(row["bee_week_coverage"]),
                "bee_coverage": float(row["bee_coverage"]),
                "week_coverage": float(row["week_coverage"]),
                "difference_vs_documented_links": (
                    float(comparison["mean_difference"]) if comparison else None
                ),
                "difference_ci95": (
                    [float(comparison["bootstrap_ci95_low"]), float(comparison["bootstrap_ci95_high"])]
                    if comparison else None
                ),
            })
        views.append({
            "design": design,
            "label": "Leave one study out" if design == "leave_one_study_out" else "Rolling future-year test",
            "methods": method_rows,
        })

    return {
        "research_release": "V16-2026-08-20",
        "status": "completed_offline_validation",
        "operational_status": "research_only_not_used_for_live_location_recommendations",
        "source_directory": str(RESULTS.relative_to(ROOT)).replace("\\", "/"),
        "validation_designs": manifest["validation_designs"],
        "link_layers": manifest["link_layers"],
        "mixture_size_displayed": 5,
        "views": views,
        "not_claimed": manifest["not_claimed"],
    }
