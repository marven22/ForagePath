"""Versioned policy boundary between completed research and live releases.

The module intentionally does *not* fabricate nationwide mixtures from model
benchmark files.  It exposes the selected component policy and gates live use
until a reproducible end-to-end recommendation release is built and replayed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ELIGIBILITY = ROOT / "results" / "us_state_eligibility_audit_v1" / "state_eligibility_summary.csv"
FULL_FLOWERS = ROOT / "results" / "us_eligible_state_benchmark_v1" / "flower_overall_model_summary.csv"
FULL_BEES = ROOT / "results" / "us_eligible_state_benchmark_v1" / "bee_overall_model_summary.csv"
FALLBACK_FLOWERS = ROOT / "results" / "us_fallback_state_benchmark_v1" / "flower_overall_model_summary.csv"
FALLBACK_BEES = ROOT / "results" / "us_fallback_state_benchmark_v1" / "bee_overall_model_summary.csv"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise LookupError(f"Required validated-pipeline artifact is missing: {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _best(rows: list[dict[str, str]], metric: str, *, lower: bool) -> dict[str, Any]:
    winner = sorted(rows, key=lambda item: float(item[metric]), reverse=not lower)[0]
    return {"method": winner["method"], "metric": metric, "value": float(winner[metric])}


def state_policy(state_code: str) -> dict[str, Any]:
    state_code = state_code.upper().strip()
    rows = _read(ELIGIBILITY)
    state = next((row for row in rows if row["state_code"] == state_code), None)
    if not state:
        raise LookupError(f"No U.S. state evidence audit exists for {state_code}.")
    tier = state["eligibility"]
    if tier == "full_state_model_eligible":
        flower = _best(_read(FULL_FLOWERS), "weighted_brier", lower=True)
        bee = _best(_read(FULL_BEES), "future_mean_log_probability", lower=False)
        mode = "full_state_pipeline_pending_release"
        local = "Statewide bee timing is the validated default; local blending requires an additional local-evidence gate."
    elif tier == "statewide_fallback_only":
        flower = _best(_read(FALLBACK_FLOWERS), "weighted_brier", lower=True)
        bee = _best(_read(FALLBACK_BEES), "future_mean_log_probability", lower=False)
        mode = "statewide_fallback_pipeline_pending_release"
        local = "Use broader evidence for bee timing; do not present grid-specific activity as validated."
    else:
        flower = bee = None
        mode = "abstain_pending_data_acquisition"
        local = "Evidence is insufficient for a validated model-driven recommendation."
    return {
        "pipeline_version": "national-evidence-policy-v1",
        "state_code": state_code,
        "eligibility": tier,
        "operational_mode": mode,
        "selected_flower_model": flower,
        "selected_bee_model": bee,
        "network_policy": "strict traceable A/B documented bee–plant links only",
        "mixture_policy": "Phase 2 CatBoost-supported-link optimizer is research-validated and must be packaged with a frozen state release before live use.",
        "localization_policy": local,
        "release_gate": "A state can become live only after a versioned model artifact, optimized mixtures, provenance ledger, and end-to-end historical replay are recorded.",
        "audit_counts": {key: int(state[key]) for key in ("flower_model_plants", "activity_bee_species", "linked_plants", "linked_bees", "strict_edges")},
        "claim_boundary": "This policy supports potential forage continuity only; it does not prove bee health, crop yield, establishment, or causal field benefit.",
    }


def catalog() -> dict[str, Any]:
    rows = _read(ELIGIBILITY)
    return {
        "pipeline_version": "national-evidence-policy-v1",
        "states": [state_policy(row["state_code"]) for row in rows],
        "live_release_note": "V40 remains the only live PA/MD/MI recommendation release until the new pipeline passes end-to-end replay.",
    }
