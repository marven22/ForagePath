"""Read-only nationwide statewide guidance, kept separate from local releases."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .spatial_context import point_in_geometry


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/reference/national_statewide_preview.json"
LOCAL_CATALOG_PATH = ROOT / "data/reference/national_localized_guidance_v2.json"
MAP_CONTEXT_PATH = ROOT / "data/reference/national_map_context_v1.json"


class NationalStatewidePreview:
    """Serve frozen nationwide candidates without representing them as local validation."""

    def __init__(self) -> None:
        self.payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.local_payload = json.loads(LOCAL_CATALOG_PATH.read_text(encoding="utf-8"))
        self.map_payload = json.loads(MAP_CONTEXT_PATH.read_text(encoding="utf-8"))

    def has_state(self, state_code: str) -> bool:
        return state_code.upper() in self.payload["states"]

    def state(self, state_code: str) -> dict[str, Any]:
        state = self.payload["states"].get(state_code.upper())
        if not state:
            raise LookupError(f"Unknown U.S. state '{state_code}'.")
        return state

    def catalog_states(self) -> list[dict[str, Any]]:
        result = []
        for code, item in sorted(self.payload["states"].items()):
            available = item["availability"] == "statewide_guidance"
            local_count = len(self.local_payload["states"].get(code, {}).get("cells", {}))
            map_count = len(self.map_payload["states"].get(code, {}).get("cells", []))
            result.append({
                "state_code": code,
                "state_name": item["state_name"],
                "modeled_cells": map_count if available else 0,
                "published_cells": local_count if local_count else (1 if available else 0),
                "guidance_level": (
                    "localized_with_statewide_fallback" if local_count else
                    "statewide_only" if available else "unavailable"
                ),
            })
        return result

    def planting_classes(self, state_code: str) -> list[dict[str, Any]]:
        state = self.state(state_code)
        if state["availability"] == "abstain":
            return []
        return [{
            "planting_class": "statewide native forage mix",
            "modeled_cells": 1,
            "mixture_size": len(state["plants"]),
        }]

    def local_cells(self, state_code: str) -> list[dict[str, Any]]:
        code = state_code.upper()
        local = self.local_payload["states"].get(code, {}).get("cells", {})
        cells = self.map_payload["states"].get(code, {}).get("cells", [])
        result = []
        for item in cells:
            cell_id = item["cell_id"]
            supported = cell_id in local
            result.append({
                "cell_id": cell_id,
                "center_latitude": item["center_latitude"],
                "center_longitude": item["center_longitude"],
                "resolution_degrees": item["resolution_degrees"],
                "has_published_recommendation": supported,
                "cell_validation_pass": True if supported else None,
                "evidence_grade": "modeled_limited" if supported else "modeled_no_recommendation",
                "county_name": None,
                "state_area_fraction": item["state_area_fraction"],
            })
        return result

    def boundary(self, state_code: str) -> dict[str, Any]:
        code = state_code.upper()
        item = self.map_payload["states"].get(code)
        if not item:
            self.state(code)
            return {"type": "FeatureCollection", "features": []}
        return json.loads((ROOT / item["boundary_path"]).read_text(encoding="utf-8"))

    @staticmethod
    def _cell_id(state_code: str, latitude: float, longitude: float) -> str:
        south = math.floor((latitude + 1e-10) / 0.5) * 0.5
        west = math.floor((longitude + 1e-10) / 0.5) * 0.5
        return f"{state_code}_0p5_{south:.1f}_{west:.1f}"

    def recommend(self, state_code: str, latitude: float | None, longitude: float | None) -> dict[str, Any]:
        code = state_code.upper()
        state = self.state(code)
        if state["availability"] == "abstain":
            raise LookupError(state["reason"])
        if latitude is not None and longitude is not None:
            boundary = self.boundary(code)
            state_feature = boundary["features"][0]
            if not point_in_geometry(longitude, latitude, state_feature["geometry"]):
                raise ValueError(f"The selected point is outside {state['state_name']}.")
            cell_id = self._cell_id(code, latitude, longitude)
            local = self.local_payload["states"].get(code, {}).get("cells", {}).get(cell_id)
            if local:
                return self._local_recommendation(code, state, cell_id, latitude, longitude, local)
        plants = []
        groups: dict[str, set[str]] = {}
        for item in state["plants"]:
            item_groups = item["documented_native_bee_groups"]
            for group in item_groups:
                groups.setdefault(group["bee_group"], set()).update(group["bees"])
            plants.append({
                "scientific_name": item["scientific_name"],
                "display_name": item["common_name"] or item["scientific_name"],
                "planting_class": "statewide native forage mix",
                "coarse_habitat": None,
                "local_suitability_score": None,
                "suitability_interpretation": "Statewide evidence guidance; local establishment screening is still needed.",
                "phenology_confidence": None,
                "modeled_peak_flowering_week": item["modeled_peak_flowering_week"],
                "modeled_weeks_at_or_above_20_percent": item["modeled_weeks_at_or_above_20_percent"],
                "documented_eligible_native_bee_species_count": len({bee for group in item_groups for bee in group["bees"]}),
                "documented_native_bee_groups": item_groups,
                "common_name": item["common_name"],
                "growth_form": None,
                "season_group": None,
                "moisture_guidance": None,
                "light_guidance": None,
                "acidic_preference": None,
                "guidance_status": "not_location_screened",
                "guidance_source": "Frozen national statewide guidance catalog",
                "practical_note": "Confirm local site, seed source, and establishment requirements before planting.",
                "image": None,
            })
        evidence = {
            "plant_occurrence_records": state["flower_records"],
            "flowering_observations": state["flower_records"],
            "flowering_observation_years": state["flower_years_observed"],
            "model_years": state["flower_years_observed"],
            "documented_state_network_links": state["strict_edges"],
            "bee_occurrence_records": state["bee_records"],
            "bee_sampling_events": None,
            "bee_observation_years": None,
            "native_bee_species": state["bee_species"],
            "cell_validation_pass": None,
            "bee_coverage_source": f"{state['bee_profile_scope']} bee activity profile",
        }
        return {
            "release": {
                "release_id": "national-statewide-guidance",
                "methodology_version": "statewide evidence guidance",
                "published_at_utc": self.payload["generated_at_utc"],
            },
            "request": {
                "state_code": code,
                "planting_class": "statewide native forage mix",
                "requested_cell_id": None,
                "requested_latitude": latitude,
                "requested_longitude": longitude,
                "resolved_cell_id": f"STATEWIDE_{code}",
                "modeled_cell_center": (latitude or 0.0, longitude or 0.0),
                "distance_to_modeled_cell_km": 0.0,
            },
            "spatial_decision": {
                "policy_version": "national-statewide-guidance-v1",
                "level": "statewide",
                "evidence_grade": "statewide_guidance",
                "label": "Statewide evidence guidance",
                "county_name": None,
                "target_cell_id": f"STATEWIDE_{code}",
                "source_cell_id": f"STATEWIDE_{code}",
                "distance_km": 0.0,
                "county_limit_km": 0.1,
                "ecological_limit_km": 0.1,
                "environmental_similarity": None,
                "environment_features_compared": 0,
                "reason": state["reason"],
            },
            "local_evidence": evidence,
            "source_evidence": evidence,
            "recommendation": {
                "plants": plants,
                "mixture_size": len(plants),
                "decision_status": "statewide_evidence_guidance",
                "robust_change_accepted": False,
                "improvement_probability": 0.0,
                "lower_tail_availability_score": 0.0,
                "minimum_candidate_plant_stability": 0.0,
                "policy": "strict_documented_links_statewide_candidate",
                "ai_used_operationally": False,
                "ai_gate_reason": "This is statewide evidence guidance, not a grid-level recommendation.",
            },
            "documented_native_bee_support": [
                {"bee_group": group, "species_count": len(bees), "species": sorted(bees)}
                for group, bees in sorted(groups.items())
            ],
            "interpretation": {
                "confidence_signal": "statewide_guidance_only",
                "claim_boundary": self.payload["claim_boundary"],
                "important_note": "Only documented plant-bee links are used; missing links are unknown rather than confirmed absences.",
            },
        }

    def _local_recommendation(
        self, code: str, state: dict[str, Any], cell_id: str, latitude: float, longitude: float, local: dict[str, Any]
    ) -> dict[str, Any]:
        groups: dict[str, set[str]] = {}
        plants = []
        for item in local["plants"]:
            item_groups = item["documented_native_bee_groups"]
            for group in item_groups:
                groups.setdefault(group["bee_group"], set()).update(group["bees"])
            plants.append({
                "scientific_name": item["scientific_name"],
                "display_name": item["common_name"] or item["scientific_name"],
                "planting_class": "statewide native forage mix",
                "coarse_habitat": None,
                "local_suitability_score": None,
                "suitability_interpretation": "Exact-grid candidate passed a 2022-to-2023 held-out potential-overlap replay.",
                "phenology_confidence": None,
                "modeled_peak_flowering_week": item["modeled_peak_flowering_week"],
                "modeled_weeks_at_or_above_20_percent": item["modeled_weeks_at_or_above_20_percent"],
                "documented_eligible_native_bee_species_count": len({bee for group in item_groups for bee in group["bees"]}),
                "documented_native_bee_groups": item_groups,
                "common_name": item["common_name"], "growth_form": None, "season_group": None,
                "moisture_guidance": None, "light_guidance": None, "acidic_preference": None,
                "guidance_status": "local_replay_supported", "guidance_source": "Held-out local potential-overlap replay",
                "practical_note": "This local result remains a research recommendation; confirm site conditions and establishment requirements before planting.",
                "image": None,
            })
        evidence = {
            "plant_occurrence_records": local["local_evidence"]["flowering_checks"],
            "flowering_observations": local["local_evidence"]["flowering_checks"],
            "flowering_observation_years": local["local_evidence"]["flowering_years"],
            "model_years": local["local_evidence"]["flowering_years"],
            "documented_state_network_links": local["local_evidence"]["local_strict_edges"],
            "bee_occurrence_records": local["local_evidence"]["bee_records"],
            "bee_sampling_events": None, "bee_observation_years": None,
            "native_bee_species": local["local_evidence"]["supported_bee_species"],
            "cell_validation_pass": True,
            "bee_coverage_source": "Exact-grid native-bee activity evidence and held-out potential-overlap replay",
        }
        return {
            "release": {"release_id": "national-localized-guidance", "methodology_version": "held-out local potential-overlap replay", "published_at_utc": self.local_payload["generated_at_utc"]},
            "request": {"state_code": code, "planting_class": "statewide native forage mix", "requested_cell_id": None,
                        "requested_latitude": latitude, "requested_longitude": longitude, "resolved_cell_id": cell_id,
                        "modeled_cell_center": [(local["bounds"][0] + local["bounds"][2]) / 2, (local["bounds"][1] + local["bounds"][3]) / 2],
                        "distance_to_modeled_cell_km": 0.0},
            "spatial_decision": {"policy_version": "national-localized-guidance-v2", "level": "local_grid", "evidence_grade": "local_replay_supported",
                                 "label": "Exact-grid evidence-supported guidance", "county_name": None, "target_cell_id": cell_id,
                                 "source_cell_id": cell_id, "distance_km": 0.0, "county_limit_km": 0.1, "ecological_limit_km": 0.1,
                                 "environmental_similarity": None, "environment_features_compared": local["local_evidence"]["environment_features_available"],
                                 "reason": self.local_payload["selection"]},
            "local_evidence": evidence, "source_evidence": evidence,
            "recommendation": {"plants": plants, "mixture_size": len(plants), "decision_status": "localized_replay_supported",
                               "robust_change_accepted": True, "improvement_probability": 0.0,
                               "lower_tail_availability_score": local["local_evidence"]["local_documented_overlap"],
                               "minimum_candidate_plant_stability": 0.0, "policy": "strict_documented_links_held_out_local_replay",
                               "ai_used_operationally": False, "ai_gate_reason": "The recommendation is deterministic and evidence-gated."},
            "documented_native_bee_support": [{"bee_group": group, "species_count": len(bees), "species": sorted(bees)} for group, bees in sorted(groups.items())],
            "interpretation": {"confidence_signal": "held_out_local_potential_overlap_improved", "claim_boundary": self.local_payload["claim_boundary"],
                               "important_note": "The replay measures documented-link seasonal overlap, not observed feeding or field outcomes."},
        }
