from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .database import DEFAULT_DB, connect
from .national_preview import NationalStatewidePreview
from .plant_profiles import load_profiles
from .spatial_context import SpatialContext


class RecommendationService:
    """Read-only application boundary for published scientific recommendations."""

    def __init__(self, database_path: Path | str = DEFAULT_DB) -> None:
        self.database_path = Path(database_path)
        self.plant_profiles = load_profiles()
        self.spatial = SpatialContext()
        self.national_preview = NationalStatewidePreview()

    def health(self) -> dict[str, Any]:
        with connect(self.database_path, read_only=True) as db:
            release = db.execute(
                "SELECT release_id, methodology_version, published_at_utc "
                "FROM data_releases WHERE is_current = 1 AND status = 'published'"
            ).fetchone()
            spatial_release = self.spatial.payload.get("release", {}).get("release_id")
            release_matches = bool(release and release["release_id"] == spatial_release)
            return {
                "status": "ready" if release_matches else (
                    "spatial_release_mismatch" if release else "no_published_release"
                ),
                "database": str(self.database_path.resolve()),
                "current_release": dict(release) if release else None,
            }

    def list_states(self) -> list[dict[str, Any]]:
        with connect(self.database_path, read_only=True) as db:
            rows = db.execute(
                "SELECT s.state_code, s.state_name, COUNT(DISTINCT g.cell_id) AS modeled_cells, "
                "COUNT(DISTINCT r.cell_id) AS published_cells "
                "FROM states s JOIN grid_cells g USING(state_code) "
                "LEFT JOIN current_recommendations r ON r.state_code=s.state_code AND r.cell_id=g.cell_id "
                "GROUP BY s.state_code, s.state_name ORDER BY s.state_code"
            ).fetchall()
            local = {row["state_code"]: dict(row) for row in rows}
        # PA/MD/MI keep their detailed published local catalogs. The other
        # states are explicitly statewide guidance or abstentions.
        catalog = {row["state_code"]: row for row in self.national_preview.catalog_states()}
        catalog.update(local)
        return [catalog[state] for state in sorted(catalog)]

    def list_planting_classes(self, state_code: str) -> list[dict[str, Any]]:
        state_code = state_code.upper().strip()
        if state_code not in self.spatial.payload["states"]:
            return self.national_preview.planting_classes(state_code)
        with connect(self.database_path, read_only=True) as db:
            rows = db.execute(
                "SELECT planting_class, COUNT(*) AS modeled_cells, "
                "MIN((SELECT COUNT(*) FROM recommendation_plants rp "
                "WHERE rp.recommendation_id=r.recommendation_id)) AS mixture_size "
                "FROM current_recommendations r WHERE state_code=? "
                "GROUP BY planting_class ORDER BY planting_class", (state_code,)
            ).fetchall()
            if not rows:
                raise LookupError(f"No published recommendations are available for {state_code}.")
            return [dict(row) for row in rows]

    def list_cells(self, state_code: str) -> list[dict[str, Any]]:
        state_code = state_code.upper().strip()
        if state_code not in self.spatial.payload["states"]:
            self.national_preview.state(state_code)
            return self.national_preview.local_cells(state_code)
        cells = self.spatial.state(state_code)["cells"]
        result = []
        for cell_id, cell in sorted(cells.items()):
            validation = cell["evidence"]["cell_validation_pass"]
            published = bool(cell["has_published_recommendation"])
            grade = "strong_local" if published and validation is True else (
                "modeled_limited" if published else "modeled_no_recommendation"
            )
            south, west, north, east = cell["bounds"]
            result.append({
                "cell_id": cell_id,
                "center_latitude": (south + north) / 2,
                "center_longitude": (west + east) / 2,
                "resolution_degrees": north - south,
                "has_published_recommendation": published,
                "cell_validation_pass": validation,
                "evidence_grade": grade,
                "county_name": cell["county"]["name"] if cell["county"] else None,
                "state_area_fraction": cell["state_area_fraction"],
            })
        return result

    def boundary(self, state_code: str) -> dict[str, Any]:
        if state_code not in self.spatial.payload["states"]:
            return self.national_preview.boundary(state_code)
        return self.spatial.boundary(state_code.upper().strip())

    def recommend(
        self,
        *,
        state_code: str,
        planting_class: str,
        cell_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        mixture_size: int | None = None,
    ) -> dict[str, Any]:
        state_code = state_code.upper().strip()
        if state_code not in self.spatial.payload["states"]:
            allowed = "statewide native forage mix"
            if planting_class.strip().casefold() != allowed:
                raise ValueError(
                    "Nationwide guidance uses the 'statewide native forage mix' planting type."
                )
            if cell_id is not None:
                raise ValueError("Statewide guidance does not use a grid cell. Select the state and request its statewide mix.")
            return self.national_preview.recommend(state_code, latitude, longitude)
        with connect(self.database_path, read_only=True) as db:
            release = db.execute(
                "SELECT * FROM data_releases WHERE is_current = 1 AND status = 'published'"
            ).fetchone()
            if not release:
                raise LookupError("No validated recommendation release is published.")
            selection = self._resolve_recommendation(
                db, release["release_id"], state_code, planting_class, cell_id, latitude, longitude
            )
            recommendation = selection["recommendation"]
            source_cell = selection["source_cell"]
            plants = self._load_plants(db, recommendation["recommendation_id"], source_cell["cell_id"], state_code)
            for plant in plants:
                profile = self.plant_profiles.get(plant["scientific_name"], {})
                plant["common_name"] = profile.get("common_name")
                plant["growth_form"] = profile.get("growth_form")
                plant["season_group"] = profile.get("season_group")
                plant["moisture_guidance"] = profile.get("moisture_guidance")
                plant["light_guidance"] = profile.get("light_guidance")
                plant["acidic_preference"] = profile.get("acidic_preference")
                plant["guidance_status"] = profile.get("guidance_status")
                plant["guidance_source"] = profile.get("guidance_source")
                plant["practical_note"] = profile.get("practical_note")
                plant["image"] = profile.get("image")
            if mixture_size is not None and mixture_size != len(plants):
                raise ValueError(
                    f"This planting type has a validated mixture size of {len(plants)}; "
                    f"a size of {mixture_size} has not been optimized and validated yet."
                )
            bee_groups: dict[str, set[str]] = {}
            for plant in plants:
                for group in plant["documented_native_bee_groups"]:
                    bee_groups.setdefault(group["bee_group"], set()).update(group["bees"])
            return {
                "release": {
                    "release_id": release["release_id"],
                    "methodology_version": release["methodology_version"],
                    "published_at_utc": release["published_at_utc"],
                },
                "request": {
                    "state_code": state_code,
                    "planting_class": recommendation["planting_class"],
                    "requested_cell_id": cell_id,
                    "requested_latitude": latitude,
                    "requested_longitude": longitude,
                    "resolved_cell_id": source_cell["cell_id"],
                    "modeled_cell_center": [
                        source_cell["representative_latitude"], source_cell["representative_longitude"]
                    ],
                    "distance_to_modeled_cell_km": selection["distance_km"],
                },
                "spatial_decision": selection["spatial_decision"],
                "local_evidence": selection["local_evidence"],
                "source_evidence": source_cell["evidence"],
                "recommendation": {
                    "plants": plants,
                    "mixture_size": len(plants),
                    "decision_status": recommendation["decision_status"],
                    "robust_change_accepted": bool(recommendation["change_gate_passed"]),
                    "improvement_probability": recommendation["improvement_probability"],
                    "lower_tail_availability_score": recommendation["availability_p10"],
                    "minimum_candidate_plant_stability": recommendation["minimum_plant_stability"],
                    "policy": recommendation["policy"],
                    "ai_used_operationally": bool(recommendation["ai_gate_open"]),
                    "ai_gate_reason": recommendation["ai_gate_reason"],
                },
                "documented_native_bee_support": [
                    {"bee_group": group, "species_count": len(bees), "species": sorted(bees)}
                    for group, bees in sorted(bee_groups.items())
                ],
                "interpretation": {
                    "confidence_signal": (
                        "passed_robust_change_gate" if recommendation["change_gate_passed"]
                        else "baseline_retained_because_change_gate_did_not_pass"
                    ),
                    "claim_boundary": recommendation["claim_boundary"],
                    "important_note": "Missing plant-bee links are unknown, not confirmed absences.",
                },
            }

    def _resolve_recommendation(
        self, db, release_id, state_code, planting_class, cell_id, latitude, longitude
    ) -> dict[str, Any]:
        available = [dict(row) for row in db.execute(
            "SELECT r.*, (SELECT GROUP_CONCAT(scientific_name, ';') FROM ("
            "SELECT p.scientific_name FROM recommendation_plants rp JOIN plants p USING(plant_id) "
            "WHERE rp.recommendation_id=r.recommendation_id ORDER BY rp.ordinal)) AS mixture_key "
            "FROM current_recommendations r WHERE r.state_code=? AND lower(r.planting_class)=lower(?)",
            (state_code, planting_class),
        )]
        if not available:
            classes = [row[0] for row in db.execute(
                "SELECT DISTINCT planting_class FROM current_recommendations WHERE state_code=? ORDER BY 1",
                (state_code,),
            )]
            raise LookupError(f"No '{planting_class}' recommendation exists for {state_code}. Available: {classes}")

        if cell_id:
            target_cell = self.spatial.cell(state_code, cell_id)
            if not target_cell:
                raise LookupError(f"Unknown modeled cell '{cell_id}' for {state_code}.")
            location = {"county": target_cell["county"], "cell_id": cell_id, "cell": target_cell}
            latitude = target_cell["representative_latitude"]
            longitude = target_cell["representative_longitude"]
        else:
            if latitude is None or longitude is None:
                raise ValueError("Provide cell_id or both latitude and longitude.")
            location = self.spatial.location(state_code, latitude, longitude)
            target_cell = location["cell"]

        context_cells = self.spatial.state(state_code)["cells"]
        candidates = [row for row in available if row["cell_id"] in context_cells]
        if not candidates:
            raise LookupError(f"No spatially supported recommendations are available for {state_code}.")

        def distance(row: dict[str, Any]) -> float:
            source = context_cells[row["cell_id"]]
            return self._haversine(
                latitude, longitude, source["representative_latitude"], source["representative_longitude"]
            )

        direct = next((row for row in candidates if row["cell_id"] == location["cell_id"]), None)
        similarity = None
        similarity_features = 0
        if direct:
            selected = direct
            level = "local_grid"
            validation = target_cell["evidence"]["cell_validation_pass"] if target_cell else None
            if validation is True:
                grade, label = "strong_local", "Strong local evidence"
                reason = "The selected point falls inside a published grid that passed its cell-level validation audit."
            else:
                grade, label = "supported_fallback", "Local model, limited validation"
                reason = "The selected point falls inside a published grid, but complete cell-level bee validation is unavailable or did not pass."
        else:
            selected = None
            county = location["county"]
            county_limit = float(self.spatial.policy["county_fallback_max_km"])
            if county:
                county_candidates = [
                    row for row in candidates
                    if context_cells[row["cell_id"]].get("county")
                    and context_cells[row["cell_id"]]["county"]["geoid"] == county["geoid"]
                    and distance(row) <= county_limit
                ]
                if county_candidates:
                    selected = min(county_candidates, key=distance)
                    level, grade, label = "county", "supported_fallback", "County-level fallback"
                    reason = "No published recommendation exists for the exact grid; the nearest published grid in the same county was used."

            if selected is None and target_cell:
                ecological_limit = float(self.spatial.policy["ecological_fallback_max_km"])
                minimum_features = int(self.spatial.policy["minimum_environment_features"])
                ecological_candidates = []
                for row in candidates:
                    source = context_cells[row["cell_id"]]
                    if source.get("ecological_cluster") != target_cell.get("ecological_cluster"):
                        continue
                    score, features = self.spatial.environmental_similarity(
                        state_code, location["cell_id"], row["cell_id"]
                    )
                    if score is not None and features >= minimum_features and distance(row) <= ecological_limit:
                        ecological_candidates.append((score, features, -distance(row), row))
                if ecological_candidates:
                    similarity, similarity_features, _, selected = max(ecological_candidates, key=lambda item: item[:3])
                    level, grade, label = "ecological_region", "supported_fallback", "Environmental-region fallback"
                    reason = "No defensible grid or county recommendation was available; a validated grid in the same modeled ecological cluster was selected by measured environmental similarity."

            if selected is None:
                counts: dict[str, int] = {}
                for row in candidates:
                    counts[row["mixture_key"]] = counts.get(row["mixture_key"], 0) + 1
                modal = sorted(counts, key=lambda key: (-counts[key], key))[0]
                modal_candidates = [row for row in candidates if row["mixture_key"] == modal]
                selected = min(modal_candidates, key=distance)
                level, grade, label = "statewide", "statewide_guidance", "Statewide guidance"
                reason = "Local, county, and environmentally matched evidence were insufficient; the state's most frequently published mixture was used."

        source_cell = {"cell_id": selected["cell_id"], **context_cells[selected["cell_id"]]}
        distance_km = distance(selected)
        local_evidence = target_cell["evidence"] if target_cell else {
            "plant_occurrence_records": None, "flowering_observations": None,
            "flowering_observation_years": None, "model_years": None,
            "documented_state_network_links": None, "bee_occurrence_records": None,
            "bee_sampling_events": None, "bee_observation_years": None,
            "native_bee_species": None, "cell_validation_pass": None,
            "bee_coverage_source": None,
        }
        return {
            "recommendation": selected,
            "source_cell": source_cell,
            "distance_km": distance_km,
            "local_evidence": local_evidence,
            "spatial_decision": {
                "policy_version": self.spatial.policy["policy_version"],
                "level": level,
                "evidence_grade": grade,
                "label": label,
                "county_name": location["county"]["name"] if location["county"] else None,
                "target_cell_id": location["cell_id"],
                "source_cell_id": selected["cell_id"],
                "distance_km": distance_km,
                "county_limit_km": float(self.spatial.policy["county_fallback_max_km"]),
                "ecological_limit_km": float(self.spatial.policy["ecological_fallback_max_km"]),
                "environmental_similarity": similarity,
                "environment_features_compared": similarity_features,
                "reason": reason,
            },
        }

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        radius_km = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
        return 2 * radius_km * math.asin(math.sqrt(a))

    @staticmethod
    def _load_plants(db, recommendation_id, cell_id, state_code):
        rows = db.execute(
            "SELECT rp.ordinal, p.plant_id, p.scientific_name, p.display_name, sp.planting_class, "
            "sp.coarse_habitat, sp.phenology_confidence, ps.suitability_score, ps.interpretation "
            "FROM recommendation_plants rp JOIN plants p USING(plant_id) "
            "JOIN recommendations r ON r.recommendation_id = rp.recommendation_id "
            "JOIN state_plants sp ON sp.release_id = r.release_id AND sp.state_code = r.state_code "
            "AND sp.plant_id = p.plant_id "
            "LEFT JOIN plant_suitability ps ON ps.release_id = r.release_id AND ps.cell_id = r.cell_id "
            "AND ps.plant_id = p.plant_id "
            "WHERE rp.recommendation_id = ? ORDER BY rp.ordinal", (recommendation_id,)
        ).fetchall()
        result = []
        for row in rows:
            phenology = db.execute(
                "SELECT iso_week, AVG(flowering_probability) AS probability "
                "FROM phenology_predictions pp JOIN recommendations r ON r.release_id = pp.release_id "
                "WHERE r.recommendation_id = ? AND pp.cell_id = ? AND pp.plant_id = ? "
                "GROUP BY iso_week ORDER BY iso_week",
                (recommendation_id, cell_id, row["plant_id"]),
            ).fetchall()
            active = [x["iso_week"] for x in phenology if x["probability"] >= 0.20]
            peak = max(phenology, key=lambda x: x["probability"])["iso_week"] if phenology else None
            groups = db.execute(
                "SELECT COALESCE(b.bee_group, 'unclassified') AS bee_group, b.scientific_name "
                "FROM plant_bee_links l JOIN bees b USING(bee_id) "
                "JOIN state_bees sb ON sb.release_id = l.release_id AND sb.state_code = l.state_code "
                "AND sb.bee_id = l.bee_id "
                "WHERE l.release_id = (SELECT release_id FROM recommendations WHERE recommendation_id = ?) "
                "AND l.state_code = ? AND l.plant_id = ? AND l.evidence_type = 'documented' "
                "AND sb.is_native = 1 AND sb.eligible = 1 ORDER BY bee_group, b.scientific_name",
                (recommendation_id, state_code, row["plant_id"]),
            ).fetchall()
            grouped: dict[str, list[str]] = {}
            for group in groups:
                grouped.setdefault(group["bee_group"], []).append(group["scientific_name"])
            result.append({
                "scientific_name": row["scientific_name"],
                "display_name": row["display_name"],
                "planting_class": row["planting_class"],
                "coarse_habitat": row["coarse_habitat"],
                "local_suitability_score": row["suitability_score"],
                "suitability_interpretation": row["interpretation"],
                "phenology_confidence": row["phenology_confidence"],
                "modeled_peak_flowering_week": peak,
                "modeled_weeks_at_or_above_20_percent": active,
                "documented_eligible_native_bee_species_count": len(groups),
                "documented_native_bee_groups": [
                    {"bee_group": group, "bees": bees} for group, bees in sorted(grouped.items())
                ],
            })
        return result
