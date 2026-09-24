from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = ROOT / "data" / "reference" / "spatial_context.json"


def _point_in_ring(longitude: float, latitude: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous[:2]
        x2, y2 = current[:2]
        if ((y1 > latitude) != (y2 > latitude)) and (
            longitude < (x2 - x1) * (latitude - y1) / ((y2 - y1) or 1e-15) + x1
        ):
            inside = not inside
        previous = current
    return inside


def _point_in_polygon(longitude: float, latitude: float, polygon: list[list[list[float]]]) -> bool:
    return bool(polygon and _point_in_ring(longitude, latitude, polygon[0]) and not any(
        _point_in_ring(longitude, latitude, hole) for hole in polygon[1:]
    ))


def point_in_geometry(longitude: float, latitude: float, geometry: dict[str, Any]) -> bool:
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return _point_in_polygon(longitude, latitude, coordinates)
    if geometry.get("type") == "MultiPolygon":
        return any(_point_in_polygon(longitude, latitude, polygon) for polygon in coordinates)
    return False


class SpatialContext:
    def __init__(self, path: Path = DEFAULT_CONTEXT):
        self.path = path.resolve()
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.policy = self.payload["policy"]
        self.environment_features: list[str] = self.payload["environment_features"]
        self.boundaries: dict[str, dict[str, Any]] = {}
        for state, item in self.payload["states"].items():
            boundary_path = ROOT / item["boundary_path"]
            self.boundaries[state] = json.loads(boundary_path.read_text(encoding="utf-8"))

    def state(self, state_code: str) -> dict[str, Any]:
        try:
            return self.payload["states"][state_code]
        except KeyError as exc:
            raise LookupError(f"No spatial context is available for {state_code}.") from exc

    def boundary(self, state_code: str) -> dict[str, Any]:
        self.state(state_code)
        return self.boundaries[state_code]

    def location(self, state_code: str, latitude: float, longitude: float) -> dict[str, Any]:
        boundary = self.boundary(state_code)
        county = None
        for feature in boundary["features"]:
            if feature["properties"].get("kind") != "county":
                continue
            if point_in_geometry(longitude, latitude, feature["geometry"]):
                county = {
                    "geoid": str(feature["properties"]["GEOID"]),
                    "name": str(feature["properties"]["BASENAME"]),
                }
                break
        state_feature = next(
            feature for feature in boundary["features"] if feature["properties"].get("kind") == "state"
        )
        # County polygons are the primary containment source because independently simplified
        # state and county coastlines can differ slightly at their shared edges.
        if county is None and not point_in_geometry(longitude, latitude, state_feature["geometry"]):
            raise ValueError(f"The selected point is outside {state_feature['properties']['BASENAME']}.")
        step = float(self.policy["grid_resolution_degrees"])
        south = math.floor((latitude + 1e-10) / step) * step
        west = math.floor((longitude + 1e-10) / step) * step
        cell_id = f"{state_code}_0p5_{south:.1f}_{west:.1f}"
        return {"county": county, "cell_id": cell_id, "cell": self.state(state_code)["cells"].get(cell_id)}

    def cell(self, state_code: str, cell_id: str) -> dict[str, Any] | None:
        return self.state(state_code)["cells"].get(cell_id)

    def environmental_similarity(
        self, state_code: str, target_cell_id: str, candidate_cell_id: str
    ) -> tuple[float | None, int]:
        cells = self.state(state_code)["cells"]
        target = cells.get(target_cell_id)
        candidate = cells.get(candidate_cell_id)
        if not target or not candidate:
            return None, 0
        usable = []
        for feature in self.environment_features:
            target_value = target["environment"].get(feature)
            candidate_value = candidate["environment"].get(feature)
            values = [cell["environment"].get(feature) for cell in cells.values()]
            values = [float(value) for value in values if value is not None]
            if target_value is None or candidate_value is None or len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            standard_deviation = math.sqrt(variance)
            if standard_deviation <= 1e-12:
                continue
            usable.append((float(target_value) - float(candidate_value)) / standard_deviation)
        if not usable:
            return None, 0
        distance = math.sqrt(sum(value * value for value in usable) / len(usable))
        return 1.0 / (1.0 + distance), len(usable)
