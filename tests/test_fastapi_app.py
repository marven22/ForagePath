from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_FEEDBACK = Path("data/app/test_feedback.sqlite3").resolve()
TEST_FEEDBACK.unlink(missing_ok=True)
os.environ["FEEDBACK_DATABASE_URL"] = f"sqlite:///{TEST_FEEDBACK.as_posix()}"

from fastapi.testclient import TestClient

from app_backend.api import app


class FastAPIApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.context.__exit__(None, None, None)
        TEST_FEEDBACK.unlink(missing_ok=True)

    def test_health_reports_published_release(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ready", response.json()["status"])
        self.assertEqual("v40-2026-08-17", response.json()["release"]["release_id"])
        self.assertNotIn("database", response.json())

    def test_state_inventory(self) -> None:
        response = self.client.get("/v1/states")
        self.assertEqual(200, response.status_code)
        observed = {
            row["state_code"]: (row["modeled_cells"], row["published_cells"])
            for row in response.json()["states"]
        }
        self.assertEqual(50, len(observed))
        self.assertEqual({"PA": (59, 59), "MD": (25, 5), "MI": (26, 26)}, {
            state: observed[state] for state in ("PA", "MD", "MI")
        })
        self.assertEqual((186, 1), observed["CA"])
        self.assertEqual((0, 0), observed["HI"])

    def test_exact_cell_recommendation(self) -> None:
        response = self.client.get("/v1/recommendations", params={
            "state": "PA",
            "planting_class": "herbaceous strip",
            "cell_id": "PA_0p5_40.0_-75.5",
        })
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual(5, payload["recommendation"]["mixture_size"])
        self.assertEqual("documented_network", payload["recommendation"]["policy"])
        self.assertFalse(payload["recommendation"]["ai_used_operationally"])
        self.assertGreater(len(payload["documented_native_bee_support"]), 0)
        self.assertIn("claim_boundary", payload["interpretation"])
        self.assertTrue(all(plant["common_name"] for plant in payload["recommendation"]["plants"]))
        self.assertTrue(all(plant["image"]["local_path"] for plant in payload["recommendation"]["plants"]))
        self.assertEqual("strong_local", payload["spatial_decision"]["evidence_grade"])
        self.assertTrue(payload["local_evidence"]["cell_validation_pass"])

    def test_coordinate_resolution(self) -> None:
        response = self.client.get("/v1/recommendations", params={
            "state": "MD",
            "planting_class": "woody hedgerow",
            "latitude": 38.75,
            "longitude": -75.75,
        })
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("MD_0p5_38.5_-76.0", response.json()["request"]["resolved_cell_id"])
        self.assertEqual(3, response.json()["recommendation"]["mixture_size"])

    def test_nationwide_statewide_guidance_is_explicit(self) -> None:
        response = self.client.get("/v1/recommendations", params={
            "state": "CA",
            "planting_class": "statewide native forage mix",
        })
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("statewide", payload["spatial_decision"]["level"])
        self.assertEqual("statewide_evidence_guidance", payload["recommendation"]["decision_status"])
        self.assertGreater(payload["recommendation"]["mixture_size"], 0)
        self.assertNotIn("Taraxacum officinale", {
            plant["scientific_name"] for plant in payload["recommendation"]["plants"]
        })

    def test_replay_supported_local_grid_overrides_statewide_fallback(self) -> None:
        # MA_0p5_42.0_-72.5 is one of the exact-grid products that survived
        # the corrected, leakage-controlled 2022-to-2023 replay release.
        cells = self.client.get("/v1/states/MA/cells")
        self.assertEqual(200, cells.status_code)
        self.assertEqual(17, len(cells.json()["cells"]))
        self.assertEqual(1, sum(cell["has_published_recommendation"] for cell in cells.json()["cells"]))
        response = self.client.get("/v1/recommendations", params={
            "state": "MA", "planting_class": "statewide native forage mix",
            "latitude": 42.25, "longitude": -72.25,
        })
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("local_grid", payload["spatial_decision"]["level"])
        self.assertEqual("local_replay_supported", payload["spatial_decision"]["evidence_grade"])
        self.assertEqual("localized_replay_supported", payload["recommendation"]["decision_status"])

    def test_national_map_layer_is_available_for_statewide_guidance(self) -> None:
        cells = self.client.get("/v1/states/CA/cells")
        boundary = self.client.get("/v1/states/CA/boundaries")
        self.assertEqual(200, cells.status_code, cells.text)
        self.assertEqual(186, len(cells.json()["cells"]))
        self.assertFalse(any(cell["has_published_recommendation"] for cell in cells.json()["cells"]))
        self.assertEqual(200, boundary.status_code)
        self.assertEqual("state", boundary.json()["features"][0]["properties"]["kind"])

    def test_unvalidated_mixture_size_is_rejected(self) -> None:
        response = self.client.get("/v1/recommendations", params={
            "state": "MI",
            "planting_class": "herbaceous strip",
            "cell_id": "MI_0p5_42.0_-84.0",
            "mixture_size": 4,
        })
        self.assertEqual(422, response.status_code)
        self.assertEqual("invalid_request", response.json()["error"])

    def test_catalog_endpoints(self) -> None:
        classes = self.client.get("/v1/states/PA/planting-classes")
        cells = self.client.get("/v1/states/MD/cells")
        boundaries = self.client.get("/v1/states/MD/boundaries")
        self.assertEqual(200, classes.status_code)
        self.assertEqual({"herbaceous strip", "woody hedgerow"}, {
            row["planting_class"] for row in classes.json()["planting_classes"]
        })
        self.assertEqual(25, len(cells.json()["cells"]))
        self.assertEqual(5, sum(row["has_published_recommendation"] for row in cells.json()["cells"]))
        self.assertEqual(200, boundaries.status_code)
        self.assertGreater(len(boundaries.json()["features"]), 20)

    def test_chat_status_and_message_validation(self) -> None:
        status = self.client.get("/v1/chat/status")
        self.assertEqual(200, status.status_code)
        self.assertEqual("gpt-5.6-luna", status.json()["model"])
        invalid = self.client.post("/v1/chat", json={
            "messages": [{"role": "assistant", "content": "No user message"}], "context": {},
        })
        self.assertEqual(422, invalid.status_code)

    def test_validated_pipeline_policy_is_explicitly_non_live(self) -> None:
        full = self.client.get("/v1/pipeline/states/PA")
        fallback = self.client.get("/v1/pipeline/states/CT")
        limited = self.client.get("/v1/pipeline/states/HI")
        self.assertEqual(200, full.status_code, full.text)
        self.assertEqual("catboost", full.json()["selected_flower_model"]["method"])
        self.assertEqual("state_species_profile", full.json()["selected_bee_model"]["method"])
        self.assertIn("pending_release", full.json()["operational_mode"])
        self.assertEqual("national_species_profile", fallback.json()["selected_bee_model"]["method"])
        self.assertEqual("abstain_pending_data_acquisition", limited.json()["operational_mode"])
        self.assertIsNone(limited.json()["selected_flower_model"])

    def test_spatial_hierarchy_and_state_boundary(self) -> None:
        scenarios = [
            (39.7, -76.75, "county"),
            (37.95, -76.25, "ecological_region"),
            (39.392857, -79.35, "statewide"),
        ]
        for latitude, longitude, expected_level in scenarios:
            with self.subTest(level=expected_level):
                response = self.client.get("/v1/recommendations", params={
                    "state": "MD", "planting_class": "herbaceous strip",
                    "latitude": latitude, "longitude": longitude,
                })
                self.assertEqual(200, response.status_code, response.text)
                self.assertEqual(expected_level, response.json()["spatial_decision"]["level"])
        outside = self.client.get("/v1/recommendations", params={
            "state": "PA", "planting_class": "herbaceous strip",
            "latitude": 38.0, "longitude": -75.0,
        })
        self.assertEqual(422, outside.status_code)

    def test_local_frontend_cors(self) -> None:
        response = self.client.options("/v1/states", headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        })
        self.assertEqual(200, response.status_code)
        self.assertEqual("http://127.0.0.1:5173", response.headers["access-control-allow-origin"])

    def test_feedback_and_outcome_are_quarantined(self) -> None:
        recommendation = self.client.get("/v1/recommendations", params={
            "state": "PA", "planting_class": "herbaceous strip", "cell_id": "PA_0p5_40.0_-75.5",
        }).json()
        response = self.client.post("/v1/feedback", json={
            "release_id": recommendation["release"]["release_id"],
            "state_code": "PA", "cell_id": recommendation["request"]["resolved_cell_id"],
            "planting_class": recommendation["request"]["planting_class"],
            "selected_plants": [plant["scientific_name"] for plant in recommendation["recommendation"]["plants"]],
            "helpful_rating": 4, "intended_action": "planning", "comments": "API test",
        })
        self.assertEqual(201, response.status_code, response.text)
        self.assertEqual("recorded_unverified", response.json()["status"])
        outcome = self.client.post("/v1/outcomes", json={
            "feedback_id": response.json()["identifier"], "observation_date": "2026-08-17",
            "plants_attempted": [plant["scientific_name"] for plant in recommendation["recommendation"]["plants"]],
            "establishment_fraction": .75, "flowering_observed": True, "pollinators_observed": True,
            "observed_groups": ["bumble bee"], "notes": "Unverified test outcome",
        })
        self.assertEqual(201, outcome.status_code, outcome.text)
        self.assertEqual("recorded_unverified", outcome.json()["status"])


if __name__ == "__main__":
    unittest.main()
