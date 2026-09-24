from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from app_backend.chat_agent import GroundedChatAgent
from app_backend.database import DEFAULT_DB
from app_backend.recommendation_service import RecommendationService


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                type="function_call", name="get_recommendation", call_id="call-1",
                arguments=json.dumps({
                    "state_code": "PA", "planting_class": "herbaceous strip",
                    "latitude": 40.0, "longitude": -75.3,
                }),
            )
            return SimpleNamespace(output=[tool_call], output_text="")
        return SimpleNamespace(
            output=[],
            output_text="The validated engine returned five plants using local-grid evidence.",
        )


class GroundedChatAgentTests(unittest.TestCase):
    def test_recommendation_is_returned_from_validated_service(self) -> None:
        fake = SimpleNamespace(responses=FakeResponses())
        agent = GroundedChatAgent(RecommendationService(DEFAULT_DB), client=fake, model="test-model")
        result = agent.chat(
            [{"role": "user", "content": "Recommend plants here."}],
            {"state_code": "PA", "planting_class": "herbaceous strip", "latitude": 40.0, "longitude": -75.3},
        )
        self.assertEqual([{"name": "get_recommendation"}], result["tools_used"])
        self.assertEqual(5, result["recommendation"]["recommendation"]["mixture_size"])
        self.assertEqual("local_grid", result["recommendation"]["spatial_decision"]["level"])
        self.assertFalse(result["recommendation"]["recommendation"]["ai_used_operationally"])
        self.assertFalse(fake.responses.calls[0]["store"])
        self.assertEqual("low", fake.responses.calls[0]["reasoning"]["effort"])

    def test_agent_tools_are_read_only_and_bounded(self) -> None:
        fake = SimpleNamespace(responses=FakeResponses())
        agent = GroundedChatAgent(RecommendationService(DEFAULT_DB), client=fake)
        states, recommendation = agent._execute_tool("get_available_states", {})
        state_codes = {item["state_code"] for item in states["states"]}
        self.assertEqual(50, len(state_codes))
        self.assertTrue({"MD", "MI", "PA"}.issubset(state_codes))
        self.assertIsNone(recommendation)
        with self.assertRaises(ValueError):
            agent._execute_tool("save_feedback", {})


if __name__ == "__main__":
    unittest.main()
