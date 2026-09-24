from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "data" / "reference" / "plant_profiles.json"


def load_profiles(path: Path | str = DEFAULT_PROFILES) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["scientific_name"].casefold(): item for item in payload["profiles"]}
