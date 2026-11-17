"""Run the locked Phase 2 rolling-year and leave-study-out tests on V16."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase2_three_network_decision_v1 as phase2
import run_phase2_robust_validation_v1 as robust


def main() -> int:
    data = ROOT / "data" / "interim" / "factorized_model_v16_appalachian" / "factorized_event_records.csv.gz"
    out = ROOT / "results" / "phase2_v16_appalachian_robust_validation"
    phase2.DATA = data
    robust.DATA = data
    robust.OUT = out
    robust.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
