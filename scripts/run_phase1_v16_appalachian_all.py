"""Retrain every retained Phase 1 model from scratch on V16.

Defaults in the source benchmark modules are left untouched.  This runner
redirects their input/output constants only for the reproducible V16 run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA_DIR = ROOT / "data" / "interim" / "factorized_model_v16_appalachian"
DATA_FILE = DATA_DIR / "factorized_event_records.csv.gz"
OUT = ROOT / "results" / "phase1_v16_appalachian_all"

import run_factorized_ai_experiment_v15 as tower
import run_phase1_shared_baselines_v1 as baselines
import run_phase1_shared_gnns_v1 as compact_gnns
import run_phase1_rich_gnns_v1 as rich_gnns


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Build V16 first: {DATA_FILE}")
    tower.DATA = DATA_DIR
    tower.OUT = OUT / "two_tower"
    baselines.DATA = DATA_FILE
    baselines.OUT = OUT / "baselines"
    compact_gnns.DATA = DATA_FILE
    compact_gnns.OUT = OUT / "compact_gnns"
    rich_gnns.DATA = DATA_FILE
    rich_gnns.OUT = OUT / "rich_gnns"

    runs = [
        ("two_tower", tower.main),
        ("baselines", baselines.main),
        ("compact_gnns", compact_gnns.main),
        ("rich_gnns", rich_gnns.main),
    ]
    completed = []
    for name, func in runs:
        print(f"START {name}", flush=True)
        func()
        completed.append(name)
        (OUT / "progress.json").write_text(json.dumps({"completed": completed, "total": len(runs)}, indent=2), encoding="utf-8")
        print(f"DONE {name}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "dataset": "factorized_model_v16_appalachian",
        "protocol": "same Phase 1 outer splits and model settings as V15; all models initialized from scratch",
        "models": ["strict evidence", "degree heuristic", "CatBoost", "two-tower neural", "compact GraphSAGE", "compact HGT", "rich GraphSAGE", "rich HGT"],
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
