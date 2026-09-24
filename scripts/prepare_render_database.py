from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "deploy" / "pollinator_forage_v40.sqlite3.gz"
MANIFEST = ROOT / "deploy" / "pollinator_forage_v40.manifest.json"
SPATIAL_CONTEXT = ROOT / "data" / "reference" / "spatial_context.json"
TARGET = Path(os.getenv(
    "POLLINATOR_DATABASE_PATH", ROOT / "data" / "app" / "pollinator_forage.sqlite3"
))
if not TARGET.is_absolute():
    TARGET = ROOT / TARGET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(ARTIFACT) != manifest["compressed_sha256"]:
        raise SystemExit("Deployment database archive failed checksum validation.")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    with gzip.open(ARTIFACT, "rb") as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    if sha256(temporary) != manifest["database_sha256"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit("Extracted deployment database failed checksum validation.")
    with closing(sqlite3.connect(temporary)) as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        release = db.execute(
            "SELECT release_id FROM data_releases WHERE is_current=1 AND status='published'"
        ).fetchone()
    if integrity != "ok" or not release or release[0] != manifest["release_id"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit("Extracted deployment database is not the expected published release.")
    spatial = json.loads(SPATIAL_CONTEXT.read_text(encoding="utf-8"))
    if spatial.get("release", {}).get("release_id") != release[0]:
        temporary.unlink(missing_ok=True)
        raise SystemExit("Spatial context does not match the published database release.")
    for state, item in spatial.get("states", {}).items():
        boundary = ROOT / item["boundary_path"]
        if not boundary.exists() or sha256(boundary) != item["boundary_sha256"]:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Boundary artifact failed validation for {state}.")
    temporary.replace(TARGET)
    print(f"Prepared validated database release {release[0]} at {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
