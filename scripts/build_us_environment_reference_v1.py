from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import shapefile
from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "us_environment_reference_v1.json"
USER_AGENT = "Villanova-Pollinator-US-Environment-Reference/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> dict:
    if path.exists() and path.stat().st_size:
        return {"status": "resumed", "source_url": url}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=240) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
        headers = {
            key.lower(): value for key, value in response.headers.items()
            if key.lower() in {"etag", "last-modified", "content-length", "content-type"}
        }
    temporary.replace(path)
    return {"status": "downloaded", "source_url": url, "headers": headers, "retrieved_at_utc": utc_now()}


def extract(zip_path: Path, directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        return
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(directory)


def bbox_contains(bbox, lon: float, lat: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def point_in_ring(lon: float, lat: float, points: list, start: int, end: int) -> bool:
    inside = False
    previous = points[end - 1]
    for current in points[start:end]:
        x1, y1 = previous[:2]
        x2, y2 = current[:2]
        if (y1 > lat) != (y2 > lat):
            crossing = (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-15) + x1
            if lon < crossing:
                inside = not inside
        previous = current
    return inside


def shape_contains(shape, lon: float, lat: float) -> bool:
    if not bbox_contains(shape.bbox, lon, lat):
        return False
    starts = list(shape.parts) + [len(shape.points)]
    # Even-odd rule works for multipart polygons and holes without relying on ring orientation.
    return sum(point_in_ring(lon, lat, shape.points, starts[i], starts[i + 1]) for i in range(len(starts) - 1)) % 2 == 1


def load_shapes(directory: Path) -> tuple[list[str], list[tuple[dict, object]]]:
    shp = next(directory.rglob("*.shp"))
    reader = shapefile.Reader(str(shp))
    fields = [field[0] for field in reader.fields[1:]]
    return fields, [(dict(zip(fields, item.record)), item.shape) for item in reader.iterShapeRecords()]


def first_field(record: dict, names: list[str]) -> str:
    for name in names:
        if name in record and record[name] not in {None, ""}:
            return str(record[name])
    return ""


def file_record(path: Path, role: str, **extra) -> dict:
    return {
        "relative_path": str(path.relative_to(ROOT)), "role": role,
        "bytes": path.stat().st_size, "sha256": sha256_path(path), **extra,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and lock U.S. state/ecoregion site reference")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Versioned JSON configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshot = ROOT / "data" / "snapshots" / config["snapshot_id"]
    manifest_path = snapshot / "manifest.json"
    lock_path = snapshot / "LOCKED.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures = []
        for item in manifest["files"]:
            path = ROOT / item["relative_path"]
            actual = sha256_path(path) if path.exists() else "MISSING"
            if actual != item["sha256"]:
                failures.append({"path": item["relative_path"], "actual": actual})
        if failures:
            raise RuntimeError(json.dumps(failures, indent=2))
        print(f"VERIFIED {config['snapshot_id']}: {len(manifest['files'])} files")
        return 0
    if lock_path.exists():
        raise RuntimeError("Reference snapshot is locked; use --verify or build a new version")
    raw = snapshot / "raw"
    extracted = snapshot / "extracted"
    derived = snapshot / "derived"
    raw.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    states_zip = raw / "cb_2025_us_state_500k.zip"
    eco_zip = raw / "NA_CEC_Eco_Level3.zip"
    state_download = download(config["sources"]["census_states"], states_zip)
    eco_download = download(config["sources"]["epa_level3_ecoregions"], eco_zip)
    states_dir = extracted / "census_states"
    eco_dir = extracted / "epa_level3_ecoregions"
    extract(states_zip, states_dir)
    extract(eco_zip, eco_dir)
    _, state_shapes = load_shapes(states_dir)
    _, eco_shapes = load_shapes(eco_dir)
    # The EPA/CEC ecoregion archive is projected, whereas NPN sites are WGS84
    # longitude/latitude. Transform points before polygon containment checks.
    eco_prj = next(eco_dir.rglob("*.prj"))
    eco_transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_wkt(eco_prj.read_text(encoding="utf-8")), always_xy=True)

    state_by_code = {
        first_field(record, ["STUSPS", "STATE_ABBR"]): shape
        for record, shape in state_shapes
        if first_field(record, ["STUSPS", "STATE_ABBR"])
    }
    sites = pd.read_csv(ROOT / config["site_inventory"])
    rows = []
    for index, site in enumerate(sites.itertuples(index=False), 1):
        lon, lat, state = float(site.longitude), float(site.latitude), str(site.state)
        eco_x, eco_y = eco_transformer.transform(lon, lat)
        state_shape = state_by_code.get(state)
        in_recorded_state = bool(state_shape and shape_contains(state_shape, lon, lat))
        containing_states = [code for code, shape in state_by_code.items() if shape_contains(shape, lon, lat)]
        eco_matches = []
        for record, shape in eco_shapes:
            if shape_contains(shape, eco_x, eco_y):
                eco_matches.append({
                    "code": first_field(record, ["NA_L3CODE", "US_L3CODE", "L3CODE"]),
                    "name": first_field(record, ["NA_L3NAME", "US_L3NAME", "L3NAME"]),
                })
        rows.append({
            **site._asdict(), "coordinate_in_recorded_state": in_recorded_state,
            "coordinate_containing_states": "|".join(sorted(containing_states)),
            "coordinate_qc": "pass" if in_recorded_state else "quarantine",
            "ecoregion_level3_code": "|".join(sorted({item["code"] for item in eco_matches if item["code"]})),
            "ecoregion_level3_name": "|".join(sorted({item["name"] for item in eco_matches if item["name"]})),
            "ecoregion_match_count": len(eco_matches),
        })
        if index % 1000 == 0:
            print(f"SPATIAL QC {index}/{len(sites)}", flush=True)
    joined = pd.DataFrame(rows)
    joined_path = derived / "usa_npn_sites_spatial_qc.csv"
    joined.to_csv(joined_path, index=False)
    summary = {
        "snapshot_id": config["snapshot_id"], "built_at_utc": utc_now(),
        "sites": len(joined), "coordinate_pass": int(joined.coordinate_in_recorded_state.sum()),
        "coordinate_quarantine": int((~joined.coordinate_in_recorded_state).sum()),
        "sites_with_one_level3_ecoregion": int(joined.ecoregion_match_count.eq(1).sum()),
        "sites_without_level3_ecoregion": int(joined.ecoregion_match_count.eq(0).sum()),
        "sites_with_multiple_level3_ecoregions": int(joined.ecoregion_match_count.gt(1).sum()),
        "rule": config["coordinate_rule"],
    }
    summary_path = snapshot / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = snapshot / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    config_copy = provenance / CONFIG_PATH.name
    script_copy = provenance / Path(__file__).name
    site_copy = provenance / Path(config["site_inventory"]).name
    shutil.copy2(CONFIG_PATH, config_copy)
    shutil.copy2(Path(__file__), script_copy)
    shutil.copy2(ROOT / config["site_inventory"], site_copy)
    extracted_files = sorted(path for path in extracted.rglob("*") if path.is_file())
    files = [
        file_record(config_copy, "frozen configuration"), file_record(script_copy, "frozen builder"),
        file_record(site_copy, "frozen USA-NPN site inventory"),
        file_record(states_zip, "raw Census state boundary archive", **state_download),
        file_record(eco_zip, "raw EPA/CEC Level III ecoregion archive", **eco_download),
        *(file_record(path, "extracted source component") for path in extracted_files),
        file_record(joined_path, "site coordinate QC and Level III ecoregion join"),
        file_record(summary_path, "spatial reference summary"),
    ]
    manifest = {"snapshot_id": config["snapshot_id"], "created_at_utc": utc_now(), "immutable": True, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_path.write_text(json.dumps({
        "snapshot_id": config["snapshot_id"], "locked_at_utc": utc_now(),
        "manifest_sha256": sha256_path(manifest_path),
        "rule": "Never edit this directory. Build a new version for every refresh.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"LOCKED {snapshot.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
