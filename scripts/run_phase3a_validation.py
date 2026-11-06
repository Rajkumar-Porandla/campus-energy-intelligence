"""Validate canonical BDG2 inputs and write Phase 3A coverage once."""
import hashlib
import json
from pathlib import Path

from cei.forecasting.phase3a import coverage_report, load_experiment_config, validate_canonical

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/cei/config/phase3a_experiment.json"
DATA = ROOT / "data/processed/bdg2/panther"
OUTPUT = ROOT / "data/reports/bdg2/panther/phase3a_corrected_validation.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config = load_experiment_config(CONFIG)
    if len(config["core_building_ids"]) != 29 or any(
            "Quintin" in building for building in config["core_building_ids"]):
        raise ValueError("Configuration must contain 29 core buildings and exclude Quintin")
    core_path, weather_path = DATA / "electricity_core.csv", DATA / "weather.csv"
    inputs = (core_path, weather_path, DATA / "electricity_sensitivity_quintin.csv",
              ROOT / "data/raw/bdg2/panther/electricity.csv",
              ROOT / "data/raw/bdg2/panther/metadata.csv",
              ROOT / "data/raw/bdg2/panther/weather.csv")
    before = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    readings, timestamps = validate_canonical(core_path, weather_path, config)
    rows = coverage_report(readings, config)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Canonical data checksum changed during read-only validation")
    report = {
        "configuration": config,
        "coverage_exclusion_count_precedence": (
            "Exclusive counts: an origin with any missing target is counted as missing_target; "
            "insufficient_history is counted only when all targets are present."
        ),
        "canonical_validation": {"status": "passed", "timestamp_count": len(timestamps),
                                 "building_count": len(readings), "checksums_before": before,
                                 "checksums_after": after},
        "coverage": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(rows)} building/task/partition/lookback rows)")


if __name__ == "__main__":
    main()
