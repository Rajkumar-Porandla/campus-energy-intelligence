"""Run validation-only Phase 3B.1 baseline experiments once."""
import hashlib
import json
from pathlib import Path

from cei.forecasting.phase3a import load_experiment_config, validate_canonical
from cei.forecasting.phase3b_baselines import (
    evaluate_validation, execution_metadata, load_baseline_config,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE3A_CONFIG = ROOT / "src/cei/config/phase3a_experiment.json"
BASELINE_CONFIG = ROOT / "src/cei/config/phase3b_baseline_experiment.json"
DATA = ROOT / "data/processed/bdg2/panther"
OUTPUT = ROOT / "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    phase3a_config = load_experiment_config(PHASE3A_CONFIG)
    baseline_config = load_baseline_config(BASELINE_CONFIG)
    buildings = phase3a_config["core_building_ids"]
    if len(buildings) != 29 or any("Quintin" in building for building in buildings):
        raise ValueError("Baseline population must contain 29 core buildings and exclude Quintin")
    core_path, weather_path = DATA / "electricity_core.csv", DATA / "weather.csv"
    inputs = (core_path, weather_path, DATA / "electricity_sensitivity_quintin.csv",
              ROOT / "data/raw/bdg2/panther/electricity.csv",
              ROOT / "data/raw/bdg2/panther/metadata.csv",
              ROOT / "data/raw/bdg2/panther/weather.csv")
    before = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    readings, timestamps = validate_canonical(core_path, weather_path, phase3a_config)
    results = evaluate_validation(readings, phase3a_config, baseline_config)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Input checksum changed during baseline evaluation")
    report = {
        "phase": "3B.1",
        "status": "validation_only",
        "configuration": baseline_config,
        "phase3a_configuration": phase3a_config,
        "execution": execution_metadata(),
        "canonical_validation": {"status": "passed", "timestamp_count": len(timestamps),
                                 "building_count": len(readings), "checksums_before": before,
                                 "checksums_after": after},
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(results['rows'])} rows)")


if __name__ == "__main__":
    main()
