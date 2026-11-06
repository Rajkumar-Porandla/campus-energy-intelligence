"""Run validation-only Phase 3B.3 unsupervised anomaly detection experiments."""
import hashlib
import json
from pathlib import Path

from cei.anomaly.phase3b_anomaly import (
    FEATURE_MODES, MODEL_NAMES, OPERATING_POINTS, evaluate_anomaly_experiment,
    execution_metadata, load_anomaly_config, load_sensitivity_series,
)
from cei.forecasting.phase3a import load_experiment_config, validate_canonical

ROOT = Path(__file__).resolve().parents[1]
PHASE3A_CONFIG = ROOT / "src/cei/config/phase3a_experiment.json"
ANOMALY_CONFIG = ROOT / "src/cei/config/phase3b3_anomaly_experiment.json"
DATA = ROOT / "data/processed/bdg2/panther"
PREVIOUS_OUTPUT = ROOT / "data/reports/bdg2/panther/phase3b3_anomaly_validation.json"
OUTPUT = ROOT / "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    phase3a = load_experiment_config(PHASE3A_CONFIG)
    anomaly = load_anomaly_config(ANOMALY_CONFIG)
    buildings = phase3a["core_building_ids"]
    if len(buildings) != 29 or any("Quintin" in building for building in buildings):
        raise ValueError("Core population must contain 29 buildings and exclude Quintin")
    core_path = DATA / "electricity_core.csv"
    weather_path = DATA / "weather.csv"
    quintin_path = DATA / "electricity_sensitivity_quintin.csv"
    inputs = (
        core_path, weather_path, quintin_path,
        ROOT / "data/raw/bdg2/panther/electricity.csv",
        ROOT / "data/raw/bdg2/panther/metadata.csv",
        ROOT / "data/raw/bdg2/panther/weather.csv",
    )
    before = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    readings, timestamps = validate_canonical(core_path, weather_path, phase3a)
    sensitivity_series = None
    if anomaly["include_sensitivity_separately"]:
        sensitivity_series = load_sensitivity_series(
            quintin_path, anomaly["sensitivity_building_id"],
            phase3a["site_id"], timestamps)
    results = evaluate_anomaly_experiment(readings, phase3a, anomaly, sensitivity_series)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Input checksum changed during anomaly evaluation")
    zero_shift = results["zero_frequency_shift_summary"]
    train_zero_rate = zero_shift["core_train_valid_zero_per_1000_macro_mean"]
    val_zero_rate = zero_shift["core_validation_valid_zero_per_1000_macro_mean"]
    report = {
        "phase": "3B.3",
        "report_version": anomaly.get("report_version", "3B.3-remediated-v1"),
        "supersedes_report": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
        "remediated_audit_findings": [
            "B1: Conditional event duration macro aggregation excluding zero-event buildings (null/None when event_count == 0) with buildings_with_events_count.",
            "B2: Full empirical multiplicity LOF novelty scoring with training-only positive reachability floor and leave-one-out training calibration.",
            "F1: Explicit detector feature-space scope and dimensionality metadata distinguishing 1D univariate statistical_zscore from 3D multivariate isolation_forest and lof_novelty in seasonal_deviation mode.",
            "F2: Building-level and macro quantification of the 2016-to-2017 valid-zero regime shift and report-level limitation documentation.",
        ],
        "status": "validation_only",
        "ground_truth_labels": "none",
        "operating_point_semantics": (
            "Training-score upper-tail quantile operating points (99.5th, 99th, 98th percentiles / "
            "nominal 0.5%, 1%, 2% training alert rates) frozen before validation; not validated "
            "detection accuracy, expected future alert rates, or proof of true anomalies."
        ),
        "feature_space_semantics": anomaly.get("feature_space_semantics", {}),
        "scientific_limitations": [
            (
                "No confirmed ground-truth anomaly labels exist in BDG2 Panther; flagged hours and "
                "contiguous events represent statistical or geometric departures from 2016 training "
                "behavior, not verified equipment faults or operational anomalies."
            ),
            (
                f"Major 2016-to-2017 valid-zero regime shift: the {zero_shift['core_building_count']} "
                f"core Panther education buildings average {train_zero_rate:.2f} valid zero hours per "
                f"1,000 observed hours in 2016 (training) versus {val_zero_rate:.2f} valid zero hours "
                "per 1,000 observed hours in H1 2017 (validation). Because all models, scalers, and "
                "quantile thresholds are strictly frozen on 2016 training data without refitting on "
                "validation, validation alert rates reflect both hourly departures and this structural "
                "operating-regime shift."
            ),
            (
                "In seasonal_deviation mode, statistical_zscore is an intentional 1D univariate "
                "slot-deviation baseline (|slot_zscore|), whereas isolation_forest and lof_novelty "
                "evaluate the 3D joint vector [slot_zscore, lag_1h_scaled_diff, lag_24h_scaled_diff] "
                "over the identical eligible timestamp mask."
            ),
            (
                "Discrete meter quantization and score ties at upper-tail training quantiles can make "
                "exact 5.0, 10.0, or 20.0 per 1,000-hour realized training alert rates mathematically "
                "impossible under strict exceedance (score > threshold); both nominal_training_flagged_per_1000_hours "
                "and training_tied_at_threshold_hours are reported per building."
            ),
        ],
        "configuration": anomaly,
        "phase3a_configuration": phase3a,
        "execution": execution_metadata(),
        "feature_modes": list(FEATURE_MODES),
        "anomaly_models": list(MODEL_NAMES),
        "operating_points": list(OPERATING_POINTS),
        "canonical_validation": {
            "status": "passed",
            "timestamp_count": len(timestamps),
            "core_building_count": len(readings),
            "checksums_before": before,
            "checksums_after": after,
        },
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(results['rows'])} core rows)")


if __name__ == "__main__":
    main()

