import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cei.anomaly.phase3b_anomaly import (
    FEATURE_MODES, MODEL_NAMES, OPERATING_POINTS, contiguous_event_durations,
    detector_feature_metadata, eligible_anomaly_timestamps,
    evaluate_anomaly_experiment, evaluate_building_mode, event_summary,
    extract_features, fit_and_score_detector, fit_feature_extractor,
    load_anomaly_config, macro_summary, validate_anomaly_config,
)
from cei.forecasting.phase3a import load_experiment_config

PHASE3A_CONFIG_PATH = Path(__file__).parents[1] / "src/cei/config/phase3a_experiment.json"
ANOMALY_CONFIG_PATH = Path(__file__).parents[1] / "src/cei/config/phase3b3_anomaly_experiment.json"
FORBIDDEN_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "auc"}


def _setup():
    phase3a = load_experiment_config(PHASE3A_CONFIG_PATH)
    phase3a["core_building_ids"] = ["core_a", "core_b"]
    phase3a["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-03-03 23:00:00"],
        "validation": ["2016-03-04 00:00:00", "2016-03-10 23:00:00"],
        "test": ["2016-03-11 00:00:00", "2016-03-12 23:00:00"],
    }
    anomaly = load_anomaly_config(ANOMALY_CONFIG_PATH)
    start = datetime(2016, 1, 1)
    hours = int((datetime(2016, 3, 12, 23) - start).total_seconds() / 3600) + 1
    series_a = {}
    series_b = {}
    for h in range(hours):
        stamp = start + timedelta(hours=h)
        if stamp.hour in (1, 2, 3, 4):
            series_a[stamp] = 0.0
            series_b[stamp] = 0.0
        else:
            diurnal = 25.0 * (1.0 - math.cos(2.0 * math.pi * stamp.hour / 24.0))
            weekly = 6.0 * math.sin(2.0 * math.pi * (h % 168) / 168.0)
            ripple = 2.0 * math.sin(h / 5.3)
            base = max(0.5, diurnal + weekly + ripple)
            series_a[stamp] = round(base, 4)
            series_b[stamp] = round(base * 1.4 + 1.5 * math.cos(h / 7.1), 4)
    return phase3a, anomaly, {"core_a": series_a, "core_b": series_b}


def test_anomaly_config_forbids_test_evaluation_and_validates_quantiles():
    _, anomaly, _ = _setup()
    bad_test = dict(anomaly, evaluation_partition="test")
    with pytest.raises(ValueError, match="train and validation only"):
        validate_anomaly_config(bad_test)

    bad_flag = dict(anomaly, test_evaluation="run")
    with pytest.raises(ValueError, match="not_run"):
        validate_anomaly_config(bad_flag)

    bad_quantiles = dict(anomaly, operating_point_quantiles={
        "0.5pct": 0.995, "1.0pct": 0.99, "2.0pct": 0.50})
    with pytest.raises(ValueError, match="upper-tail quantiles"):
        validate_anomaly_config(bad_quantiles)


def test_threshold_calibration_is_training_only_and_unaffected_by_validation_perturbations():
    phase3a, anomaly, readings = _setup()
    base_series = dict(readings["core_a"])
    perturbed_series = dict(base_series)
    spike_stamp = datetime(2016, 3, 7, 12)
    perturbed_series[spike_stamp] = 9999.0

    for mode in FEATURE_MODES:
        rows_base, flags_base, _ = evaluate_building_mode(
            "core_a", base_series, phase3a, anomaly, mode)
        rows_pert, flags_pert, _ = evaluate_building_mode(
            "core_a", perturbed_series, phase3a, anomaly, mode)
        for r_base, r_pert in zip(rows_base, rows_pert):
            assert r_base["model"] == r_pert["model"]
            assert r_base["operating_point"] == r_pert["operating_point"]
            assert r_base["threshold_value"] == r_pert["threshold_value"]
            assert r_base["training_flagged_hours"] == r_pert["training_flagged_hours"]
            assert r_base["train_score_summary"] == r_pert["train_score_summary"]
        for model_name in MODEL_NAMES:
            assert spike_stamp in flags_pert[(mode, model_name, "0.5pct")]
            assert spike_stamp not in flags_base[(mode, model_name, "0.5pct")]

        # Also verify point-by-point validation scores outside the causal lag neighborhood of spike_stamp
        val_info = eligible_anomaly_timestamps(
            base_series, phase3a["partitions"]["validation"], mode, anomaly["deviation_lags_hours"])
        train_info = eligible_anomaly_timestamps(
            base_series, phase3a["partitions"]["train"], mode, anomaly["deviation_lags_hours"])
        ext_base = fit_feature_extractor(base_series, train_info["eligible_timestamps"], mode, anomaly)
        ext_pert = fit_feature_extractor(perturbed_series, train_info["eligible_timestamps"], mode, anomaly)
        X_train = extract_features(base_series, train_info["eligible_timestamps"], ext_base)
        X_val_base = extract_features(base_series, val_info["eligible_timestamps"], ext_base)
        X_val_pert = extract_features(perturbed_series, val_info["eligible_timestamps"], ext_pert)
        affected_stamps = {
            spike_stamp,
            spike_stamp + timedelta(hours=1),
            spike_stamp + timedelta(hours=24),
        }
        for model_name in MODEL_NAMES:
            _, s_val_base = fit_and_score_detector(model_name, X_train, X_val_base, ext_base, anomaly)
            _, s_val_pert = fit_and_score_detector(model_name, X_train, X_val_pert, ext_pert, anomaly)
            for idx, stamp in enumerate(val_info["eligible_timestamps"]):
                if stamp not in affected_stamps:
                    assert s_val_base[idx] == s_val_pert[idx]


def test_missing_values_and_valid_zeros_are_preserved_and_masks_match_across_models():
    phase3a, anomaly, readings = _setup()
    series = dict(readings["core_a"])
    # Inject one missing hour in train and one missing hour in validation
    series[datetime(2016, 1, 10, 10)] = None
    series[datetime(2016, 3, 5, 10)] = None

    raw_rows, _, _ = evaluate_building_mode("core_a", series, phase3a, anomaly, "raw_consumption")
    dev_rows, _, _ = evaluate_building_mode("core_a", series, phase3a, anomaly, "seasonal_deviation")

    assert len(raw_rows) == len(MODEL_NAMES) * len(OPERATING_POINTS)
    assert len(dev_rows) == len(MODEL_NAMES) * len(OPERATING_POINTS)

    for row in raw_rows:
        assert row["train_missing_hours"] == 1
        assert row["train_insufficient_history_exclusions"] == 0
        assert row["train_eligible_hours"] == 63 * 24 - 1
        assert row["validation_missing_hours"] == 1
        assert row["validation_insufficient_history_exclusions"] == 0
        assert row["validation_eligible_hours"] == 7 * 24 - 1
        assert row["validation_valid_zero_hours"] == 7 * 4
        assert row["validation_flagged_zero_hours"] == 0

    for row in dev_rows:
        # Missing at t excludes t (missing_hours=1), and in train excludes the first 24 hours
        # prior to start (24) plus t+1h and t+24h (2) => 26 total insufficient_history_exclusions.
        # In validation, causal lags at the partition boundary reach back into train, so only
        # t+1h and t+24h (2) are excluded for insufficient history.
        assert row["train_missing_hours"] == 1
        assert row["train_insufficient_history_exclusions"] == 26
        assert row["train_eligible_hours"] == 63 * 24 - 27
        assert row["validation_missing_hours"] == 1
        assert row["validation_insufficient_history_exclusions"] == 2
        assert row["validation_eligible_hours"] == 7 * 24 - 3
        assert row["validation_flagged_zero_hours"] == 0


def test_event_grouping_respects_hourly_contiguity_and_missing_gaps():
    t0 = datetime(2017, 1, 10, 8)
    flagged = [
        t0,
        t0 + timedelta(hours=1),
        t0 + timedelta(hours=2),
        t0 + timedelta(hours=4),
        t0 + timedelta(hours=7),
        t0 + timedelta(hours=8),
    ]
    assert contiguous_event_durations(flagged) == [3, 1, 2]
    summary = event_summary(flagged)
    assert summary == {
        "event_count": 3,
        "single_hour_event_count": 1,
        "mean_event_duration_hours": 2.0,
        "median_event_duration_hours": 2.0,
        "max_event_duration_hours": 3,
    }
    assert event_summary([]) == {
        "event_count": 0,
        "single_hour_event_count": 0,
        "mean_event_duration_hours": None,
        "median_event_duration_hours": None,
        "max_event_duration_hours": None,
    }


def test_zero_event_buildings_return_null_duration_and_do_not_dilute_macro_duration():
    import json
    phase3a, anomaly, readings = _setup()
    # Building core_a gets a 3-hour anomaly event in validation; core_b has no validation spike
    spike_t0 = datetime(2016, 3, 8, 12)
    for offset in range(3):
        readings["core_a"][spike_t0 + timedelta(hours=offset)] = 850.0

    results = evaluate_anomaly_experiment(readings, phase3a, anomaly)
    # Verify JSON serialization with allow_nan=False succeeds
    serialized = json.dumps(results, allow_nan=False)
    assert "null" in serialized

    for m in results["macro_summary"]:
        selected = [
            r for r in results["rows"]
            if r["feature_mode"] == m["feature_mode"]
            and r["model"] == m["model"]
            and r["operating_point"] == m["operating_point"]
        ]
        nonzero = [r for r in selected if r["event_count"] > 0]
        zero_bldgs = [r for r in selected if r["event_count"] == 0]
        for zb in zero_bldgs:
            assert zb["mean_event_duration_hours"] is None
            assert zb["median_event_duration_hours"] is None
            assert zb["max_event_duration_hours"] is None
        assert m["building_count"] == 2
        assert m["buildings_with_events_count"] == len(nonzero)
        assert m["zero_event_buildings_count"] == len(zero_bldgs)
        if nonzero:
            expected_mean = sum(r["mean_event_duration_hours"] for r in nonzero) / len(nonzero)
            assert math.isclose(m["mean_event_duration_macro_mean"], expected_mean, rel_tol=1e-12)
            assert m["mean_event_duration_macro_mean"] >= 1.0
        else:
            assert m["mean_event_duration_macro_mean"] is None
            assert m["max_event_duration_max"] is None


def test_lof_repeated_zeros_and_duplicate_heavy_series_do_not_collapse_quantiles():
    phase3a, anomaly, _ = _setup()
    start = datetime(2016, 1, 1)
    hours = int((datetime(2016, 3, 12, 23) - start).total_seconds() / 3600) + 1
    dup_series = {}
    # Construct a realistic bimodal series where ~37.5% of hours are exact 0.0 (9 hours/day)
    # and positive hours are quantized to 0.5 kWh steps around 180-240 kWh, plus rare training tail points
    for h in range(hours):
        stamp = start + timedelta(hours=h)
        if stamp.hour < 9:
            dup_series[stamp] = 0.0
        else:
            cyc = 200.0 + 25.0 * math.sin(2.0 * math.pi * stamp.hour / 24.0) + 8.0 * math.cos(h / 11.0)
            dup_series[stamp] = round(cyc * 2.0) / 2.0
    # Add 25 distinct upper-tail training observations so the top 2% tail has realistic variation
    for idx in range(25):
        dup_series[datetime(2016, 1, 5, 12) + timedelta(days=idx * 2)] = 280.0 + idx * 3.5
    # Add 2 validation anomalies
    dup_series[datetime(2016, 3, 6, 14)] = 420.0
    dup_series[datetime(2016, 3, 9, 15)] = 450.0

    rows, _, _ = evaluate_building_mode("dup_bldg", dup_series, phase3a, anomaly, "raw_consumption")
    lof_rows = {r["operating_point"]: r for r in rows if r["model"] == "lof_novelty"}
    assert lof_rows["0.5pct"]["threshold_value"] > lof_rows["1.0pct"]["threshold_value"]
    assert lof_rows["1.0pct"]["threshold_value"] > lof_rows["2.0pct"]["threshold_value"]
    for op, r in lof_rows.items():
        assert r["validation_flagged_zero_hours"] == 0
        assert r["training_flagged_hours"] >= 5
        assert r["validation_flagged_hours"] >= 2
        assert r["training_tied_at_threshold_hours"] >= 0
        assert r["training_unique_score_count"] > 10

    # Also verify explicit tie reporting when an upper-tail score plateau straddles the 2.0pct quantile
    plateau_series = dict(dup_series)
    for idx in range(45):
        plateau_series[datetime(2016, 1, 2, 12) + timedelta(hours=idx * 24)] = 310.0
    plateau_rows, _, _ = evaluate_building_mode(
        "plateau_bldg", plateau_series, phase3a, anomaly, "raw_consumption"
    )
    plateau_lof_2pct = next(
        r for r in plateau_rows if r["model"] == "lof_novelty" and r["operating_point"] == "2.0pct"
    )
    assert plateau_lof_2pct["training_tied_at_threshold_hours"] >= 2


def test_feature_space_metadata_and_zero_frequency_shift_diagnostics():
    phase3a, anomaly, readings = _setup()
    # Give core_a 0 zeros in validation while keeping zeros in training to verify shift math
    for stamp in list(readings["core_a"]):
        if stamp >= datetime(2016, 3, 4, 0) and readings["core_a"][stamp] == 0.0:
            readings["core_a"][stamp] = 18.0
    quintin = dict(readings["core_b"])
    results = evaluate_anomaly_experiment(readings, phase3a, anomaly, quintin)

    for r in results["rows"]:
        expected_meta = detector_feature_metadata(r["model"], r["feature_mode"])
        assert r["detector_feature_scope"] == expected_meta["detector_feature_scope"]
        assert r["detector_feature_dimension"] == expected_meta["detector_feature_dimension"]
        assert r["detector_feature_names"] == expected_meta["detector_feature_names"]
        if r["feature_mode"] == "seasonal_deviation":
            if r["model"] == "statistical_zscore":
                assert r["detector_feature_dimension"] == 1
            else:
                assert r["detector_feature_dimension"] == 3

    shift = results["zero_frequency_shift_summary"]
    assert shift["core_building_count"] == 2
    assert "sensitivity_quintin" in shift
    b_map = {b["building_id"]: b for b in shift["core_buildings"]}
    assert b_map["core_a"]["train_valid_zero_hours"] == 63 * 4
    assert b_map["core_a"]["validation_valid_zero_hours"] == 0
    assert b_map["core_a"]["valid_zero_per_1000_delta_validation_minus_train"] < -150.0


def test_quintin_is_excluded_from_core_and_no_supervised_metrics_exist():
    phase3a, anomaly, readings = _setup()
    # Inject a simultaneous spike in both core buildings to verify coincidence counting
    simultaneous_spike = datetime(2016, 3, 8, 15)
    readings["core_a"][simultaneous_spike] = 500.0
    readings["core_b"][simultaneous_spike] = 500.0
    quintin_series = dict(readings["core_a"])
    quintin_series[simultaneous_spike] = 500.0

    results = evaluate_anomaly_experiment(readings, phase3a, anomaly, quintin_series)
    assert len(results["rows"]) == 2 * len(FEATURE_MODES) * len(MODEL_NAMES) * len(OPERATING_POINTS)
    assert all("Quintin" not in r["building_id"] for r in results["rows"])
    assert all(m["building_count"] == 2 for m in results["macro_summary"])
    assert "sensitivity_quintin" in results
    assert results["sensitivity_quintin"]["building_id"] == "Panther_education_Quintin"
    assert len(results["sensitivity_quintin"]["rows"]) == (
        len(FEATURE_MODES) * len(MODEL_NAMES) * len(OPERATING_POINTS)
    )

    coincidences = [
        c for c in results["coincidence_summary"]
        if c["feature_mode"] == "raw_consumption" and c["operating_point"] == "0.5pct"
    ]
    assert len(coincidences) == 3
    for c in coincidences:
        assert c["max_simultaneous_buildings_flagged"] == 2
        assert c["hours_with_multi_building_coincidence"] >= 1

    def _collect_keys(obj):
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(str(k).lower())
                keys.update(_collect_keys(v))
        elif isinstance(obj, list):
            for item in obj:
                keys.update(_collect_keys(item))
        return keys

    all_keys = _collect_keys(results)
    assert FORBIDDEN_METRIC_KEYS.isdisjoint(all_keys)
