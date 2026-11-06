import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cei.forecasting.phase3a import (
    baseline_forecast, coverage_report, eligible_origin, forecast_blocks,
    load_experiment_config, validate_canonical,
)
from cei.forecasting.phase3b_baselines import (
    _macro, evaluate_baseline, evaluate_validation, training_mase_scale,
    training_mase_stats, validate_baseline_config,
)
from cei.forecasting.phase3b_candidates import (
    MODEL_NAMES, _batch_recursive_forecast, _fit_sarima, _fit_tabular,
    _sarima_forecaster, candidate_origin_forecasts, compare_with_baseline,
    evaluate_candidates, feature_lags, load_candidate_config,
    recursive_forecast, sarima_recursive_forecast, training_examples,
    validate_candidate_config,
)

CONFIG_PATH = Path(__file__).parents[1] / "src/cei/config/phase3a_experiment.json"
CANDIDATE_CONFIG_PATH = Path(__file__).parents[1] / "src/cei/config/phase3b2_candidate_experiment.json"


def _config():
    return load_experiment_config(CONFIG_PATH)


def test_daily_blocks_cross_boundary_and_are_non_overlapping():
    config = _config()
    blocks = list(forecast_blocks("validation", "daily_profile", config))
    assert blocks[0][0] == datetime(2016, 12, 31, 23)
    assert blocks[0][1][0] == datetime(2017, 1, 1)
    assert blocks[0][1][-1] == datetime(2017, 1, 1, 23)
    assert blocks[1][0] == blocks[0][1][-1]


def test_config_values_drive_forecast_generation():
    config = _config()
    config["horizons_hours"]["daily_profile"] = 12
    blocks = list(forecast_blocks("validation", "daily_profile", config))
    assert len(blocks[0][1]) == 12
    assert blocks[1][1][0] == blocks[0][1][-1] + timedelta(hours=1)


def test_invalid_or_incomplete_config_fails_clearly():
    config = _config()
    del config["partitions"]
    with pytest.raises(ValueError, match="missing:.*partitions"):
        list(forecast_blocks("train", "hourly", config))


def test_history_and_missing_target_exclusions_are_distinguished():
    origin = datetime(2016, 1, 1, 1)
    targets = (origin + timedelta(hours=1),)
    series = {origin: 0.0}
    assert eligible_origin(series, origin, targets, 2) == (False, True, True)
    series[targets[0]] = 4.0
    assert eligible_origin(series, origin, targets, 1) == (True, False, False)


def test_baselines_use_configured_period_and_only_available_history():
    config = _config()
    origin = datetime(2016, 1, 8)
    targets = (origin + timedelta(hours=1), origin + timedelta(hours=2))
    series = {origin - timedelta(hours=i): float(i) for i in range(0, 170)}
    assert baseline_forecast(series, origin, targets, "persistence", config) == (0.0, 0.0)
    assert baseline_forecast(series, origin, targets, "daily_seasonal_naive", config) == (23.0, 22.0)
    assert baseline_forecast(series, origin, targets, "weekly_seasonal_naive", config) == (167.0, 166.0)
    config["baseline_settings"]["daily_seasonal_period_hours"] = 12
    assert baseline_forecast(series, origin, targets, "daily_seasonal_naive", config) == (11.0, 10.0)


def test_weather_schema_requires_primary_columns_and_accepts_missing_values(tmp_path):
    config = _config()
    config["core_building_ids"] = ["core"]
    config["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-01-01 01:00:00"],
        "validation": ["2016-01-01 02:00:00", "2016-01-01 03:00:00"],
        "test": ["2016-01-01 04:00:00", "2016-01-01 05:00:00"],
    }
    electricity = tmp_path / "electricity.csv"
    with electricity.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "building_id", "site_id", "energy_consumption"])
        for hour in range(6):
            writer.writerow([f"2016-01-01 {hour:02}:00:00", "core", "Panther", "0"])
    weather = tmp_path / "weather.csv"
    columns = ["timestamp", "site_id", *config["required_primary_weather_columns"]]
    with weather.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for hour in range(6):
            values = ["" if hour == 0 else "1"] * 5
            writer.writerow([f"2016-01-01 {hour:02}:00:00", "Panther", *values])
    validate_canonical(electricity, weather, config)

    with weather.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns[:-1])
        for hour in range(6):
            writer.writerow([f"2016-01-01 {hour:02}:00:00", "Panther", "", "1", "1", "1"])
    with pytest.raises(ValueError, match="Canonical weather columns are incomplete"):
        validate_canonical(electricity, weather, config)


def test_coverage_reports_reduction_from_shortest_method_valid_lookback():
    config = _config()
    start = datetime(2016, 1, 1)
    hours = int((datetime(2017, 12, 31, 23) - start).total_seconds() / 3600) + 1
    series = {start + timedelta(hours=i): 0.0 for i in range(hours)}
    rows = coverage_report({"core-building": series}, config)
    assert len(rows) == 3 * 3 * 3
    assert all(row["building_id"] != "Quintin" for row in rows)
    weekly = [row for row in rows if row["task"] == "weekly_profile"]
    assert {row["lookback_hours"] for row in weekly} == {24, 48, 168}
    assert {row["registered_common_lookback_hours"] for row in weekly} == {168}
    assert {row["shortest_method_valid_lookback_hours"] for row in weekly} == {168}
    assert all(row["coverage_reduction_origins"] == 0 for row in weekly)
    assert all(row["coverage_reduction_pct"] == 0 for row in weekly)
    assert all("coverage_reduction_pct" in row for row in rows)


def test_reported_exclusion_precedence_is_exclusive():
    # Coverage counts missing targets first; history is counted only with complete targets.
    config = _config()
    config["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-01-01 00:00:00"],
        "validation": ["2016-01-01 01:00:00", "2016-01-01 01:00:00"],
        "test": ["2016-01-01 02:00:00", "2016-01-01 02:00:00"],
    }
    config["horizons_hours"] = {"hourly": 1, "daily_profile": 1, "weekly_profile": 1}
    start = datetime(2016, 1, 1)
    rows = coverage_report({"core": {start: None, start + timedelta(hours=1): 0.0,
                                     start + timedelta(hours=2): 0.0}}, config)
    train = next(row for row in rows if row["partition"] == "train" and row["task"] == "hourly"
                 and row["lookback_hours"] == 24)
    assert train["missing_target_exclusions"] == 1
    assert train["insufficient_history_exclusions"] == 0


def test_baseline_metrics_use_training_daily_scaling_and_preserve_zero():
    phase3a = _config()
    phase3a["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-01-02 23:00:00"],
        "validation": ["2016-01-03 00:00:00", "2016-01-03 01:00:00"],
        "test": ["2016-01-03 02:00:00", "2016-01-03 03:00:00"],
    }
    baseline = {"phase": "3B.1", "training_partition": "train",
                "evaluation_partition": "validation", "mase_scaling_period_hours": 24,
                "candidate_lookbacks_hours": [24], "include_sensitivity": False}
    start = datetime(2016, 1, 1)
    series = {start + timedelta(hours=i): float(i // 24) for i in range(74)}
    for hour in range(47, 51):
        series[start + timedelta(hours=hour)] = 0.0
    assert training_mase_scale(series, phase3a["partitions"]["train"], 24) == 23 / 24
    result = evaluate_baseline(series, "hourly", phase3a, baseline, 24)
    assert result["eligible_origins"] == 2
    assert result["valid_training_pair_count"] == 24
    assert result["mase_training_daily_naive"] == 0.0


def test_mase_training_stats_exclude_missing_pairs():
    start = datetime(2016, 1, 1)
    series = {start + timedelta(hours=24): 2.0,
              start + timedelta(hours=25): None,
              start + timedelta(hours=48): 5.0}
    assert training_mase_stats(
        series, ["2016-01-01 00:00:00", "2016-01-03 00:00:00"], 24) == (1, 3.0)


def test_mase_is_undefined_for_zero_or_missing_training_denominator():
    phase3a = _config()
    phase3a["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-01-02 23:00:00"],
        "validation": ["2016-01-03 00:00:00", "2016-01-03 00:00:00"],
        "test": ["2016-01-03 01:00:00", "2016-01-03 01:00:00"],
    }
    baseline = {"phase": "3B.1", "training_partition": "train",
                "evaluation_partition": "validation", "mase_scaling_period_hours": 24,
                "candidate_lookbacks_hours": [24], "include_sensitivity": False}
    start = datetime(2016, 1, 1)
    constant_series = {start + timedelta(hours=i): 1.0 for i in range(73)}
    zero_result = evaluate_baseline(constant_series, "hourly", phase3a, baseline, 24)
    assert zero_result["valid_training_pair_count"] == 24
    assert zero_result["mase_training_daily_naive"] is None
    assert zero_result["mase_scaling_denominator"] is None

    missing_series = dict(constant_series)
    for hour in range(24):
        missing_series[start + timedelta(hours=hour)] = None
    missing_result = evaluate_baseline(missing_series, "hourly", phase3a, baseline, 24)
    assert missing_result["valid_training_pair_count"] == 0
    assert missing_result["mase_training_daily_naive"] is None
    assert missing_result["mase_scaling_denominator"] is None


def test_macro_mase_handles_undefined_building_scales():
    rows = [{"task": task, "lookback_hours": 24, "mae": 1.0, "rmse": 1.0,
             "mean_error": 0.0, "mase_training_daily_naive": None}
            for task in ("hourly", "daily_profile", "weekly_profile")]
    summaries = _macro(rows)
    assert len(summaries) == 3
    assert all(summary["mase_macro_mean"] is None for summary in summaries)


def test_baseline_config_forbids_test_evaluation():
    with pytest.raises(ValueError, match="train and validation only"):
        validate_baseline_config({"phase": "3B.1", "training_partition": "train",
                                 "evaluation_partition": "test", "mase_scaling_period_hours": 24,
                                 "candidate_lookbacks_hours": [24], "include_sensitivity": False})


def test_candidate_features_are_causal_and_exclude_missing_training_rows():
    start = datetime(2016, 1, 1)
    series = {start + timedelta(hours=i): float(i) for i in range(7)}
    series[start + timedelta(hours=2)] = None
    lags = (1, 2)
    features, targets = training_examples(
        series, ["2016-01-01 00:00:00", "2016-01-01 06:00:00"], lags)
    assert features.tolist() == [[4.0, 3.0], [5.0, 4.0]]
    assert targets.tolist() == [5.0, 6.0]
    assert feature_lags({"feature_lags_hours": [1, 2, 4]}, 2) == (1,)


def test_recursive_candidate_forecast_never_reads_future_actuals():
    class StubModel:
        def predict(self, features):
            return features[:, 0] + 1.0

    start = datetime(2016, 1, 1)
    origin = start + timedelta(hours=2)
    targets = (start + timedelta(hours=3), start + timedelta(hours=4))
    series = {
        start: 0.0,
        start + timedelta(hours=1): 1.0,
        origin: 2.0,
        targets[0]: 9999.0,
        targets[1]: 9999.0,
    }
    predictions = recursive_forecast(StubModel(), series, origin, targets, (1,))
    batch_predictions = _batch_recursive_forecast(
        StubModel(), series, [(origin, targets)], (1,))
    assert predictions == (3.0, 4.0)
    assert batch_predictions == [(3.0, 4.0)]


def test_candidate_config_forbids_test_evaluation():
    with pytest.raises(ValueError, match="train and validation only"):
        validate_candidate_config({"phase": "3B.2", "training_partition": "train",
                                   "evaluation_partition": "test", "test_evaluation": "not_run",
                                   "multi_step_strategy": "frozen_origin_recursive",
                                   "candidate_lookbacks_hours": [24], "random_seed": 42,
                                   "feature_lags_hours": [1], "models": {
                                       "sarima": {}, "random_forest": {}, "xgboost": {}}})
    with pytest.raises(ValueError, match="Unsupported multi-step forecasting strategy"):
        validate_candidate_config({"phase": "3B.2", "training_partition": "train",
                                   "evaluation_partition": "validation", "test_evaluation": "not_run",
                                   "multi_step_strategy": "frozen_origin_direct_repeat",
                                   "candidate_lookbacks_hours": [24], "random_seed": 42,
                                   "feature_lags_hours": [1], "models": {
                                       "sarima": {}, "random_forest": {}, "xgboost": {}}})


def _synthetic_candidate_setup():
    import math
    phase3a = _config()
    phase3a["core_building_ids"] = ["core"]
    phase3a["partitions"] = {
        "train": ["2016-01-01 00:00:00", "2016-01-21 23:00:00"],
        "validation": ["2016-01-22 00:00:00", "2016-02-11 23:00:00"],
        "test": ["2016-02-12 00:00:00", "2016-02-13 23:00:00"],
    }
    candidate = load_candidate_config(CANDIDATE_CONFIG_PATH)
    candidate["candidate_lookbacks_hours"] = [24, 168]
    start = datetime(2016, 1, 1)
    hours = int((datetime(2016, 2, 13, 23) - start).total_seconds() / 3600) + 1
    series = {
        start + timedelta(hours=h): 20.0 + 10.0 * math.sin(h / 3.8) + 4.0 * math.cos(h / 11.0)
        for h in range(hours)
    }
    return phase3a, candidate, series


def test_evaluated_path_produces_non_repeated_24_and_168_step_forecasts():
    phase3a, candidate, series = _synthetic_candidate_setup()
    train_partition = phase3a["partitions"]["train"]
    val_partition = phase3a["partitions"]["validation"]
    lags = feature_lags(candidate, 168)
    models = {
        "sarima": _sarima_forecaster(
            _fit_sarima(series, train_partition, "daily_profile", candidate["models"]["sarima"]),
            series, train_partition, val_partition),
        "random_forest": _fit_tabular(
            "random_forest", series, train_partition, lags,
            candidate["models"]["random_forest"], candidate["random_seed"]),
        "xgboost": _fit_tabular(
            "xgboost", series, train_partition, lags,
            candidate["models"]["xgboost"], candidate["random_seed"]),
    }
    for task, expected_horizon in (("daily_profile", 24), ("weekly_profile", 168)):
        for name in MODEL_NAMES:
            forecast = candidate_origin_forecasts(
                series, task, 168, name, models[name], phase3a, candidate)
            assert len(forecast["eligible_blocks"]) == 21 * 24 // expected_horizon
            for (origin, targets), block in zip(
                    forecast["eligible_blocks"], forecast["block_predictions"]):
                assert len(targets) == expected_horizon
                assert len(block) == expected_horizon
                assert len(set(block)) > 1
                if name != "sarima":
                    assert block == recursive_forecast(
                        models[name], series, origin, targets, lags)


def test_post_origin_observations_cannot_influence_evaluated_forecasts_and_later_origins_update():
    phase3a, candidate, series = _synthetic_candidate_setup()
    train_partition = phase3a["partitions"]["train"]
    val_partition = phase3a["partitions"]["validation"]
    first_origin = datetime(2016, 1, 21, 23)

    for task, lookback, horizon in (("daily_profile", 24, 24), ("weekly_profile", 168, 168)):
        lags = feature_lags(candidate, lookback)
        second_origin = first_origin + timedelta(hours=horizon)
        perturbed = dict(series)
        for step in range(1, horizon + 1):
            perturbed[first_origin + timedelta(hours=step)] += 9999.0

        for name in MODEL_NAMES:
            if name == "sarima":
                fitted = _fit_sarima(
                    series, train_partition, task, candidate["models"]["sarima"])
                model_a = _sarima_forecaster(fitted, series, train_partition, val_partition)
                model_b = _sarima_forecaster(fitted, perturbed, train_partition, val_partition)
                truncated = {k: v for k, v in series.items() if k <= second_origin}
                model_second_trunc = _sarima_forecaster(
                    fitted, truncated, train_partition,
                    [val_partition[0], second_origin.isoformat(sep=" ")])
                second_targets = tuple(
                    second_origin + timedelta(hours=step) for step in range(1, horizon + 1))
                assert sarima_recursive_forecast(
                    model_a, second_origin, second_targets) == sarima_recursive_forecast(
                        model_second_trunc, second_origin, second_targets)
            else:
                fitted = _fit_tabular(
                    name, series, train_partition, lags,
                    candidate["models"][name], candidate["random_seed"])
                model_a = model_b = fitted
            out_a = candidate_origin_forecasts(
                series, task, lookback, name, model_a, phase3a, candidate)
            out_b = candidate_origin_forecasts(
                perturbed, task, lookback, name, model_b, phase3a, candidate)
            assert out_a["eligible_blocks"][0][0] == first_origin
            assert len(out_a["block_predictions"][0]) == horizon
            assert out_a["block_predictions"][0] == out_b["block_predictions"][0]
            assert out_a["eligible_blocks"][1][0] == second_origin
            assert out_a["block_predictions"][1] != out_b["block_predictions"][1]


def test_missing_targets_valid_zeros_and_shared_masks_preserved_in_candidate_evaluation():
    phase3a, candidate, series = _synthetic_candidate_setup()
    # Set valid zeros on the first validation day and a missing target on the second day
    series[datetime(2016, 1, 22, 0)] = 0.0
    series[datetime(2016, 1, 22, 5)] = 0.0
    series[datetime(2016, 1, 23, 4)] = None
    baseline = {"phase": "3B.1", "training_partition": "train",
                "evaluation_partition": "validation", "mase_scaling_period_hours": 24,
                "candidate_lookbacks_hours": [24, 168], "include_sensitivity": False}
    baseline_results = evaluate_validation({"core": series}, phase3a, baseline)
    candidate_results = evaluate_candidates({"core": series}, phase3a, candidate)
    comparisons = compare_with_baseline(candidate_results["rows"], baseline_results["rows"])
    assert len(comparisons) == len(MODEL_NAMES) * 3 * 2
    daily_rows = [r for r in candidate_results["rows"]
                  if r["task"] == "daily_profile" and r["lookback_hours"] == 24]
    assert len(daily_rows) == 3
    for row in daily_rows:
        assert row["candidate_origins"] == 21
        assert row["missing_target_exclusions"] == 1
        assert row["insufficient_history_exclusions"] == 1
        assert row["eligible_origins"] == 19
        assert row["eligible_target_hours"] == 19 * 24
