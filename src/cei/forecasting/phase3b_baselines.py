"""Validation-only Phase 3B.1 baseline forecasts and metrics."""
import json
import math
import platform
from datetime import datetime, timedelta
from pathlib import Path

from cei.forecasting.phase3a import (
    TASKS, baseline_forecast, eligible_origin, forecast_blocks,
)


def load_baseline_config(path):
    with Path(path).open(encoding="utf-8") as source:
        return validate_baseline_config(json.load(source))


def validate_baseline_config(config):
    if not isinstance(config, dict):
        raise ValueError("Baseline configuration must be a JSON object")
    required = {"phase", "training_partition", "evaluation_partition",
                "mase_scaling_period_hours", "candidate_lookbacks_hours",
                "include_sensitivity"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Baseline configuration is missing: {sorted(missing)}")
    if config["phase"] != "3B.1":
        raise ValueError("Baseline configuration must be for Phase 3B.1")
    if config["training_partition"] != "train" or config["evaluation_partition"] != "validation":
        raise ValueError("Baseline evaluation must use train and validation only")
    if not isinstance(config["mase_scaling_period_hours"], int) or config["mase_scaling_period_hours"] < 1:
        raise ValueError("mase_scaling_period_hours must be a positive integer")
    lookbacks = config["candidate_lookbacks_hours"]
    if not isinstance(lookbacks, list) or not lookbacks or len(lookbacks) != len(set(lookbacks)):
        raise ValueError("candidate_lookbacks_hours must be a nonempty unique list")
    if any(not isinstance(value, int) or value < 1 for value in lookbacks):
        raise ValueError("candidate_lookbacks_hours must contain positive integers")
    if not isinstance(config["include_sensitivity"], bool):
        raise ValueError("include_sensitivity must be boolean")
    return config


def _origins(partition, task, phase3a_config):
    horizon = phase3a_config["horizons_hours"][task]
    if horizon != 1:
        return list(forecast_blocks(partition, task, phase3a_config))
    start, end = (datetime.fromisoformat(value)
                  for value in phase3a_config["partitions"][partition])
    return [(stamp - timedelta(hours=1), (stamp,)) for stamp in
            (start + timedelta(hours=offset)
             for offset in range(int((end - start).total_seconds() / 3600) + 1))]


def training_mase_scale(series, partition, period):
    pair_count, scale = training_mase_stats(series, partition, period)
    if pair_count == 0 or scale is None:
        raise ValueError("Training MASE scaling denominator is unavailable or zero")
    return scale


def training_mase_stats(series, partition, period):
    start, end = (datetime.fromisoformat(value) for value in partition)
    difference_total = 0.0
    pair_count = 0
    stamp = start + timedelta(hours=period)
    while stamp <= end:
        earlier = stamp - timedelta(hours=period)
        if (stamp in series and earlier in series and
                series[stamp] is not None and series[earlier] is not None):
            difference_total += abs(series[stamp] - series[earlier])
            pair_count += 1
        stamp += timedelta(hours=1)
    if pair_count == 0:
        return 0, None
    scale = difference_total / pair_count
    return pair_count, (None if math.isclose(difference_total, 0.0) else scale)


def evaluate_baseline(series, task, phase3a_config, baseline_config, lookback):
    method = phase3a_config["baseline_comparisons"][task]
    period_key = {
        "persistence": "persistence_period_hours",
        "daily_seasonal_naive": "daily_seasonal_period_hours",
        "weekly_seasonal_naive": "weekly_seasonal_period_hours",
    }[method]
    period = phase3a_config["baseline_settings"][period_key]
    common_history = max(lookback, period)
    origins = _origins(baseline_config["evaluation_partition"], task, phase3a_config)
    eligible = target_excluded = history_excluded = 0
    actuals = []
    predictions = []
    for origin, targets in origins:
        ok, no_history, no_target = eligible_origin(series, origin, targets, common_history)
        if not ok:
            if no_target:
                target_excluded += 1
            elif no_history:
                history_excluded += 1
            continue
        prediction = baseline_forecast(series, origin, targets, method, phase3a_config)
        if prediction is None:
            raise RuntimeError("Eligible origin produced no baseline forecast")
        actuals.extend(series[target] for target in targets)
        predictions.extend(prediction)
        eligible += 1
    if not actuals:
        raise ValueError(f"No eligible validation origins for {task} lookback {lookback}")
    valid_training_pair_count, scale = training_mase_stats(
        series, phase3a_config["partitions"][baseline_config["training_partition"]],
        baseline_config["mase_scaling_period_hours"])
    errors = [prediction - actual for prediction, actual in zip(predictions, actuals)]
    absolute = [abs(error) for error in errors]
    return {
        "task": task,
        "method": method,
        "lookback_hours": lookback,
        "registered_common_lookback_hours": common_history,
        "candidate_origins": len(origins),
        "eligible_origins": eligible,
        "eligible_target_hours": len(actuals),
        "missing_target_exclusions": target_excluded,
        "insufficient_history_exclusions": history_excluded,
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mean_error": sum(errors) / len(errors),
        "valid_training_pair_count": valid_training_pair_count,
        "mase_training_daily_naive": (sum(absolute) / len(absolute) / scale
                                       if scale is not None else None),
        "mase_scaling_denominator": scale,
    }


def _macro(rows):
    grouped = []
    for task in TASKS:
        for lookback in sorted({row["lookback_hours"] for row in rows}):
            selected = [row for row in rows if row["task"] == task and
                        row["lookback_hours"] == lookback]
            mase_values = [row["mase_training_daily_naive"] for row in selected
                           if row["mase_training_daily_naive"] is not None]
            grouped.append({
                "task": task,
                "lookback_hours": lookback,
                "building_count": len(selected),
                "mae_macro_mean": sum(row["mae"] for row in selected) / len(selected),
                "mae_median": sorted(row["mae"] for row in selected)[len(selected) // 2],
                "mae_range": [min(row["mae"] for row in selected), max(row["mae"] for row in selected)],
                "rmse_macro_mean": sum(row["rmse"] for row in selected) / len(selected),
                "mean_error_macro_mean": sum(row["mean_error"] for row in selected) / len(selected),
                "mase_macro_mean": (sum(mase_values) / len(mase_values)
                                    if mase_values else None),
            })
    return grouped


def evaluate_validation(readings, phase3a_config, baseline_config):
    validate_baseline_config(baseline_config)
    rows = []
    for building, series in readings.items():
        for task in TASKS:
            for lookback in baseline_config["candidate_lookbacks_hours"]:
                row = evaluate_baseline(series, task, phase3a_config, baseline_config, lookback)
                row["building_id"] = building
                rows.append(row)
    return {"rows": rows, "macro_summary": _macro(rows)}


def execution_metadata():
    return {"python_version": platform.python_version(), "platform": platform.platform(),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
