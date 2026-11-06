"""Training-only Phase 3B.2 candidate forecasts and validation metrics."""
import json
import math
import platform
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from cei.forecasting.phase3a import TASKS, eligible_origin, forecast_blocks
from cei.forecasting.phase3b_baselines import training_mase_stats

MODEL_NAMES = ("sarima", "random_forest", "xgboost")
METHODS = {
    "hourly": ("persistence", 1),
    "daily_profile": ("daily_seasonal_naive", 24),
    "weekly_profile": ("weekly_seasonal_naive", 168),
}


def load_candidate_config(path):
    with Path(path).open(encoding="utf-8") as source:
        return validate_candidate_config(json.load(source))


def validate_candidate_config(config):
    required = {"phase", "training_partition", "evaluation_partition",
                "candidate_lookbacks_hours", "test_evaluation", "random_seed",
                "multi_step_strategy", "feature_lags_hours", "models"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Candidate configuration is incomplete")
    if config["phase"] != "3B.2" or config["training_partition"] != "train" \
            or config["evaluation_partition"] != "validation":
        raise ValueError("Candidate evaluation must use train and validation only")
    if config["test_evaluation"] != "not_run":
        raise ValueError("Candidate test evaluation must remain not_run")
    if config["multi_step_strategy"] != "frozen_origin_recursive":
        raise ValueError("Unsupported multi-step forecasting strategy")
    lookbacks = config["candidate_lookbacks_hours"]
    if not lookbacks or len(lookbacks) != len(set(lookbacks)) or any(
            not isinstance(value, int) or value < 1 for value in lookbacks):
        raise ValueError("candidate_lookbacks_hours must contain unique positive integers")
    lags = config["feature_lags_hours"]
    if not lags or len(lags) != len(set(lags)) or any(
            not isinstance(value, int) or value < 1 for value in lags):
        raise ValueError("feature_lags_hours must contain unique positive integers")
    if set(config["models"]) != set(MODEL_NAMES):
        raise ValueError("models must define SARIMA, Random Forest, and XGBoost")
    return config


def feature_lags(config, lookback):
    return tuple(lag for lag in config["feature_lags_hours"] if lag < lookback)


def _features(series, stamp, lags):
    values = [series.get(stamp - timedelta(hours=lag)) for lag in lags]
    return None if any(value is None for value in values) else values


def training_examples(series, partition, lags):
    start, end = (datetime.fromisoformat(value) for value in partition)
    first = start + timedelta(hours=max(lags))
    rows, targets = [], []
    stamp = first
    while stamp <= end:
        values = _features(series, stamp, lags)
        target = series.get(stamp)
        if values is not None and target is not None:
            rows.append(values)
            targets.append(target)
        stamp += timedelta(hours=1)
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def recursive_forecast(model, series, origin, targets, lags):
    available = {stamp: value for stamp, value in series.items() if stamp <= origin}
    predictions = []
    for target in targets:
        values = _features(available, target, lags)
        if values is None:
            raise RuntimeError("Eligible origin lacks a causal model feature")
        prediction = float(model.predict(np.asarray([values], dtype=float))[0])
        predictions.append(prediction)
        available[target] = prediction
    return tuple(predictions)


def _fit_tabular(name, series, partition, lags, settings, seed):
    features, targets = training_examples(series, partition, lags)
    if not len(targets):
        raise ValueError("No complete training rows for candidate features")
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=seed, **settings)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=seed, objective="reg:squarederror",
                             verbosity=0, **settings)
    return model.fit(features, targets)


def _fit_sarima(series, partition, task, settings):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    start, end = (datetime.fromisoformat(value) for value in partition)
    values = []
    stamp = start
    while stamp <= end:
        value = series.get(stamp)
        values.append(np.nan if value is None else value)
        stamp += timedelta(hours=1)
    model = SARIMAX(np.asarray(values, dtype=float), order=tuple(settings["order"]),
                    seasonal_order=tuple(settings["seasonal_order_by_task"][task]),
                    trend="c", enforce_stationarity=False,
                    enforce_invertibility=False)
    return model.fit(disp=False, maxiter=settings["maxiter"])


def _origins(partition, task, config):
    horizon = config["horizons_hours"][task]
    if horizon != 1:
        return list(forecast_blocks(partition, task, config))
    start, end = (datetime.fromisoformat(value) for value in config["partitions"][partition])
    return [(stamp - timedelta(hours=1), (stamp,)) for stamp in
            (start + timedelta(hours=offset)
             for offset in range(int((end - start).total_seconds() / 3600) + 1))]


def _sarima_forecaster(fitted, series, train_partition, evaluation_partition):
    start = datetime.fromisoformat(train_partition[0])
    _, validation_end = (datetime.fromisoformat(value) for value in evaluation_partition)
    values = []
    index_by_stamp = {}
    stamp = start
    while stamp <= validation_end:
        index_by_stamp[stamp] = len(values)
        value = series.get(stamp)
        values.append(np.nan if value is None else value)
        stamp += timedelta(hours=1)
    applied = fitted.apply(np.asarray(values, dtype=float)).filter_results
    return applied, index_by_stamp


def sarima_recursive_forecast(forecaster, origin, targets):
    applied, index_by_stamp = forecaster
    if any(target <= origin for target in targets):
        raise ValueError("Target timestamps must strictly follow origin")
    state = applied.filtered_state[:, index_by_stamp[origin]].copy()
    design = applied.design[:, :, 0]
    transition = applied.transition[:, :, 0]
    state_intercept = applied.state_intercept[:, 0]
    obs_intercept = applied.obs_intercept[:, 0]
    predictions = []
    for _ in targets:
        state = transition @ state + state_intercept
        predictions.append(float((design @ state + obs_intercept)[0]))
    return tuple(predictions)


def _batch_recursive_forecast(model, series, eligible_blocks, lags):
    if not eligible_blocks:
        return []
    horizon = len(eligible_blocks[0][1])
    histories = [
        {origin - timedelta(hours=step): series[origin - timedelta(hours=step)]
         for step in range(max(lags))}
        for origin, _ in eligible_blocks
    ]
    step_predictions = []
    for step in range(horizon):
        batch = []
        for index, (_, targets) in enumerate(eligible_blocks):
            values = _features(histories[index], targets[step], lags)
            if values is None:
                raise RuntimeError("Eligible origin lacks a causal model feature")
            batch.append(values)
        predicted = model.predict(np.asarray(batch, dtype=float)).tolist()
        for index, (_, targets) in enumerate(eligible_blocks):
            histories[index][targets[step]] = float(predicted[index])
        step_predictions.append(predicted)
    return [
        tuple(float(step_predictions[step][index]) for step in range(horizon))
        for index in range(len(eligible_blocks))
    ]


def candidate_origin_forecasts(series, task, lookback, model_name, model, phase3a,
                               candidate_config):
    method, period = METHODS[task]
    common_history = max(lookback, period)
    origins = _origins(candidate_config["evaluation_partition"], task, phase3a)
    eligible_blocks = []
    target_excluded = history_excluded = 0
    for origin, targets in origins:
        ok, no_history, no_target = eligible_origin(series, origin, targets, common_history)
        if not ok:
            if no_target:
                target_excluded += 1
            elif no_history:
                history_excluded += 1
            continue
        eligible_blocks.append((origin, targets))
    lags = feature_lags(candidate_config, lookback)
    if model_name == "sarima":
        block_predictions = [sarima_recursive_forecast(model, origin, targets)
                             for origin, targets in eligible_blocks]
    else:
        block_predictions = _batch_recursive_forecast(model, series, eligible_blocks, lags)
    return {
        "common_history": common_history,
        "candidate_origins": len(origins),
        "eligible_blocks": eligible_blocks,
        "block_predictions": block_predictions,
        "missing_target_exclusions": target_excluded,
        "insufficient_history_exclusions": history_excluded,
    }


def _metrics(actuals, predictions, scale):
    errors = [prediction - actual for prediction, actual in zip(predictions, actuals)]
    absolute = [abs(error) for error in errors]
    return {
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mean_error": sum(errors) / len(errors),
        "mase_training_daily_naive": (sum(absolute) / len(absolute) / scale
                                       if scale is not None else None),
    }


def _candidate_row(series, building, task, lookback, model_name, model, phase3a,
                   candidate_config, scale, pair_count):
    forecast = candidate_origin_forecasts(
        series, task, lookback, model_name, model, phase3a, candidate_config)
    actuals = [series[target] for _, targets in forecast["eligible_blocks"] for target in targets]
    predictions = [value for block in forecast["block_predictions"] for value in block]
    if not actuals:
        raise ValueError(f"No eligible validation origins for {building} {task} {lookback}")
    row = {"building_id": building, "model": model_name, "task": task,
           "lookback_hours": lookback,
           "registered_common_lookback_hours": forecast["common_history"],
           "candidate_origins": forecast["candidate_origins"],
           "eligible_origins": len(forecast["eligible_blocks"]),
           "eligible_target_hours": len(actuals),
           "missing_target_exclusions": forecast["missing_target_exclusions"],
           "insufficient_history_exclusions": forecast["insufficient_history_exclusions"],
           "valid_training_pair_count": pair_count,
           "mase_scaling_denominator": scale}
    row.update(_metrics(actuals, predictions, scale))
    return row


def evaluate_candidates(readings, phase3a_config, candidate_config):
    validate_candidate_config(candidate_config)
    training_partition = phase3a_config["partitions"][candidate_config["training_partition"]]
    evaluation_partition = phase3a_config["partitions"][candidate_config["evaluation_partition"]]
    baseline_scale_period = 24
    rows = []
    for building, series in readings.items():
        scales = training_mase_stats(series, training_partition, baseline_scale_period)
        sarima_models = {}
        for task in TASKS:
            fitted = _fit_sarima(
                series, training_partition, task, candidate_config["models"]["sarima"])
            sarima_models[task] = _sarima_forecaster(
                fitted, series, training_partition, evaluation_partition)
        for lookback in candidate_config["candidate_lookbacks_hours"]:
            lags = feature_lags(candidate_config, lookback)
            models = {
                name: _fit_tabular(name, series, training_partition, lags,
                                   candidate_config["models"][name], candidate_config["random_seed"])
                for name in ("random_forest", "xgboost")
            }
            for task in TASKS:
                for name in MODEL_NAMES:
                    model = sarima_models[task] if name == "sarima" else models[name]
                    rows.append(_candidate_row(
                        series, building, task, lookback, name, model, phase3a_config,
                        candidate_config, scales[1], scales[0]))
    return {"rows": rows, "macro_summary": _macro(rows)}


def _summary(selected, model):
    metrics = {key: [row[key] for row in selected if row["model"] == model]
               for key in ("mae", "rmse", "mean_error", "mase_training_daily_naive")}
    return {f"{key}_macro_mean": (sum(values) / len(values) if values else None)
            for key, values in metrics.items()} | {
                "mae_median": np.median(metrics["mae"]).item(),
                "mae_range": [min(metrics["mae"]), max(metrics["mae"])],
            }


def _macro(rows):
    grouped = []
    for model in MODEL_NAMES:
        for task in TASKS:
            for lookback in sorted({row["lookback_hours"] for row in rows}):
                selected = [row for row in rows if row["model"] == model and
                            row["task"] == task and row["lookback_hours"] == lookback]
                summary = {"model": model, "task": task, "lookback_hours": lookback,
                           "building_count": len(selected)}
                summary.update(_summary(selected, model))
                grouped.append(summary)
    return grouped


def compare_with_baseline(candidate_rows, baseline_rows):
    baseline_index = {(row["building_id"], row["task"], row["lookback_hours"]): row
                      for row in baseline_rows}
    comparisons = []
    for candidate in candidate_rows:
        key = (candidate["building_id"], candidate["task"], candidate["lookback_hours"])
        baseline = baseline_index[key]
        for field in ("candidate_origins", "eligible_origins", "eligible_target_hours",
                      "missing_target_exclusions", "insufficient_history_exclusions"):
            if candidate[field] != baseline[field]:
                raise ValueError(f"Candidate mask differs from baseline for {key}")
        comparisons.append({
            "building_id": candidate["building_id"], "model": candidate["model"],
            "task": candidate["task"], "lookback_hours": candidate["lookback_hours"],
            "baseline_method": baseline["method"],
            "candidate_mae": candidate["mae"], "baseline_mae": baseline["mae"],
            "mae_delta": candidate["mae"] - baseline["mae"],
            "candidate_rmse": candidate["rmse"], "baseline_rmse": baseline["rmse"],
            "rmse_delta": candidate["rmse"] - baseline["rmse"],
            "candidate_mean_error": candidate["mean_error"],
            "baseline_mean_error": baseline["mean_error"],
            "candidate_mase": candidate["mase_training_daily_naive"],
            "baseline_mase": baseline["mase_training_daily_naive"],
            "mase_delta": (candidate["mase_training_daily_naive"] - baseline["mase_training_daily_naive"]
                           if candidate["mase_training_daily_naive"] is not None and
                           baseline["mase_training_daily_naive"] is not None else None),
            "mae_improved": candidate["mae"] < baseline["mae"],
        })
    return comparisons


def execution_metadata():
    return {"python_version": platform.python_version(),
            "platform": platform.platform(),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
