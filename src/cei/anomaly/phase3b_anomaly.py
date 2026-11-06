"""Training-calibrated Phase 3B.3 unsupervised anomaly detection and validation diagnostics."""
import csv
import json
import math
import platform
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from cei.forecasting.phase3a import ELECTRICITY_COLUMNS

FEATURE_MODES = ("raw_consumption", "seasonal_deviation")
MODEL_NAMES = ("statistical_zscore", "isolation_forest", "lof_novelty")
OPERATING_POINTS = ("0.5pct", "1.0pct", "2.0pct")


def load_anomaly_config(path):
    with Path(path).open(encoding="utf-8") as source:
        return validate_anomaly_config(json.load(source))


def validate_anomaly_config(config):
    required = {
        "phase", "training_partition", "evaluation_partition", "test_evaluation",
        "random_seed", "include_sensitivity_separately", "sensitivity_building_id",
        "feature_modes", "seasonal_reference_period_hours", "deviation_lags_hours",
        "operating_point_quantiles", "models",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Anomaly configuration is incomplete")
    if (config["phase"] != "3B.3" or config["training_partition"] != "train"
            or config["evaluation_partition"] != "validation"):
        raise ValueError("Anomaly evaluation must use train and validation only")
    if config["test_evaluation"] != "not_run":
        raise ValueError("Anomaly test evaluation must remain not_run")
    if not isinstance(config["include_sensitivity_separately"], bool):
        raise ValueError("include_sensitivity_separately must be boolean")
    if tuple(config["feature_modes"]) != FEATURE_MODES:
        raise ValueError(f"feature_modes must be {list(FEATURE_MODES)}")
    if config["seasonal_reference_period_hours"] != 168:
        raise ValueError("seasonal_reference_period_hours must be 168")
    lags = config["deviation_lags_hours"]
    if not isinstance(lags, list) or not lags or any(not isinstance(v, int) or v < 1 for v in lags):
        raise ValueError("deviation_lags_hours must contain positive integers")
    quantiles = config["operating_point_quantiles"]
    if not isinstance(quantiles, dict) or set(quantiles) != set(OPERATING_POINTS):
        raise ValueError("operating_point_quantiles must define 0.5pct, 1.0pct, and 2.0pct")
    if any(not isinstance(q, (int, float)) or not (0.9 < float(q) < 1.0) for q in quantiles.values()):
        raise ValueError("operating_point_quantiles must be upper-tail quantiles in (0.9, 1.0)")
    if not isinstance(config["models"], dict) or set(config["models"]) != set(MODEL_NAMES):
        raise ValueError(f"models must define {list(MODEL_NAMES)}")
    z_cfg = config["models"]["statistical_zscore"]
    if z_cfg.get("feature_scope") != "univariate_primary_feature":
        raise ValueError("statistical_zscore must declare feature_scope='univariate_primary_feature'")
    for m_name in ("isolation_forest", "lof_novelty"):
        if config["models"][m_name].get("feature_scope") != "full_mode_feature_vector":
            raise ValueError(f"{m_name} must declare feature_scope='full_mode_feature_vector'")
    lof_cfg = config["models"]["lof_novelty"]
    if lof_cfg.get("duplicate_handling") != "full_empirical_multiplicity_with_positive_reachability_floor":
        raise ValueError(
            "lof_novelty must declare duplicate_handling="
            "'full_empirical_multiplicity_with_positive_reachability_floor'"
        )
    if float(lof_cfg.get("min_reachability_floor", 0.0)) <= 0.0:
        raise ValueError("lof_novelty min_reachability_floor must be positive")
    return config


def detector_feature_metadata(model_name, feature_mode):
    if feature_mode == "raw_consumption":
        return {
            "detector_feature_scope": (
                "univariate_primary_feature"
                if model_name == "statistical_zscore"
                else "full_mode_feature_vector"
            ),
            "detector_feature_dimension": 1,
            "detector_feature_names": ["energy_consumption"],
        }
    if model_name == "statistical_zscore":
        return {
            "detector_feature_scope": "univariate_primary_feature",
            "detector_feature_dimension": 1,
            "detector_feature_names": ["slot_zscore"],
        }
    return {
        "detector_feature_scope": "full_mode_feature_vector",
        "detector_feature_dimension": 3,
        "detector_feature_names": ["slot_zscore", "lag_1h_scaled_diff", "lag_24h_scaled_diff"],
    }


def load_sensitivity_series(path, expected_building_id, expected_site_id, expected_timestamps):
    series = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not ELECTRICITY_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Sensitivity electricity columns are incomplete")
        for row in reader:
            if row["building_id"] != expected_building_id or row["site_id"] != expected_site_id:
                raise ValueError(f"Unexpected sensitivity building key: {row['building_id']}")
            stamp = datetime.fromisoformat(row["timestamp"])
            if stamp in series:
                raise ValueError(f"Duplicate sensitivity timestamp: {stamp}")
            raw = row["energy_consumption"].strip()
            value = None if raw == "" else float(raw)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Non-finite sensitivity reading at {stamp}")
            series[stamp] = value
    if sorted(series) != list(expected_timestamps):
        raise ValueError("Sensitivity electricity index does not match canonical timestamps")
    return series


def _partition_timestamps(partition_bounds):
    start, end = (datetime.fromisoformat(value) for value in partition_bounds)
    hours = int((end - start).total_seconds() / 3600) + 1
    return [start + timedelta(hours=offset) for offset in range(hours)]


def eligible_anomaly_timestamps(series, partition_bounds, feature_mode, lags):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    timestamps = _partition_timestamps(partition_bounds)
    required_lags = () if feature_mode == "raw_consumption" else tuple(lags)
    eligible = []
    observed_hours = missing_hours = valid_zero_hours = history_excluded = 0
    for stamp in timestamps:
        value = series.get(stamp)
        if value is None:
            missing_hours += 1
            continue
        observed_hours += 1
        if value == 0.0:
            valid_zero_hours += 1
        if any(series.get(stamp - timedelta(hours=lag)) is None for lag in required_lags):
            history_excluded += 1
            continue
        eligible.append(stamp)
    return {
        "partition_hours": len(timestamps),
        "observed_hours": observed_hours,
        "missing_hours": missing_hours,
        "valid_zero_hours": valid_zero_hours,
        "insufficient_history_exclusions": history_excluded,
        "eligible_hours": len(eligible),
        "eligible_timestamps": tuple(eligible),
    }


def _hour_of_week(stamp):
    return stamp.weekday() * 24 + stamp.hour


def _robust_location_scale(values, epsilon):
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale <= epsilon:
        std = float(np.std(arr))
        scale = max(std, epsilon)
    return median, scale


def fit_feature_extractor(series, train_timestamps, feature_mode, config):
    if not train_timestamps:
        raise ValueError("No eligible training timestamps for anomaly feature extractor")
    settings = config["models"]["statistical_zscore"]
    epsilon = float(settings["epsilon"])
    floor_fraction = float(settings["scale_floor_fraction"])
    train_values = [series[stamp] for stamp in train_timestamps]
    global_median, global_scale = _robust_location_scale(train_values, epsilon)
    if feature_mode == "raw_consumption":
        return {
            "feature_mode": feature_mode,
            "lags": (),
            "global_median": global_median,
            "global_scale": global_scale,
            "epsilon": epsilon,
        }
    scale_floor = max(floor_fraction * global_scale, epsilon)
    slot_buckets = {slot: [] for slot in range(config["seasonal_reference_period_hours"])}
    for stamp in train_timestamps:
        slot_buckets[_hour_of_week(stamp)].append(series[stamp])
    slot_medians = {}
    slot_scales = {}
    for slot, bucket in slot_buckets.items():
        if bucket:
            med, sc = _robust_location_scale(bucket, epsilon)
            slot_medians[slot] = med
            slot_scales[slot] = max(sc, scale_floor)
        else:
            slot_medians[slot] = global_median
            slot_scales[slot] = max(global_scale, scale_floor)
    return {
        "feature_mode": feature_mode,
        "lags": tuple(config["deviation_lags_hours"]),
        "global_median": global_median,
        "global_scale": global_scale,
        "slot_medians": slot_medians,
        "slot_scales": slot_scales,
        "epsilon": epsilon,
    }


def extract_features(series, timestamps, extractor):
    mode = extractor["feature_mode"]
    rows = []
    for stamp in timestamps:
        value = series.get(stamp)
        if value is None:
            raise ValueError(f"Missing target reading at eligible timestamp {stamp}")
        if mode == "raw_consumption":
            rows.append([float(value)])
        else:
            slot = _hour_of_week(stamp)
            z_slot = (value - extractor["slot_medians"][slot]) / extractor["slot_scales"][slot]
            lag_diffs = []
            for lag in extractor["lags"]:
                lag_val = series.get(stamp - timedelta(hours=lag))
                if lag_val is None:
                    raise ValueError(f"Missing causal lag {lag}h at eligible timestamp {stamp}")
                lag_diffs.append((value - lag_val) / extractor["global_scale"])
            rows.append([float(z_slot), *[float(d) for d in lag_diffs]])
    return np.asarray(rows, dtype=float)


def _standardize_from_train(train_features, eval_features, epsilon):
    medians = np.median(train_features, axis=0)
    mads = 1.4826 * np.median(np.abs(train_features - medians), axis=0)
    stds = np.std(train_features, axis=0)
    scales = np.where(mads > epsilon, mads, np.maximum(stds, epsilon))
    return (train_features - medians) / scales, (eval_features - medians) / scales


def _lof_empirical_novelty_scores(scaled_train, scaled_eval, settings):
    from sklearn.neighbors import NearestNeighbors
    n_train = len(scaled_train)
    k = min(int(settings["n_neighbors"]), max(1, n_train - 1))
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=settings.get("metric", "minkowski"),
        p=int(settings.get("p", 2)),
        n_jobs=int(settings.get("n_jobs", 1)),
    )
    nn.fit(scaled_train)
    dist_tr_all, idx_tr_all = nn.kneighbors(scaled_train, n_neighbors=k + 1)
    dist_tr = np.empty((n_train, k), dtype=float)
    idx_tr = np.empty((n_train, k), dtype=int)
    for i in range(n_train):
        row_idx = idx_tr_all[i]
        row_dist = dist_tr_all[i]
        self_pos = np.where(row_idx == i)[0]
        if len(self_pos) > 0:
            pos = int(self_pos[0])
            idx_tr[i] = np.delete(row_idx, pos)
            dist_tr[i] = np.delete(row_dist, pos)
        else:
            idx_tr[i] = row_idx[:k]
            dist_tr[i] = row_dist[:k]

    min_floor = float(settings.get("min_reachability_floor", 1e-3))
    pos_k_dists = dist_tr[:, -1][dist_tr[:, -1] > 1e-9]
    reach_floor = (
        max(float(np.median(pos_k_dists)), min_floor)
        if len(pos_k_dists) > 0
        else min_floor
    )

    k_dist_train = np.maximum(dist_tr[:, -1], reach_floor)
    reach_tr = np.maximum(dist_tr, k_dist_train[idx_tr])
    lrd_train = 1.0 / np.mean(reach_tr, axis=1)
    lof_train = np.mean(lrd_train[idx_tr], axis=1) / lrd_train

    dist_val, idx_val = nn.kneighbors(scaled_eval, n_neighbors=k)
    reach_val = np.maximum(dist_val, k_dist_train[idx_val])
    lrd_val = 1.0 / np.mean(reach_val, axis=1)
    lof_val = np.mean(lrd_train[idx_val], axis=1) / lrd_val
    return lof_train.astype(float), lof_val.astype(float)


def fit_and_score_detector(model_name, train_features, eval_features, extractor, config):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown anomaly model: {model_name}")
    if model_name == "statistical_zscore":
        if extractor["feature_mode"] == "raw_consumption":
            med = extractor["global_median"]
            sc = extractor["global_scale"]
            train_scores = np.abs(train_features[:, 0] - med) / sc
            eval_scores = np.abs(eval_features[:, 0] - med) / sc
        else:
            train_scores = np.abs(train_features[:, 0])
            eval_scores = np.abs(eval_features[:, 0])
        return train_scores.astype(float), eval_scores.astype(float)

    if model_name == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        settings = {
            k: v for k, v in config["models"]["isolation_forest"].items()
            if k != "feature_scope"
        }
        detector = IsolationForest(random_state=config["random_seed"], **settings)
        detector.fit(train_features)
        return (-detector.score_samples(train_features)).astype(float), (
            -detector.score_samples(eval_features)).astype(float)

    settings = config["models"]["lof_novelty"]
    epsilon = float(config["models"]["statistical_zscore"]["epsilon"])
    scaled_train, scaled_eval = _standardize_from_train(train_features, eval_features, epsilon)
    return _lof_empirical_novelty_scores(scaled_train, scaled_eval, settings)


def score_distribution_summary(scores):
    arr = np.asarray(scores, dtype=float)
    quantiles = np.quantile(arr, [0.5, 0.95, 0.98, 0.99, 0.995], method="linear")
    return {
        "min": float(np.min(arr)),
        "median": float(quantiles[0]),
        "mean": float(np.mean(arr)),
        "p95": float(quantiles[1]),
        "p98": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "p99_5": float(quantiles[4]),
        "max": float(np.max(arr)),
    }


def contiguous_event_durations(flagged_timestamps):
    if not flagged_timestamps:
        return []
    ordered = sorted(flagged_timestamps)
    durations = []
    current = 1
    for prev, curr in zip(ordered, ordered[1:]):
        if curr - prev == timedelta(hours=1):
            current += 1
        else:
            durations.append(current)
            current = 1
    durations.append(current)
    return durations


def event_summary(flagged_timestamps):
    durations = contiguous_event_durations(flagged_timestamps)
    if not durations:
        return {
            "event_count": 0,
            "single_hour_event_count": 0,
            "mean_event_duration_hours": None,
            "median_event_duration_hours": None,
            "max_event_duration_hours": None,
        }
    return {
        "event_count": len(durations),
        "single_hour_event_count": sum(1 for d in durations if d == 1),
        "mean_event_duration_hours": float(sum(durations) / len(durations)),
        "median_event_duration_hours": float(np.median(durations)),
        "max_event_duration_hours": int(max(durations)),
    }


def evaluate_building_mode(building_id, series, phase3a_config, anomaly_config, feature_mode):
    lags = anomaly_config["deviation_lags_hours"]
    train_info = eligible_anomaly_timestamps(
        series, phase3a_config["partitions"][anomaly_config["training_partition"]],
        feature_mode, lags)
    val_info = eligible_anomaly_timestamps(
        series, phase3a_config["partitions"][anomaly_config["evaluation_partition"]],
        feature_mode, lags)
    train_stamps = train_info["eligible_timestamps"]
    val_stamps = val_info["eligible_timestamps"]
    extractor = fit_feature_extractor(series, train_stamps, feature_mode, anomaly_config)
    train_features = extract_features(series, train_stamps, extractor)
    val_features = extract_features(series, val_stamps, extractor)
    val_is_zero = np.asarray([series[stamp] == 0.0 for stamp in val_stamps], dtype=bool)

    rows = []
    flagged_sets = {}
    for model_name in MODEL_NAMES:
        feat_meta = detector_feature_metadata(model_name, feature_mode)
        train_scores, val_scores = fit_and_score_detector(
            model_name, train_features, val_features, extractor, anomaly_config)
        train_dist = score_distribution_summary(train_scores)
        val_dist = score_distribution_summary(val_scores)
        unique_score_count = int(len(np.unique(train_scores)))
        for op_name in OPERATING_POINTS:
            quantile = float(anomaly_config["operating_point_quantiles"][op_name])
            threshold = float(np.quantile(train_scores, quantile, method="linear"))
            train_flags = train_scores > threshold
            val_flags = val_scores > threshold
            flagged_stamps = tuple(
                stamp for stamp, is_flagged in zip(val_stamps, val_flags) if is_flagged)
            flagged_sets[(feature_mode, model_name, op_name)] = set(flagged_stamps)
            train_flagged_count = int(np.sum(train_flags))
            train_tied_at_threshold = int(np.sum(train_scores == threshold))
            val_flagged_count = int(np.sum(val_flags))
            val_flagged_zero_count = int(np.sum(val_flags & val_is_zero))
            ev = event_summary(flagged_stamps)
            rows.append({
                "building_id": building_id,
                "feature_mode": feature_mode,
                "model": model_name,
                **feat_meta,
                "operating_point": op_name,
                "training_quantile": quantile,
                "nominal_training_flagged_per_1000_hours": round(1000.0 * (1.0 - quantile), 6),
                "threshold_value": threshold,
                "train_partition_hours": train_info["partition_hours"],
                "train_observed_hours": train_info["observed_hours"],
                "train_missing_hours": train_info["missing_hours"],
                "train_valid_zero_hours": train_info["valid_zero_hours"],
                "train_insufficient_history_exclusions": train_info["insufficient_history_exclusions"],
                "train_eligible_hours": train_info["eligible_hours"],
                "validation_partition_hours": val_info["partition_hours"],
                "validation_observed_hours": val_info["observed_hours"],
                "validation_missing_hours": val_info["missing_hours"],
                "validation_valid_zero_hours": val_info["valid_zero_hours"],
                "validation_insufficient_history_exclusions": val_info["insufficient_history_exclusions"],
                "validation_eligible_hours": val_info["eligible_hours"],
                "training_unique_score_count": unique_score_count,
                "training_tied_at_threshold_hours": train_tied_at_threshold,
                "training_flagged_hours": train_flagged_count,
                "training_flagged_fraction": train_flagged_count / len(train_stamps),
                "training_flagged_per_1000_hours": 1000.0 * train_flagged_count / len(train_stamps),
                "validation_flagged_hours": val_flagged_count,
                "validation_flagged_fraction": val_flagged_count / len(val_stamps),
                "validation_flagged_per_1000_hours": 1000.0 * val_flagged_count / len(val_stamps),
                "validation_flagged_zero_hours": val_flagged_zero_count,
                "train_score_summary": train_dist,
                "validation_score_summary": val_dist,
                **ev,
            })
    return rows, flagged_sets, set(val_stamps)


def macro_summary(rows):
    summaries = []
    for mode in FEATURE_MODES:
        for model in MODEL_NAMES:
            feat_meta = detector_feature_metadata(model, mode)
            for op in OPERATING_POINTS:
                selected = [
                    r for r in rows
                    if r["feature_mode"] == mode and r["model"] == model and r["operating_point"] == op
                ]
                if not selected:
                    continue
                rates = [r["validation_flagged_per_1000_hours"] for r in selected]
                fracs = [r["validation_flagged_fraction"] for r in selected]
                events = [r["event_count"] for r in selected]
                durations = [
                    r["mean_event_duration_hours"]
                    for r in selected
                    if r["event_count"] > 0 and r["mean_event_duration_hours"] is not None
                ]
                max_durations = [
                    r["max_event_duration_hours"]
                    for r in selected
                    if r["event_count"] > 0 and r["max_event_duration_hours"] is not None
                ]
                zeros = [r["validation_flagged_zero_hours"] for r in selected]
                summaries.append({
                    "feature_mode": mode,
                    "model": model,
                    **feat_meta,
                    "operating_point": op,
                    "building_count": len(selected),
                    "buildings_with_events_count": len(durations),
                    "zero_event_buildings_count": len(selected) - len(durations),
                    "flagged_per_1000_macro_mean": float(sum(rates) / len(rates)),
                    "flagged_per_1000_median": float(np.median(rates)),
                    "flagged_per_1000_range": [float(min(rates)), float(max(rates))],
                    "flagged_fraction_macro_mean": float(sum(fracs) / len(fracs)),
                    "flagged_fraction_median": float(np.median(fracs)),
                    "flagged_fraction_range": [float(min(fracs)), float(max(fracs))],
                    "event_count_macro_mean": float(sum(events) / len(events)),
                    "event_count_median": float(np.median(events)),
                    "event_count_range": [int(min(events)), int(max(events))],
                    "mean_event_duration_macro_mean": (
                        float(sum(durations) / len(durations)) if durations else None
                    ),
                    "max_event_duration_max": int(max(max_durations)) if max_durations else None,
                    "total_flagged_zero_hours": int(sum(zeros)),
                })
    return summaries


def coincidence_summary(flagged_by_building, eligible_by_building, validation_timestamps):
    summaries = []
    for mode in FEATURE_MODES:
        for model in MODEL_NAMES:
            for op in OPERATING_POINTS:
                key = (mode, model, op)
                counts = []
                top_hours = []
                for stamp in validation_timestamps:
                    flagged_count = sum(
                        1 for b_flags in flagged_by_building.values()
                        if stamp in b_flags.get(key, set())
                    )
                    eligible_count = sum(
                        1 for b_elig in eligible_by_building.values()
                        if stamp in b_elig.get(mode, set())
                    )
                    counts.append(flagged_count)
                    if flagged_count >= 2:
                        top_hours.append((flagged_count, stamp.isoformat(sep=" "), eligible_count))
                top_hours.sort(key=lambda item: (-item[0], item[1]))
                summaries.append({
                    "feature_mode": mode,
                    "model": model,
                    "operating_point": op,
                    "interpretation_note": (
                        "Review context only; cross-building coincidence is not proof of a "
                        "weather or operational cause."
                    ),
                    "validation_partition_hours": len(validation_timestamps),
                    "hours_with_any_building_flagged": sum(1 for c in counts if c >= 1),
                    "hours_with_multi_building_coincidence": sum(1 for c in counts if c >= 2),
                    "hours_with_5plus_buildings_flagged": sum(1 for c in counts if c >= 5),
                    "max_simultaneous_buildings_flagged": int(max(counts) if counts else 0),
                    "top_coincidence_timestamps": [
                        {
                            "timestamp": stamp_str,
                            "flagged_building_count": f_cnt,
                            "eligible_building_count": e_cnt,
                        }
                        for f_cnt, stamp_str, e_cnt in top_hours[:10]
                    ],
                })
    return summaries


def _building_zero_shift_row(building_id, series, phase3a_config, anomaly_config):
    lags = anomaly_config["deviation_lags_hours"]
    tr = eligible_anomaly_timestamps(
        series, phase3a_config["partitions"][anomaly_config["training_partition"]],
        "raw_consumption", lags)
    val = eligible_anomaly_timestamps(
        series, phase3a_config["partitions"][anomaly_config["evaluation_partition"]],
        "raw_consumption", lags)
    tr_obs = tr["observed_hours"]
    val_obs = val["observed_hours"]
    tr_z = tr["valid_zero_hours"]
    val_z = val["valid_zero_hours"]
    tr_frac = tr_z / tr_obs if tr_obs > 0 else 0.0
    val_frac = val_z / val_obs if val_obs > 0 else 0.0
    tr_per_1k = 1000.0 * tr_frac
    val_per_1k = 1000.0 * val_frac
    return {
        "building_id": building_id,
        "train_observed_hours": tr_obs,
        "train_valid_zero_hours": tr_z,
        "train_valid_zero_fraction": tr_frac,
        "train_valid_zero_per_1000_hours": tr_per_1k,
        "validation_observed_hours": val_obs,
        "validation_valid_zero_hours": val_z,
        "validation_valid_zero_fraction": val_frac,
        "validation_valid_zero_per_1000_hours": val_per_1k,
        "valid_zero_fraction_delta_validation_minus_train": val_frac - tr_frac,
        "valid_zero_per_1000_delta_validation_minus_train": val_per_1k - tr_per_1k,
    }


def zero_frequency_shift_summary(core_readings, phase3a_config, anomaly_config,
                                 sensitivity_series=None):
    core_shift_rows = [
        _building_zero_shift_row(b_id, series, phase3a_config, anomaly_config)
        for b_id, series in core_readings.items()
    ]
    tr_rates = [r["train_valid_zero_per_1000_hours"] for r in core_shift_rows]
    val_rates = [r["validation_valid_zero_per_1000_hours"] for r in core_shift_rows]
    deltas = [r["valid_zero_per_1000_delta_validation_minus_train"] for r in core_shift_rows]
    summary = {
        "limitation_note": (
            "BDG2 Panther education buildings exhibit a major structural shift in valid-zero "
            "frequency between 2016 (training) and H1 2017 (validation). All detectors, "
            "reference scalers, and quantile thresholds remain strictly frozen on 2016 training "
            "data without refitting on validation; consequently, validation alert rates reflect "
            "both point deviations and this underlying 2016-to-2017 operating regime shift."
        ),
        "core_building_count": len(core_shift_rows),
        "core_train_valid_zero_per_1000_macro_mean": float(np.mean(tr_rates)),
        "core_train_valid_zero_per_1000_median": float(np.median(tr_rates)),
        "core_train_valid_zero_per_1000_range": [float(min(tr_rates)), float(max(tr_rates))],
        "core_validation_valid_zero_per_1000_macro_mean": float(np.mean(val_rates)),
        "core_validation_valid_zero_per_1000_median": float(np.median(val_rates)),
        "core_validation_valid_zero_per_1000_range": [float(min(val_rates)), float(max(val_rates))],
        "core_delta_per_1000_macro_mean": float(np.mean(deltas)),
        "core_delta_per_1000_median": float(np.median(deltas)),
        "core_buildings": core_shift_rows,
    }
    if anomaly_config["include_sensitivity_separately"] and sensitivity_series is not None:
        sens_id = anomaly_config["sensitivity_building_id"]
        summary["sensitivity_quintin"] = _building_zero_shift_row(
            sens_id, sensitivity_series, phase3a_config, anomaly_config)
    return summary


def verify_mode_mask_parity(rows):
    grouped = {}
    for row in rows:
        key = (row["building_id"], row["feature_mode"])
        mask = (
            row["train_partition_hours"], row["train_observed_hours"], row["train_missing_hours"],
            row["train_valid_zero_hours"], row["train_insufficient_history_exclusions"],
            row["train_eligible_hours"], row["validation_partition_hours"],
            row["validation_observed_hours"], row["validation_missing_hours"],
            row["validation_valid_zero_hours"], row["validation_insufficient_history_exclusions"],
            row["validation_eligible_hours"],
        )
        if key not in grouped:
            grouped[key] = mask
        elif grouped[key] != mask:
            raise ValueError(f"Anomaly eligibility mask mismatch across models for {key}")


def evaluate_anomaly_experiment(core_readings, phase3a_config, anomaly_config,
                                sensitivity_series=None):
    validate_anomaly_config(anomaly_config)
    val_timestamps = _partition_timestamps(
        phase3a_config["partitions"][anomaly_config["evaluation_partition"]])
    core_rows = []
    flagged_by_building = {b: {} for b in core_readings}
    eligible_by_building = {b: {} for b in core_readings}
    for building, series in core_readings.items():
        if "Quintin" in building:
            raise ValueError("Sensitivity building Quintin must not enter core anomaly results")
        for mode in FEATURE_MODES:
            b_rows, b_flags, b_elig = evaluate_building_mode(
                building, series, phase3a_config, anomaly_config, mode)
            core_rows.extend(b_rows)
            flagged_by_building[building].update(b_flags)
            eligible_by_building[building][mode] = b_elig
    verify_mode_mask_parity(core_rows)
    result = {
        "rows": core_rows,
        "macro_summary": macro_summary(core_rows),
        "coincidence_summary": coincidence_summary(
            flagged_by_building, eligible_by_building, val_timestamps),
        "zero_frequency_shift_summary": zero_frequency_shift_summary(
            core_readings, phase3a_config, anomaly_config, sensitivity_series),
    }
    if anomaly_config["include_sensitivity_separately"] and sensitivity_series is not None:
        sens_id = anomaly_config["sensitivity_building_id"]
        sens_rows = []
        for mode in FEATURE_MODES:
            s_rows, _, _ = evaluate_building_mode(
                sens_id, sensitivity_series, phase3a_config, anomaly_config, mode)
            sens_rows.extend(s_rows)
        verify_mode_mask_parity(sens_rows)
        result["sensitivity_quintin"] = {
            "building_id": sens_id,
            "note": "Separate sensitivity population; excluded from all 29-building core summaries.",
            "rows": sens_rows,
        }
    return result


def execution_metadata():
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
