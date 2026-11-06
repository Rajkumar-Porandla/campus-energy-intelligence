"""Read-only Phase 3A canonical validation and forecast coverage diagnostics."""
import csv
import json
import math
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

TASKS = ("hourly", "daily_profile", "weekly_profile")
ELECTRICITY_COLUMNS = {"timestamp", "building_id", "site_id", "energy_consumption"}
WEATHER_COLUMNS = {"timestamp", "site_id"}
PRIMARY_WEATHER_COLUMNS = {
    "airTemperature", "dewTemperature", "windSpeed", "precipDepth1HR", "seaLvlPressure"
}
BASELINE_PERIOD_KEYS = {
    "persistence": "persistence_period_hours",
    "daily_seasonal_naive": "daily_seasonal_period_hours",
    "weekly_seasonal_naive": "weekly_seasonal_period_hours",
}


def _timestamp(value):
    return datetime.fromisoformat(value)


def load_experiment_config(path):
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    return validate_experiment_config(config)


def validate_experiment_config(config):
    """Validate required Phase 3A settings; return the unchanged JSON mapping."""
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    required = {"site_id", "core_building_ids", "partitions", "horizons_hours",
                "candidate_lookbacks_hours", "baseline_settings", "baseline_comparisons",
                "required_primary_weather_columns"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Experiment configuration is missing: {sorted(missing)}")
    buildings = config["core_building_ids"]
    if not isinstance(buildings, list) or not buildings or len(buildings) != len(set(buildings)):
        raise ValueError("core_building_ids must be a nonempty list of unique IDs")
    partitions = config["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise ValueError("partitions must define train, validation, and test")
    if any(not isinstance(partitions[name], list) or len(partitions[name]) != 2
           for name in ("train", "validation", "test")):
        raise ValueError("Each partition must contain exactly two inclusive timestamps")
    bounds = [tuple(_timestamp(x) for x in partitions[name])
              for name in ("train", "validation", "test")]
    if any(start > end for start, end in bounds):
        raise ValueError("Each partition must have an ordered inclusive start/end")
    if any(bounds[i][1] + timedelta(hours=1) != bounds[i + 1][0] for i in range(2)):
        raise ValueError("Partitions must be consecutive and non-overlapping")
    if not isinstance(config["horizons_hours"], dict) or set(config["horizons_hours"]) != set(TASKS) or any(
            not isinstance(v, int) or v < 1 for v in config["horizons_hours"].values()):
        raise ValueError("horizons_hours must provide positive integer hours for every task")
    lookbacks = config["candidate_lookbacks_hours"]
    if not isinstance(lookbacks, list) or not lookbacks or any(not isinstance(v, int) or v < 1 for v in lookbacks) \
            or len(lookbacks) != len(set(lookbacks)):
        raise ValueError("candidate_lookbacks_hours must contain unique positive integers")
    if not isinstance(config["baseline_comparisons"], dict) or set(config["baseline_comparisons"]) != set(TASKS):
        raise ValueError("baseline_comparisons must provide one method for every task")
    settings = config["baseline_settings"]
    if not isinstance(settings, dict) or not set(BASELINE_PERIOD_KEYS.values()).issubset(settings) or any(
            not isinstance(settings[key], int) or settings[key] < 1
            for key in BASELINE_PERIOD_KEYS.values()):
        raise ValueError("baseline_settings must define positive integer periods for all baselines")
    methods = set(BASELINE_PERIOD_KEYS)
    if any(method not in methods for method in config["baseline_comparisons"].values()):
        raise ValueError("baseline_comparisons contains an unknown method")
    weather = config["required_primary_weather_columns"]
    if not isinstance(weather, list) or not PRIMARY_WEATHER_COLUMNS.issubset(weather):
        raise ValueError("required_primary_weather_columns omits an approved primary feature")
    return config


@contextmanager
def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        yield csv.DictReader(source)


def validate_canonical(electricity_path, weather_path, config):
    """Validate canonical keys, configured date range, weather schema, and alignment."""
    validate_experiment_config(config)
    readings = {building: {} for building in config["core_building_ids"]}
    with _read_csv(electricity_path) as reader:
        if not ELECTRICITY_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Canonical electricity columns are incomplete")
        previous = None
        for row in reader:
            stamp = _timestamp(row["timestamp"])
            if previous is not None and stamp < previous:
                raise ValueError("Canonical electricity timestamps are not ordered")
            previous = stamp
            building = row["building_id"]
            if building not in readings or row["site_id"] != config["site_id"]:
                raise ValueError(f"Unexpected canonical electricity key: {building}")
            if stamp in readings[building]:
                raise ValueError(f"Duplicate electricity key: {building} {stamp}")
            raw = row["energy_consumption"].strip()
            value = None if raw == "" else float(raw)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Non-finite electricity reading at {building} {stamp}")
            readings[building][stamp] = value
    if any(not series for series in readings.values()):
        raise ValueError("Canonical electricity is missing configured buildings")
    expected = sorted(next(iter(readings.values())))
    start = _timestamp(config["partitions"]["train"][0])
    end = _timestamp(config["partitions"]["test"][1])
    if expected[0] != start or expected[-1] != end:
        raise ValueError("Canonical electricity date range differs from configured period")
    if any(sorted(series) != expected for series in readings.values()):
        raise ValueError("Electricity timestamps differ across core buildings")
    if any(b - a != timedelta(hours=1) for a, b in zip(expected, expected[1:])):
        raise ValueError("Electricity index is not hourly and continuous")
    with _read_csv(weather_path) as weather:
        required = WEATHER_COLUMNS | set(config["required_primary_weather_columns"])
        if not required.issubset(weather.fieldnames or []):
            raise ValueError(f"Canonical weather columns are incomplete: {sorted(required - set(weather.fieldnames or []))}")
        weather_stamps = []
        for row in weather:
            if row["site_id"] != config["site_id"]:
                raise ValueError("Unexpected canonical weather site")
            weather_stamps.append(_timestamp(row["timestamp"]))
            for name, raw in row.items():
                if name in WEATHER_COLUMNS or raw.strip() == "":
                    continue
                try:
                    number = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid weather value for {name}") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite weather value for {name}")
    if weather_stamps != expected:
        raise ValueError("Canonical weather index does not exactly match electricity")
    return readings, expected


def forecast_blocks(partition, task, config):
    """Yield origin and target indices for full non-overlapping partition blocks."""
    validate_experiment_config(config)
    if partition not in config["partitions"] or task not in config["horizons_hours"]:
        raise ValueError("Unknown partition or forecast task")
    start, end = (_timestamp(x) for x in config["partitions"][partition])
    horizon = config["horizons_hours"][task]
    cursor = start
    while cursor + timedelta(hours=horizon - 1) <= end:
        yield cursor - timedelta(hours=1), tuple(
            cursor + timedelta(hours=step) for step in range(horizon)
        )
        cursor += timedelta(hours=horizon)


def eligible_origin(series, origin, targets, lookback):
    """Return eligibility and missing-history/target flags; zero is observed."""
    history = [origin - timedelta(hours=step) for step in range(lookback)]
    history_missing = any(stamp not in series or series[stamp] is None for stamp in history)
    target_missing = any(stamp not in series or series[stamp] is None for stamp in targets)
    return not history_missing and not target_missing, history_missing, target_missing


def _task_origins(partition, task, config):
    horizon = config["horizons_hours"][task]
    if horizon != 1:
        return list(forecast_blocks(partition, task, config))
    start, end = (_timestamp(x) for x in config["partitions"][partition])
    return [(stamp - timedelta(hours=1), (stamp,)) for stamp in
            (start + timedelta(hours=i) for i in range(int((end-start).total_seconds()/3600)+1))]


def _baseline_period(method, config):
    if method not in BASELINE_PERIOD_KEYS:
        raise ValueError(f"Unknown baseline: {method}")
    return config["baseline_settings"][BASELINE_PERIOD_KEYS[method]]


def coverage_report(readings, config):
    """Report per-building coverage; missing targets take precedence over history."""
    validate_experiment_config(config)
    rows = []
    for building, series in readings.items():
        for partition in config["partitions"]:
            for task in TASKS:
                origins = _task_origins(partition, task, config)
                method = config["baseline_comparisons"][task]
                method_history = _baseline_period(method, config)
                candidates = config["candidate_lookbacks_hours"]
                shortest = min(candidates)
                shortest_history = max(shortest, method_history)
                baseline_eligible = sum(eligible_origin(series, origin, targets, shortest_history)[0]
                                        for origin, targets in origins)
                counts = []
                for lookback in candidates:
                    common_history = max(lookback, method_history)
                    eligible = target_excluded = history_excluded = 0
                    for origin, targets in origins:
                        ok, no_history, no_target = eligible_origin(
                            series, origin, targets, common_history)
                        if ok:
                            eligible += 1
                        elif no_target:
                            target_excluded += 1
                        elif no_history:
                            history_excluded += 1
                    counts.append((lookback, common_history, eligible,
                                   target_excluded, history_excluded))
                for lookback, common_history, eligible, target_excluded, history_excluded in counts:
                    reduction = baseline_eligible - eligible
                    rows.append({"building_id": building, "partition": partition,
                        "task": task, "horizon_hours": config["horizons_hours"][task],
                        "lookback_hours": lookback,
                        "registered_common_lookback_hours": common_history,
                        "shortest_method_valid_lookback_hours": shortest_history,
                        "shortest_lookback_eligible_origins": baseline_eligible,
                        "coverage_reduction_origins": reduction,
                        "coverage_reduction_pct": (100 * reduction / baseline_eligible
                                                    if baseline_eligible else None),
                        "candidate_origins": len(origins), "eligible_origins": eligible,
                        "eligible_target_hours": eligible * config["horizons_hours"][task],
                        "missing_target_exclusions": target_excluded,
                        "insufficient_history_exclusions": history_excluded})
    return rows


def baseline_forecast(series, origin, targets, method, config):
    """Return baseline values only when all required inputs are observed by origin."""
    period = _baseline_period(method, validate_experiment_config(config))
    inputs = ([origin] * len(targets) if method == "persistence" else
              [target - timedelta(hours=period) for target in targets])
    if any(stamp > origin or stamp not in series or series[stamp] is None for stamp in inputs):
        return None
    return tuple(series[stamp] for stamp in inputs)
