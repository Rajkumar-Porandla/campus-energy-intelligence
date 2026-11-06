"""Stream, validate, and write the canonical BDG2 Panther datasets."""

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

CANONICAL_COLUMNS = ["timestamp", "building_id", "site_id", "energy_consumption"]
WEATHER_ID_COLUMNS = ["timestamp", "site_id"]


def _missing(value: str) -> bool:
    return value.strip().lower() in {"", "na", "nan", "null"}


def _number(value: str, label: str) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {label}: {value!r}")
    return number


def _temporary_csv(path: Path, columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    writer = csv.writer(output)
    writer.writerow(columns)
    return Path(output.name), output, writer


def _publish(temp: Path, destination: Path) -> None:
    if destination.exists():
        if destination.read_bytes() != temp.read_bytes():
            raise FileExistsError(f"Refusing to overwrite migration output: {destination}")
        temp.unlink()
        return
    os.link(temp, destination)
    temp.unlink()


def validate_buildings(
    metadata_path: Path, config: dict
) -> dict[str, dict[str, str]]:
    """Confirm selected identifiers map to Panther education electricity rows."""
    selected = config["core_buildings"] + [config["sensitivity_building"]]
    if len(config["core_buildings"]) != 29 or len(set(selected)) != 30:
        raise ValueError("Panther configuration must contain 29 core and one unique sensitivity building.")
    mapping = {}
    with metadata_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"building_id", "site_id", "primaryspaceusage", "electricity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("BDG2 metadata lacks required building mapping columns.")
        wanted = set(selected)
        for row in reader:
            if row["building_id"] in wanted:
                mapping[row["building_id"]] = row
    if set(mapping) != set(selected):
        raise ValueError(f"Metadata building mismatch: missing {sorted(set(selected) - set(mapping))}")
    invalid = [
        building for building, row in mapping.items()
        if row["site_id"] != config["site_id"]
        or row["primaryspaceusage"] != "Education"
        or row["electricity"].strip().lower() not in {"yes", "true", "1"}
    ]
    if invalid:
        raise ValueError(f"Selected buildings fail site/type/electricity metadata checks: {invalid}")
    return mapping


def migrate_electricity(
    source_path: Path, core_path: Path, sensitivity_path: Path, config: dict
) -> tuple[dict, list[str]]:
    """Write timestamp-major canonical electricity CSVs from the wide source."""
    core, sensitivity = config["core_buildings"], config["sensitivity_building"]
    selected = core + [sensitivity]
    expected_start = datetime.fromisoformat(config["date_start"])
    expected_end = datetime.fromisoformat(config["date_end"])
    expected_count = int((expected_end - expected_start) / timedelta(hours=1)) + 1
    counts = {building: Counter() for building in selected}
    temps, streams, writers = [], [], []
    timestamps = []
    previous = None
    try:
        for path in (core_path, sensitivity_path):
            temp, stream, writer = _temporary_csv(path, CANONICAL_COLUMNS)
            temps.append((temp, path))
            streams.append(stream)
            writers.append(writer)
        core_writer, sensitivity_writer = writers
        with source_path.open(newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            header = next(reader, [])
            positions = {building: header.index(building) for building in selected if building in header}
            if set(positions) != set(selected):
                raise ValueError(f"Electricity meter columns missing: {sorted(set(selected) - set(positions))}")
            if not header or header[0] != "timestamp":
                raise ValueError("BDG2 electricity timestamp must be the first column.")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Malformed electricity row at record {len(timestamps) + 2}.")
                try:
                    current = datetime.fromisoformat(row[0])
                except ValueError as exc:
                    raise ValueError(f"Invalid electricity timestamp: {row[0]!r}") from exc
                if previous is not None and current - previous != timedelta(hours=1):
                    raise ValueError(f"Duplicate or non-hourly electricity timestamp: {row[0]}")
                if current != expected_start + timedelta(hours=len(timestamps)):
                    raise ValueError(f"Electricity timestamp outside expected sequence: {row[0]}")
                previous = current
                timestamps.append(row[0])
                for building in selected:
                    number = _number(row[positions[building]], building)
                    counts[building]["missing"] += number is None
                    counts[building]["zero"] += number == 0
                    counts[building]["invalid"] += 0
                    output = [
                        row[0], building, config["site_id"],
                        "" if number is None else row[positions[building]].strip(),
                    ]
                    (sensitivity_writer if building == sensitivity else core_writer).writerow(output)
        if len(timestamps) != expected_count or timestamps[-1] != expected_end.isoformat(sep=" "):
            raise ValueError(f"Expected {expected_count} hourly readings per building; found {len(timestamps)}.")
        for stream in streams:
            stream.close()
        for temp, path in temps:
            _publish(temp, path)
    finally:
        for stream in streams:
            if not stream.closed:
                stream.close()
        for temp, _ in temps:
            temp.unlink(missing_ok=True)

    def building_report(building):
        missing = counts[building]["missing"]
        return {
            "observations": len(timestamps),
            "missing_count": missing,
            "missing_pct": 100 * missing / len(timestamps),
            "zero_count": counts[building]["zero"],
            "invalid_numeric_count": counts[building]["invalid"],
        }

    summary = {
        "row_count_core": len(core) * len(timestamps),
        "row_count_sensitivity": len(timestamps),
        "timestamp_count_per_building": len(timestamps),
        "duplicate_timestamp_count": 0,
        "date_start": timestamps[0],
        "date_end": timestamps[-1],
        "core_missing_count": sum(counts[b]["missing"] for b in core),
        "core_zero_count": sum(counts[b]["zero"] for b in core),
        "all_selected_missing_count": sum(counts[b]["missing"] for b in selected),
        "all_selected_zero_count": sum(counts[b]["zero"] for b in selected),
        "per_building": {b: building_report(b) for b in selected},
    }
    return summary, timestamps


def migrate_weather(
    source_path: Path, output_path: Path, config: dict, electricity_timestamps: list[str]
) -> dict:
    """Keep Panther weather separate and verify its hourly key against meters."""
    features = config["weather_features"]
    columns = WEATHER_ID_COLUMNS + features
    temp, stream, writer = _temporary_csv(output_path, columns)
    counts = Counter()
    timestamps = []
    try:
        with source_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not set(columns).issubset(reader.fieldnames or []):
                raise ValueError("BDG2 weather file lacks required Panther feature columns.")
            for row in reader:
                if row["site_id"] != config["site_id"]:
                    continue
                timestamp = row["timestamp"]
                if timestamps and datetime.fromisoformat(timestamp) - datetime.fromisoformat(timestamps[-1]) != timedelta(hours=1):
                    raise ValueError(f"Duplicate or non-hourly Panther weather timestamp: {timestamp}")
                timestamps.append(timestamp)
                values = [timestamp, config["site_id"]]
                for feature in features:
                    number = _number(row[feature], feature)
                    counts[feature + "_missing"] += number is None
                    values.append("" if number is None else row[feature].strip())
                writer.writerow(values)
        if timestamps != electricity_timestamps:
            raise ValueError("Panther weather timestamps do not exactly match electricity timestamps.")
        stream.close()
        _publish(temp, output_path)
    finally:
        if not stream.closed:
            stream.close()
        temp.unlink(missing_ok=True)
    return {
        "row_count": len(timestamps),
        "duplicate_timestamp_count": 0,
        "aligned_with_electricity": True,
        "missing_by_feature": {
            feature: {
                "count": counts[feature + "_missing"],
                "pct": 100 * counts[feature + "_missing"] / len(timestamps),
            }
            for feature in features
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, content: dict) -> None:
    """Write deterministic JSON, accepting an identical prior migration only."""
    encoded = (json.dumps(content, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to overwrite migration record: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    ) as output:
        output.write(encoded)
        output.flush()
        os.link(output.name, path)
