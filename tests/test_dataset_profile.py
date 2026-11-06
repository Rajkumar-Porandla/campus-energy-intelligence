import csv
import json

import pytest

from cei.config.schema import load_schema
from cei.data.profile import profile_dataset


def _write_dataset(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_profile_reports_duplicate_keys_and_daily_coverage(tmp_path):
    dataset_path = tmp_path / "energy.csv"
    schema_path = tmp_path / "schema.json"
    _write_dataset(
        dataset_path,
        ["date", "building", "kwh", "label"],
        [
            {"date": "2025-01-01", "building": "Library", "kwh": "10", "label": "Low"},
            {"date": "2025-01-01", "building": "Library", "kwh": "12", "label": "High"},
            {"date": "2025-01-03", "building": "Library", "kwh": "0", "label": ""},
        ],
    )
    schema_path.write_text(
        json.dumps(
            {
                "columns": {
                    "timestamp": "date",
                    "building": "building",
                    "consumption": "kwh",
                    "usage_level": "label",
                },
                "timestamp_format": "%Y-%m-%d",
                "expected_frequency": "daily",
                "consumption_unit": "kWh",
            }
        ),
        encoding="utf-8",
    )

    profile = profile_dataset(dataset_path, load_schema(schema_path))

    assert profile["row_count"] == 3
    assert {
        column["name"]: column["inferred_type"] for column in profile["columns"]
    } == {
        "date": "date",
        "building": "string identifier",
        "kwh": "integer",
        "label": "categorical",
    }
    assert profile["quality"]["duplicate_building_timestamp_rows"] == 1
    assert profile["quality"]["duplicate_building_timestamp_key_count"] == 1
    assert profile["quality"][
        "duplicate_keys_with_conflicting_consumption_count"
    ] == 1
    assert profile["quality"]["zero_consumption_values"] == 1
    assert profile["quality"]["missing_values_by_column"]["label"] == 1
    assert profile["buildings"]["Library"]["distinct_timestamps"] == 2
    assert profile["buildings"]["Library"]["expected_daily_observations_in_span"] == 3
    assert profile["buildings"]["Library"]["missing_daily_observations_in_span"] == 1
    assert profile["buildings"]["Library"]["daily_coverage_ratio_in_span"] == pytest.approx(
        2 / 3
    )
    assert profile["timestamps"]["sampling_frequency"]["gap_days_distribution"] == {
        "2.0": 1
    }


def test_profile_rejects_missing_mapped_columns(tmp_path):
    dataset_path = tmp_path / "energy.csv"
    _write_dataset(dataset_path, ["date", "building"], [])
    schema = {
        "columns": {
            "timestamp": "date",
            "building": "building",
            "consumption": "kwh",
        }
    }

    with pytest.raises(ValueError, match="kwh"):
        profile_dataset(dataset_path, schema)
