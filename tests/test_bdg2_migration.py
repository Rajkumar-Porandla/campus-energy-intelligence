import csv
import json
from pathlib import Path

import pytest

from cei.data.bdg2_migration import (
    CANONICAL_COLUMNS,
    migrate_electricity,
    migrate_weather,
    validate_buildings,
)

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = json.loads(
        (ROOT / "src/cei/config/bdg2_panther.json").read_text(encoding="utf-8")
    )
    config["date_start"] = "2016-01-01 00:00:00"
    config["date_end"] = "2016-01-01 01:00:00"
    return config


def _write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)


def _source_fixture(tmp_path, invalid_value=None, duplicate=False):
    config = _config()
    buildings = config["core_buildings"] + [config["sensitivity_building"]]
    source = tmp_path / "electricity.csv"
    rows = []
    for timestamp in (
        ["2016-01-01 00:00:00", "2016-01-01 00:00:00"]
        if duplicate
        else ["2016-01-01 00:00:00", "2016-01-01 01:00:00"]
    ):
        values = ["1"] * len(buildings)
        values[buildings.index("Panther_education_Alecia")] = (
            "bad" if invalid_value and not rows else "0" if not rows else "1"
        )
        values[buildings.index("Panther_education_Annetta")] = "" if rows else "2"
        rows.append([timestamp, *values])
    _write_csv(source, ["timestamp", *buildings], rows)
    return config, source


def test_metadata_selection_and_canonical_migration_preserve_zero_and_missing(tmp_path):
    config, electricity = _source_fixture(tmp_path)
    metadata = tmp_path / "metadata.csv"
    buildings = config["core_buildings"] + [config["sensitivity_building"]]
    assert len(config["core_buildings"]) == 29
    assert config["sensitivity_building"] == "Panther_education_Quintin"
    assert all(building.startswith("Panther_education_") for building in buildings)
    assert config["sensitivity_building"] not in config["core_buildings"]
    _write_csv(
        metadata,
        ["building_id", "site_id", "primaryspaceusage", "electricity"],
        [[building, "Panther", "Education", "Yes"] for building in buildings],
    )
    assert set(validate_buildings(metadata, config)) == set(buildings)

    core_path = tmp_path / "core.csv"
    sensitivity_path = tmp_path / "sensitivity.csv"
    report, timestamps = migrate_electricity(
        electricity, core_path, sensitivity_path, config
    )
    assert timestamps == ["2016-01-01 00:00:00", "2016-01-01 01:00:00"]
    assert report["row_count_core"] == 58
    assert report["row_count_sensitivity"] == 2
    assert report["per_building"]["Panther_education_Alecia"]["zero_count"] == 1
    assert report["per_building"]["Panther_education_Annetta"]["missing_count"] == 1
    with core_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == CANONICAL_COLUMNS
        rows = list(reader)
    assert len(rows) == 58
    alecia_zero = next(
        row for row in rows
        if row["building_id"] == "Panther_education_Alecia"
        and row["timestamp"] == timestamps[0]
    )
    annetta_missing = next(
        row for row in rows
        if row["building_id"] == "Panther_education_Annetta"
        and row["timestamp"] == timestamps[1]
    )
    assert alecia_zero["energy_consumption"] == "0"
    assert annetta_missing["energy_consumption"] == ""
    with sensitivity_path.open(newline="", encoding="utf-8") as source:
        sensitivity_rows = list(csv.DictReader(source))
    assert {row["building_id"] for row in sensitivity_rows} == {
        config["sensitivity_building"]
    }

    # Repeated runs reproduce the same bytes and do not replace existing output.
    before = core_path.read_bytes()
    migrate_electricity(electricity, core_path, sensitivity_path, config)
    assert core_path.read_bytes() == before


def test_weather_is_separate_and_requires_exact_timestamp_alignment(tmp_path):
    config, electricity = _source_fixture(tmp_path)
    core_path, sensitivity_path = tmp_path / "core.csv", tmp_path / "sensitivity.csv"
    _, timestamps = migrate_electricity(
        electricity, core_path, sensitivity_path, config
    )
    weather_path = tmp_path / "weather.csv"
    features = config["weather_features"]
    _write_csv(
        weather_path,
        ["timestamp", "site_id", *features],
        [
            [timestamp, "Panther", "20", "", "15", "0", "", "1000", "90", "3"]
            for timestamp in timestamps
        ],
    )
    weather_output = tmp_path / "weather_panther.csv"
    report = migrate_weather(weather_path, weather_output, config, timestamps)
    assert report["aligned_with_electricity"] is True
    assert report["missing_by_feature"]["cloudCoverage"]["count"] == 2
    with weather_output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[0]["cloudCoverage"] == ""
    assert rows[0]["site_id"] == "Panther"

    with pytest.raises(ValueError, match="do not exactly match"):
        migrate_weather(weather_path, tmp_path / "misaligned.csv", config, timestamps[:1])


def test_invalid_values_and_duplicate_timestamps_fail_without_outputs(tmp_path):
    config, source = _source_fixture(tmp_path, invalid_value=True)
    with pytest.raises(ValueError, match="Invalid numeric value"):
        migrate_electricity(
            source, tmp_path / "core-invalid.csv", tmp_path / "sensitivity-invalid.csv", config
        )
    assert not (tmp_path / "core-invalid.csv").exists()

    config, source = _source_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate or non-hourly"):
        migrate_electricity(
            source, tmp_path / "core-duplicate.csv", tmp_path / "sensitivity-duplicate.csv", config
        )
    assert not (tmp_path / "core-duplicate.csv").exists()


def test_building_metadata_rejects_wrong_site_or_usage(tmp_path):
    config = _config()
    buildings = config["core_buildings"] + [config["sensitivity_building"]]
    metadata = tmp_path / "metadata.csv"
    rows = [[building, "Panther", "Education", "Yes"] for building in buildings]
    rows[0][1] = "Other"
    _write_csv(
        metadata,
        ["building_id", "site_id", "primaryspaceusage", "electricity"],
        rows,
    )
    with pytest.raises(ValueError, match="fail site/type/electricity"):
        validate_buildings(metadata, config)
