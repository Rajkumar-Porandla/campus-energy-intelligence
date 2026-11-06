import pandas as pd

from cei.data.preprocess import preprocess_dataset


def test_preprocessing_preserves_conflicts_gaps_and_zero_values(tmp_path):
    source = tmp_path / "energy.csv"
    source.write_text(
        "date,building,kwh\n"
        "2025-01-01, Library ,10\n"
        "2025-01-01,Library,12\n"
        "2025-01-03,Library,0\n"
        "2025-01-04,Library,\n"
        "2025-01-05,Library,-5\n",
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()
    schema = {
        "columns": {
            "timestamp": "date",
            "building": "building",
            "consumption": "kwh",
        },
        "timestamp_format": "%Y-%m-%d",
    }

    processed, report = preprocess_dataset(source, schema)

    assert source.read_bytes() == original_bytes
    assert len(processed) == report["input_rows"] == report["output_rows"] == 5
    assert report["rows_added_or_removed"] == 0
    assert report["duplicate_building_date_keys"] == 1
    assert report["additional_duplicate_rows_retained"] == 1
    assert report["conflicting_building_date_keys"] == 1
    assert processed["is_duplicate_building_date"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert processed["has_conflicting_consumption"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert processed["building"].tolist() == ["Library"] * 5
    assert processed["kwh"].iloc[2] == 0
    assert pd.isna(processed["kwh"].iloc[3])
    assert report["zero_consumption_values"] == 1
    assert report["negative_consumption_values"] == 1
    assert report["missing_consumption_values"] == 1
    assert processed["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-01",
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
    ]


def test_preprocessing_reports_invalid_and_missing_values_without_dropping_rows(
    tmp_path,
):
    source = tmp_path / "energy.csv"
    source.write_text(
        "date,building,kwh\n"
        "not-a-date,Library,not-a-number\n"
        ",, \n",
        encoding="utf-8",
    )
    schema = {
        "columns": {
            "timestamp": "date",
            "building": "building",
            "consumption": "kwh",
        },
        "timestamp_format": "%Y-%m-%d",
    }

    processed, report = preprocess_dataset(source, schema)

    assert len(processed) == 2
    assert report["invalid_timestamp_values"] == 1
    assert report["missing_timestamp_values"] == 1
    assert report["missing_building_identifiers"] == 1
    assert report["invalid_consumption_values"] == 1
    assert report["missing_consumption_values"] == 1
    assert processed["date"].isna().all()
    assert processed["kwh"].isna().all()
