"""Download, validate, and migrate the approved BDG2 Panther subset."""

import json
from datetime import datetime, timezone
from pathlib import Path

from cei.data.bdg2_download import SOURCE_OBJECTS, download_source_files
from cei.data.bdg2_migration import (
    migrate_electricity,
    migrate_weather,
    sha256_file,
    validate_buildings,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "cei" / "config" / "bdg2_panther.json"
RAW_DIR = ROOT / "data" / "raw" / "bdg2" / "panther"
PROCESSED_DIR = ROOT / "data" / "processed" / "bdg2" / "panther"
REPORT_DIR = ROOT / "data" / "reports" / "bdg2" / "panther"
CORE_PATH = PROCESSED_DIR / "electricity_core.csv"
SENSITIVITY_PATH = PROCESSED_DIR / "electricity_sensitivity_quintin.csv"
WEATHER_PATH = PROCESSED_DIR / "weather.csv"
MANIFEST_PATH = REPORT_DIR / "dataset_manifest.json"
REPORT_PATH = REPORT_DIR / "migration_report.json"


def _phase_25_comparison(electricity: dict, weather: dict) -> dict:
    expected_weather_missing = {
        "airTemperature": 3,
        "cloudCoverage": 7542,
        "dewTemperature": 3,
        "precipDepth1HR": 9,
        "precipDepth6HR": 16898,
        "seaLvlPressure": 323,
        "windDirection": 499,
        "windSpeed": 0,
    }
    checks = {
        "core_missing_count": (1145, electricity["core_missing_count"]),
        "all_30_missing_count": (6130, electricity["all_selected_missing_count"]),
        "all_30_zero_count": (98000, electricity["all_selected_zero_count"]),
    }
    for feature, expected in expected_weather_missing.items():
        checks[f"weather_missing_{feature}"] = (
            expected,
            weather["missing_by_feature"][feature]["count"],
        )
    return {
        name: {
            "phase_2_5_audit": expected,
            "migration": actual,
            "matches": expected == actual,
        }
        for name, (expected, actual) in checks.items()
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = download_source_files(RAW_DIR)
    metadata = validate_buildings(sources["metadata.csv"], config)
    electricity, timestamps = migrate_electricity(
        sources["electricity.csv"], CORE_PATH, SENSITIVITY_PATH, config
    )
    weather = migrate_weather(sources["weather.csv"], WEATHER_PATH, config, timestamps)
    source_files = {}
    for name, (_, oid, size) in SOURCE_OBJECTS.items():
        path = sources[name]
        source_files[name] = {
            "original_path": SOURCE_OBJECTS[name][0],
            "source_url": (
                f"{config['source_repository']}/blob/master/"
                f"{SOURCE_OBJECTS[name][0]}"
            ),
            "local_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": size,
            "sha256": sha256_file(path),
            "expected_lfs_sha256": oid,
        }
    output_files = {
        path.name: {
            "local_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (CORE_PATH, SENSITIVITY_PATH, WEATHER_PATH)
    }
    retrieval_date = datetime.now(timezone.utc).date().isoformat()
    if MANIFEST_PATH.exists():
        existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        retrieval_date = existing_manifest["retrieval_date"]
    license_info = (
        "CC BY-SA 4.0 for GitHub-sourced files, applied conservatively pending "
        "author clarification. The BDG2 Zenodo v1.0 record declares CC BY 4.0; "
        "no file-specific license resolution was found."
    )
    manifest = {
        "dataset_name": config["dataset_name"],
        "original_source_url": config["source_repository"],
        "site": {
            "site_id": config["site_id"],
            "site_name": config["site_name"],
        },
        "retrieval_date": retrieval_date,
        "source_files": source_files,
        "selected_building_identifiers": {
            "core": config["core_buildings"],
            "sensitivity": [config["sensitivity_building"]],
        },
        "building_metadata_mapping": {
            building: {
                "site_id": row["site_id"],
                "primaryspaceusage": row["primaryspaceusage"],
                "electricity": row["electricity"],
                "source_meter_column": building,
            }
            for building, row in metadata.items()
        },
        "sampling_frequency": config["sampling_frequency"],
        "date_range": {
            "start": config["date_start"],
            "end": config["date_end"],
            "expected_timestamps_per_building": 17544,
        },
        "measurement": {
            "source_field": "meter_reading",
            "unit": config["meter_reading_unit"],
            "unit_documentation": (
                "The BDG2 repository meter-reading notebook documents "
                "meter_reading in kilowatt hours (kWh)."
            ),
            "source_documentation_url": (
                f"{config['source_repository']}/blob/master/"
                "notebooks/03_EDA-meter-reading.ipynb"
            ),
            "zero_readings_preserved": True,
            "missing_readings": "Empty canonical CSV field; no imputation.",
        },
        "weather": {
            "source_file": source_files["weather.csv"]["local_path"],
            "features": config["weather_features"],
            "storage": output_files[WEATHER_PATH.name]["local_path"],
        },
        "license": license_info,
        "attribution": (
            "Miller, C. et al. (2020). The Building Data Genome Project 2, "
            "energy meter data from the ASHRAE Great Energy Predictor III "
            "competition. Scientific Data 7, 368. "
            "https://doi.org/10.1038/s41597-020-00712-x"
        ),
        "output_files": output_files,
        "transformation_history": [
            "Downloaded only raw electricity, building metadata, and weather LFS objects.",
            "Verified each downloaded source against its published Git LFS SHA-256 and size.",
            "Validated the 29 core and one sensitivity building against site, use, and electricity metadata.",
            "Selected the original wide-file meter columns and wrote timestamp-major long CSVs.",
            "Preserved raw numeric readings, including zero; represented missing readings as empty fields.",
            "Stored UCF Panther weather separately after exact timestamp-alignment validation.",
            "Did not aggregate, interpolate, impute, remove outliers, or train models.",
        ],
    }
    comparison = _phase_25_comparison(electricity, weather)
    report = {
        "dataset_name": config["dataset_name"],
        "site_id": config["site_id"],
        "date_range": {
            "start": electricity["date_start"],
            "end": electricity["date_end"],
        },
        "sampling_frequency": config["sampling_frequency"],
        "source_files": source_files,
        "source_size_total_bytes": sum(item["size_bytes"] for item in source_files.values()),
        "electricity": electricity,
        "weather": weather,
        "validation": {
            "core_building_count": len(config["core_buildings"]),
            "sensitivity_building": config["sensitivity_building"],
            "building_metadata_mapping_valid": True,
            "core_output_schema": [
                "timestamp", "building_id", "site_id", "energy_consumption"
            ],
            "core_and_sensitivity_separate": CORE_PATH != SENSITIVITY_PATH,
            "weather_separate": True,
            "phase_2_5_comparison": comparison,
            "phase_2_5_all_checks_match": all(
                item["matches"] for item in comparison.values()
            ),
        },
        "output_files": output_files,
        "license": license_info,
        "attribution": manifest["attribution"],
    }
    write_json_once(MANIFEST_PATH, manifest)
    write_json_once(REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
