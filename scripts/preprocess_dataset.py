"""Write a reproducible, row-preserving processed copy of the source CSV."""

import json
from pathlib import Path

from cei.config.schema import load_schema
from cei.data.preprocess import preprocess_dataset
from cei.features.calendar import add_calendar_features

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    schema = load_schema(ROOT / "src" / "cei" / "config" / "dataset_schema.json")
    source = ROOT / "data" / "raw" / "synthetic_campus_energy_usage.csv"
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed, report = preprocess_dataset(source, schema)
    processed = add_calendar_features(processed, schema["columns"]["timestamp"])
    report["calendar_features"] = ["month", "day_of_week"]
    processed.to_csv(output_dir / "energy_preprocessed.csv", index=False)
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
