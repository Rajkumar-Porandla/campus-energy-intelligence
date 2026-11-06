"""Create a JSON quality and suitability profile of the configured dataset."""

from pathlib import Path

from cei.config.schema import load_schema
from cei.data.profile import profile_dataset, write_profile

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    schema_path = ROOT / "src" / "cei" / "config" / "dataset_schema.json"
    dataset_path = ROOT / "data" / "raw" / "synthetic_campus_energy_usage.csv"
    profile_path = ROOT / "data" / "processed" / "dataset_profile.json"

    schema = load_schema(schema_path)
    profile = profile_dataset(dataset_path, schema)
    write_profile(profile, profile_path)
    print(f"Dataset profile written to {profile_path}")
    print(profile["assessment"])
