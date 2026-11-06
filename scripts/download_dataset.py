"""Download the configured dataset into ``data/raw``."""

from pathlib import Path

from cei.data.download import download_dataset

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    downloaded_path = download_dataset(ROOT / "data" / "raw")
    print(f"Downloaded dataset to {downloaded_path}")

