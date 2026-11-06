"""Download the configured public Kaggle dataset."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

DATASET_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "thanujagr4/synthetic-campus-energy-usage-dataset"
    "?datasetVersionNumber=1"
)


def download_dataset(destination: Path) -> Path:
    """Download and extract the dataset's sole CSV into ``destination``."""
    request = Request(
        DATASET_DOWNLOAD_URL,
        headers={"User-Agent": "campus-energy-intelligence/0.1"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to download the Kaggle dataset: {exc}") from exc

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_files = [
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".csv")
            ]
            if len(csv_files) != 1:
                raise ValueError(
                    "Expected exactly one CSV in the Kaggle archive; "
                    f"found {len(csv_files)}."
                )
            member = csv_files[0]
            output_path = destination / Path(member.filename).name
            if output_path.name in {"", ".", ".."}:
                raise ValueError("The archive contains an invalid CSV filename.")
            content = archive.read(member)
    except BadZipFile as exc:
        raise RuntimeError("The Kaggle download was not a valid ZIP archive.") from exc

    if not content:
        raise ValueError("The Kaggle archive contained an empty CSV.")

    destination.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path

