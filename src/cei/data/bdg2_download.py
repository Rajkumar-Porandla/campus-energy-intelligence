"""Download and verify the BDG2 files required for the Panther migration."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LFS_BATCH_URL = (
    "https://github.com/buds-lab/building-data-genome-project-2.git"
    "/info/lfs/objects/batch"
)
SOURCE_OBJECTS = {
    "electricity.csv": (
        "data/meters/raw/electricity.csv",
        "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418",
        174239039,
    ),
    "metadata.csv": (
        "data/metadata/metadata.csv",
        "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02",
        272024,
    ),
    "weather.csv": (
        "data/weather/weather.csv",
        "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd",
        19457782,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, oid: str, size: int) -> None:
    if path.stat().st_size != size or _sha256(path) != oid:
        raise ValueError(f"Existing source file failed LFS integrity checks: {path}")


def download_source_files(destination: Path) -> dict[str, Path]:
    """Fetch only the three required LFS objects without replacing local files."""
    destination.mkdir(parents=True, exist_ok=True)
    missing = []
    paths = {}
    for name, (_, oid, size) in SOURCE_OBJECTS.items():
        path = destination / name
        if path.exists():
            _verify_source(path, oid, size)
        else:
            missing.append({"oid": oid, "size": size})
        paths[name] = path

    if not missing:
        return paths

    request = Request(
        LFS_BATCH_URL,
        data=json.dumps(
            {"operation": "download", "transfers": ["basic"], "objects": missing}
        ).encode(),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            objects = json.load(response)["objects"]
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to request BDG2 source files: {exc}") from exc

    actions = {
        item["oid"]: item.get("actions", {}).get("download") for item in objects
    }
    for name, (_, oid, size) in SOURCE_OBJECTS.items():
        path = paths[name]
        if path.exists():
            continue
        action = actions.get(oid)
        if action is None:
            raise RuntimeError(f"Git LFS did not provide {name} for download.")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination, prefix=f".{name}.", suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                digest = hashlib.sha256()
                count = 0
                download_request = Request(
                    action["href"], headers=action.get("header", {})
                )
                with urlopen(download_request, timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
            if count != size or digest.hexdigest() != oid:
                raise ValueError(f"Downloaded BDG2 source failed LFS checks: {name}")
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                _verify_source(path, oid, size)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Unable to download BDG2 source {name}: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return paths
