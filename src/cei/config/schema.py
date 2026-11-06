"""Load and validate the configured dataset field mapping."""

import json
from pathlib import Path
from typing import Any


def load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        schema = json.load(file)
    columns = schema.get("columns")
    required = ("timestamp", "building", "consumption")
    if not isinstance(columns, dict):
        raise ValueError("Dataset schema must contain a 'columns' object.")
    missing = [role for role in required if not columns.get(role)]
    if missing:
        raise ValueError("Dataset schema is missing mappings: " + ", ".join(missing))
    return schema


def validate_columns(fieldnames: list[str], schema: dict[str, Any]) -> None:
    columns = schema["columns"]
    required = [columns[role] for role in ("timestamp", "building", "consumption")]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(
            "Dataset is missing required mapped columns: " + ", ".join(missing)
        )
