"""Conservative, row-preserving preprocessing for campus energy observations."""

from pathlib import Path
from typing import Any

import pandas as pd

from cei.config.schema import validate_columns


def preprocess_dataset(
    csv_path: Path, schema: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse mapped fields and flag duplicate keys without changing row count."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    validate_columns(list(frame.columns), schema)
    flags = {
        "duplicate_group_size",
        "is_duplicate_building_date",
        "has_conflicting_consumption",
    }
    collisions = flags.intersection(frame.columns)
    if collisions:
        raise ValueError(
            "Input contains reserved preprocessing columns: "
            + ", ".join(sorted(collisions))
        )

    columns = schema["columns"]
    timestamp_column = columns["timestamp"]
    building_column = columns["building"]
    consumption_column = columns["consumption"]
    timestamp_text = frame[timestamp_column].str.strip()
    building_values = frame[building_column].str.strip().replace("", pd.NA)
    consumption_text = frame[consumption_column].str.strip()
    timestamps = pd.to_datetime(
        timestamp_text,
        format=schema.get("timestamp_format"),
        errors="coerce",
    )
    consumption = pd.to_numeric(consumption_text.replace("", pd.NA), errors="coerce")

    valid_keys = timestamps.notna() & building_values.notna()
    key_rows = pd.DataFrame(
        {
            "building_id": building_values.loc[valid_keys].to_numpy(),
            "timestamp": timestamps.loc[valid_keys].to_numpy(),
            "consumption": consumption.loc[valid_keys].to_numpy(),
        }
    )
    groups = key_rows.groupby(
        ["building_id", "timestamp"], sort=False
    ).agg(
        row_count=("consumption", "size"),
        distinct_consumption=("consumption", "nunique"),
    )
    row_keys = pd.MultiIndex.from_frame(key_rows[["building_id", "timestamp"]])
    row_groups = groups.reindex(row_keys)

    frame[timestamp_column] = timestamps
    frame[building_column] = building_values
    frame[consumption_column] = consumption
    group_size = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    group_size.loc[valid_keys] = row_groups["row_count"].to_numpy()
    frame["duplicate_group_size"] = group_size
    frame["is_duplicate_building_date"] = group_size.gt(1).fillna(False)
    conflicting = pd.Series(False, index=frame.index)
    conflicting.loc[valid_keys] = row_groups["distinct_consumption"].gt(1).to_numpy()
    frame["has_conflicting_consumption"] = conflicting

    duplicate_groups = groups[groups["row_count"].gt(1)]
    report = {
        "source_file": csv_path.name,
        "dataset_version": schema.get("dataset", {}).get("version"),
        "schema_mapping": columns,
        "input_rows": int(len(frame)),
        "output_rows": int(len(frame)),
        "rows_added_or_removed": 0,
        "missing_timestamp_values": int(timestamp_text.eq("").sum()),
        "invalid_timestamp_values": int(
            (timestamp_text.ne("") & timestamps.isna()).sum()
        ),
        "missing_building_identifiers": int(building_values.isna().sum()),
        "missing_consumption_values": int(consumption_text.eq("").sum()),
        "invalid_consumption_values": int(
            (consumption_text.ne("") & consumption.isna()).sum()
        ),
        "negative_consumption_values": int(consumption.lt(0).sum()),
        "zero_consumption_values": int(consumption.eq(0).sum()),
        "duplicate_building_date_keys": int(len(duplicate_groups)),
        "rows_in_duplicate_groups_retained": int(
            duplicate_groups["row_count"].sum()
        ),
        "additional_duplicate_rows_retained": int(
            (duplicate_groups["row_count"] - 1).sum()
        ),
        "conflicting_building_date_keys": int(
            duplicate_groups["distinct_consumption"].gt(1).sum()
        ),
        "imputation_performed": False,
        "outlier_removal_performed": False,
    }
    return frame, report
