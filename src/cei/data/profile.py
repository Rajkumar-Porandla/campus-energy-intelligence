"""Schema validation and descriptive profiling for the source dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cei.config.schema import validate_columns


def _column_type(
    field: str,
    columns: dict[str, str],
    values: pd.Series,
    timestamps: pd.Series,
    consumption: pd.Series,
) -> str:
    non_empty = values.str.strip().ne("")
    if not non_empty.any():
        return "unknown"
    if field == columns["timestamp"]:
        if timestamps[non_empty].isna().any():
            return "mixed or invalid"
        return "datetime" if values[non_empty].str.contains(r"[T ]\d{2}:").any() else "date"
    if field == columns["building"]:
        return "string identifier"
    if field == columns["consumption"]:
        numbers = consumption[non_empty]
        if numbers.isna().any() or not np.isfinite(numbers).all():
            return "mixed or invalid"
        return "integer" if numbers.mod(1).eq(0).all() else "numeric"
    if field == columns.get("usage_level"):
        return "categorical"
    return "string"


def _gap_stats(timestamps: pd.Series) -> tuple[dict[str, Any], pd.Series]:
    unique = timestamps.dropna().drop_duplicates().sort_values()
    gaps = unique.diff().dropna().dt.total_seconds().div(86400)
    counts = gaps.value_counts().sort_index()
    return (
        {
            "distinct_timestamps": int(len(unique)),
            "median_gap_days": float(gaps.median()) if not gaps.empty else None,
            "most_common_gap_days": (
                float(counts[counts.eq(counts.max())].index[0])
                if not counts.empty
                else None
            ),
            "gap_days_distribution": {
                str(float(gap)): int(count) for gap, count in counts.items()
            },
        },
        gaps,
    )


def profile_dataset(csv_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate mapped columns and summarize observed data quality."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    columns = schema["columns"]
    validate_columns(list(frame.columns), schema)

    timestamp_column = columns["timestamp"]
    building_column = columns["building"]
    consumption_column = columns["consumption"]
    timestamp_text = frame[timestamp_column].str.strip()
    building_values = frame[building_column].str.strip()
    consumption_text = frame[consumption_column].str.strip()
    timestamps = pd.to_datetime(
        timestamp_text,
        format=schema.get("timestamp_format"),
        errors="coerce",
    )
    consumption = pd.to_numeric(consumption_text, errors="coerce")
    valid_consumption = consumption.notna() & np.isfinite(consumption.fillna(0))
    missing_counts = frame.apply(
        lambda values: int(values.str.strip().eq("").sum())
    ).to_dict()

    keys = pd.DataFrame(
        {
            "building": building_values,
            "timestamp": timestamps,
            "consumption": consumption,
        }
    )
    valid_keys = keys["timestamp"].notna() & keys["building"].ne("")
    grouped = keys.loc[valid_keys].groupby(
        ["building", "timestamp"], sort=False
    )["consumption"].agg(["size", "nunique"])
    duplicate_keys = grouped[grouped["size"].gt(1)]
    duplicate_rows = int((duplicate_keys["size"] - 1).sum())

    per_building = {}
    all_gaps = []
    for building in sorted(building_values[building_values.ne("")].unique()):
        building_mask = building_values.eq(building)
        summary, gaps = _gap_stats(timestamps[building_mask])
        all_gaps.append(gaps)
        unique_dates = timestamps[building_mask].dropna().drop_duplicates()
        start = unique_dates.min() if not unique_dates.empty else None
        end = unique_dates.max() if not unique_dates.empty else None
        expected = (
            (end - start).days + 1
            if start is not None and schema.get("expected_frequency") == "daily"
            else None
        )
        per_building[building] = {
            "row_count": int(building_mask.sum()),
            **summary,
            "date_range": {
                "start": start.isoformat() if start is not None else None,
                "end": end.isoformat() if end is not None else None,
            },
            "expected_daily_observations_in_span": expected,
            "missing_daily_observations_in_span": (
                expected - summary["distinct_timestamps"] if expected else None
            ),
            "daily_coverage_ratio_in_span": (
                summary["distinct_timestamps"] / expected if expected else None
            ),
        }

    combined_gaps = (
        pd.concat(all_gaps, ignore_index=True)
        if all_gaps
        else pd.Series(dtype=float)
    )
    gap_counts = combined_gaps.value_counts().sort_index()
    valid_timestamps = timestamps.dropna()
    valid_values = consumption[valid_consumption]
    usage_column = columns.get("usage_level")
    usage_counts = {}
    if usage_column in frame.columns:
        usage_values = frame[usage_column].str.strip()
        usage_counts = (
            usage_values[usage_values.ne("")]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        )
    fields = []
    for field in frame.columns:
        values = frame[field].str.strip()
        fields.append(
            {
                "name": field,
                "inferred_type": _column_type(
                    field, columns, values, timestamps, consumption
                ),
                "non_empty_values": int(len(frame) - missing_counts[field]),
                "missing_values": missing_counts[field],
            }
        )

    return {
        "dataset": schema.get("dataset", {}),
        "source_file": csv_path.name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": fields,
        "schema_mapping": columns,
        "measurement": {
            "unit_claimed_by_column": schema.get("consumption_unit"),
            "semantics": schema.get("consumption_semantics"),
            "semantics_note": (
                "Daily interval energy is the publisher's apparent intent; the "
                "CSV cannot independently establish meter semantics or unit."
            ),
            "expected_frequency": schema.get("expected_frequency"),
        },
        "quality": {
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_building_timestamp_key_count": int(len(duplicate_keys)),
            "duplicate_building_timestamp_rows": duplicate_rows,
            "duplicate_keys_with_conflicting_consumption_count": int(
                duplicate_keys["nunique"].gt(1).sum()
            ),
            "invalid_timestamps": int(
                (timestamps.isna() & timestamp_text.ne("")).sum()
            ),
            "invalid_consumption_values": int(
                (consumption_text.ne("") & ~valid_consumption).sum()
            ),
            "negative_consumption_values": int(valid_values.lt(0).sum()),
            "zero_consumption_values": int(valid_values.eq(0).sum()),
            "missing_values_by_column": missing_counts,
        },
        "timestamps": {
            "valid_observation_count": int(valid_timestamps.size),
            "unique_timestamps": int(valid_timestamps.nunique()),
            "date_range": {
                "start": valid_timestamps.min().isoformat()
                if not valid_timestamps.empty
                else None,
                "end": valid_timestamps.max().isoformat()
                if not valid_timestamps.empty
                else None,
            },
            "sampling_frequency": {
                "expected": schema.get("expected_frequency"),
                "most_common_observed_gap_days": (
                    float(gap_counts[gap_counts.eq(gap_counts.max())].index[0])
                    if not gap_counts.empty
                    else None
                ),
                "gap_days_distribution": {
                    str(float(gap)): int(count) for gap, count in gap_counts.items()
                },
            },
        },
        "buildings": per_building,
        "building_count": len(per_building),
        "consumption": {
            "valid_value_count": int(valid_values.size),
            "minimum": float(valid_values.min()) if not valid_values.empty else None,
            "maximum": float(valid_values.max()) if not valid_values.empty else None,
            "mean": float(valid_values.mean()) if not valid_values.empty else None,
            "median": float(valid_values.median()) if not valid_values.empty else None,
            "p25": float(valid_values.quantile(0.25)) if not valid_values.empty else None,
            "p75": float(valid_values.quantile(0.75)) if not valid_values.empty else None,
            "unit_claimed_by_column": schema.get("consumption_unit"),
        },
        "categorical_value_counts": (
            {usage_column: usage_counts} if usage_column in frame.columns else {}
        ),
        "assessment": (
            "Not suitable for forecasting as-is: observations are sparse and "
            "irregular, and repeated building/date values conflict. Preserve "
            "the observations and clarify their meaning before modeling."
        ),
    }


def write_profile(profile: dict[str, Any], output_path: Path) -> None:
    """Write a machine-readable profile."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
