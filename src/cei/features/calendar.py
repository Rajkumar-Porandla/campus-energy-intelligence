"""Calendar features supported by the available date-only observations."""

import pandas as pd


def add_calendar_features(
    observations: pd.DataFrame, timestamp_column: str
) -> pd.DataFrame:
    if timestamp_column not in observations.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if not pd.api.types.is_datetime64_any_dtype(observations[timestamp_column]):
        raise TypeError(f"{timestamp_column} must be parsed as datetime first.")
    feature_names = {"month", "day_of_week"}
    collisions = feature_names.intersection(observations.columns)
    if collisions:
        raise ValueError("Input already contains features: " + ", ".join(sorted(collisions)))

    features = observations.copy()
    features["month"] = observations[timestamp_column].dt.month.astype("Int64")
    features["day_of_week"] = observations[timestamp_column].dt.day_of_week.astype(
        "Int64"
    )
    return features
