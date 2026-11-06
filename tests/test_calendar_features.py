import pandas as pd

from cei.features.calendar import add_calendar_features


def test_calendar_features_use_dates_without_filling_missing_dates():
    observations = pd.DataFrame(
        {"Date": pd.to_datetime(["2025-01-01", "2025-01-03", None])}
    )

    features = add_calendar_features(observations, "Date")

    assert features["month"].tolist()[:2] == [1, 1]
    assert features["day_of_week"].tolist()[:2] == [2, 4]
    assert pd.isna(features["month"].iloc[2])
    assert len(features) == len(observations) == 3
    assert "month" not in observations.columns
