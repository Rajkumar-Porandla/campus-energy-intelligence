"""Peak electricity consumption detection and load-factor analytics.

Provides algorithms for identifying building-level and campus-coincident peak demand,
load duration exceedances, and diurnal peak distributions without modifying raw data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def detect_building_peaks(
    df: pd.DataFrame,
    threshold_percentile: float = 99.0
) -> Dict[str, Any]:
    """Detect peak demand events and operational load factors for a building time series.

    Parameters
    ----------
    df : pd.DataFrame
        Building time series containing 'timestamp' and 'energy_consumption'.
    threshold_percentile : float
        Percentile cutoff (e.g. 99.0) above which an observation is classified as peak demand.

    Returns
    -------
    dict
        Peak metrics including peak_kwh, mean_kwh, load_factor, threshold_kwh,
        peak_event_count, and top peak observations.
    """
    clean_series = df.dropna(subset=["energy_consumption"]).copy()
    if clean_series.empty:
        return {
            "peak_kwh": 0.0,
            "mean_kwh": 0.0,
            "load_factor": 0.0,
            "threshold_kwh": 0.0,
            "peak_event_count": 0,
            "peak_hours_df": pd.DataFrame(),
        }

    vals = clean_series["energy_consumption"].values
    peak_val = float(np.max(vals))
    mean_val = float(np.mean(vals))
    # Load factor = mean / peak (ratio in [0, 1]; lower load factor indicates peakier, more volatile load)
    load_factor = float(mean_val / peak_val) if peak_val > 0 else 1.0

    threshold_val = float(np.percentile(vals, threshold_percentile))
    peak_mask = clean_series["energy_consumption"] >= threshold_val
    peak_df = clean_series[peak_mask].sort_values("energy_consumption", ascending=False)

    return {
        "peak_kwh": round(peak_val, 2),
        "mean_kwh": round(mean_val, 2),
        "load_factor": round(load_factor, 4),
        "threshold_kwh": round(threshold_val, 2),
        "peak_event_count": int(len(peak_df)),
        "peak_hours_df": peak_df[["timestamp", "energy_consumption"]],
    }


def detect_campus_coincident_peaks(
    core_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Identify the top coincident campus peak hours across all core buildings.

    Parameters
    ----------
    core_df : pd.DataFrame
        Multi-building electricity dataframe with 'timestamp', 'building_id', 'energy_consumption'.
    top_n : int
        Number of top aggregate peak hours to return.

    Returns
    -------
    pd.DataFrame
        Top coincident peak timestamps with aggregate consumption, reporting buildings, and mean load.
    """
    agg = (
        core_df.groupby("timestamp")["energy_consumption"]
        .agg(["sum", "count", "mean", "max"])
        .reset_index()
    )
    agg.rename(
        columns={
            "sum": "campus_total_kwh",
            "count": "active_buildings_count",
            "mean": "building_mean_kwh",
            "max": "max_single_building_kwh",
        },
        inplace=True,
    )
    top_peaks = agg.sort_values("campus_total_kwh", ascending=False).head(top_n).copy()
    top_peaks["campus_total_kwh"] = top_peaks["campus_total_kwh"].round(2)
    top_peaks["building_mean_kwh"] = top_peaks["building_mean_kwh"].round(2)
    top_peaks["max_single_building_kwh"] = top_peaks["max_single_building_kwh"].round(2)
    return top_peaks


def compute_diurnal_peak_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the hourly frequency distribution of peak demand (95th+ percentile)."""
    clean_series = df.dropna(subset=["energy_consumption"]).copy()
    if clean_series.empty:
        return pd.DataFrame({"hour": list(range(24)), "peak_count": [0] * 24})

    cutoff = np.percentile(clean_series["energy_consumption"].values, 95.0)
    peaks = clean_series[clean_series["energy_consumption"] >= cutoff].copy()
    peaks["hour"] = peaks["timestamp"].dt.hour

    hour_counts = peaks.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    hour_counts.columns = ["hour", "peak_count"]
    return hour_counts

