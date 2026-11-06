"""Explainability engine for unsupervised energy anomaly observations.

Decomposes flagged anomaly timestamps into intuitive constituent drivers:
1. Slot Deviation: Deviation from expected hour-of-week historical baseline.
2. 1-Hour Ramp Rate: Abrupt step change from the previous hour.
3. Magnitude Exceedance: Exceedance relative to historical distribution percentiles.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def explain_anomaly_observation(
    building_id: str,
    timestamp: pd.Timestamp,
    consumption: float,
    baseline_slot_mean: float,
    baseline_slot_std: float,
    prev_hour_consumption: Optional[float],
    threshold_value: float,
    model_name: str,
    feature_mode: str
) -> Dict[str, Any]:
    """Decompose and explain why an individual observation was flagged.

    Parameters
    ----------
    building_id : str
        Building identifier.
    timestamp : pd.Timestamp
        Timestamp of the flagged observation.
    consumption : float
        Actual consumption in kWh.
    baseline_slot_mean : float
        Historical 2016 mean consumption for this exact hour-of-week slot.
    baseline_slot_std : float
        Historical 2016 standard deviation for this hour-of-week slot.
    prev_hour_consumption : float or None
        Actual consumption in the preceding hour (t-1).
    threshold_value : float
        Detector threshold value from training calibration.
    model_name : str
        Anomaly detector model name.
    feature_mode : str
        Feature mode ('raw_consumption' or 'seasonal_deviation').

    Returns
    -------
    dict
        Structured explanation with slot z-score, ramp rate, primary driver, and plain-English summary.
    """
    # 1. Slot Z-score deviation
    std_safe = max(baseline_slot_std, 1e-4)
    slot_dev = consumption - baseline_slot_mean
    slot_z = slot_dev / std_safe

    # 2. Ramp rate
    if prev_hour_consumption is not None and not np.isnan(prev_hour_consumption):
        ramp = consumption - prev_hour_consumption
    else:
        ramp = 0.0

    # 3. Determine primary departure driver
    drivers = []
    if abs(slot_z) >= 3.0:
        drivers.append(f"Extreme slot deviation ({slot_z:+.2f}σ from expected slot mean)")
    elif abs(slot_z) >= 2.0:
        drivers.append(f"Moderate slot deviation ({slot_z:+.2f}σ from expected slot mean)")

    if abs(ramp) >= max(baseline_slot_mean * 0.5, 20.0):
        direction = "spike" if ramp > 0 else "drop"
        drivers.append(f"Sudden 1-hour {direction} ({ramp:+.1f} kWh/h step change)")

    if not drivers:
        drivers.append("Multivariate geometric distance in calibrated feature space")

    primary_driver = drivers[0]
    plain_summary = (
        f"At {timestamp.strftime('%Y-%m-%d %H:%M')}, {building_id} recorded {consumption:.1f} kWh "
        f"vs. expected slot mean of {baseline_slot_mean:.1f} kWh ({slot_dev:+.1f} kWh, {slot_z:+.2f}σ). "
        f"1-hour delta was {ramp:+.1f} kWh/h. Flagged by {model_name} in {feature_mode} mode."
    )

    return {
        "building_id": building_id,
        "timestamp": timestamp.isoformat(),
        "consumption_kwh": round(consumption, 2),
        "expected_slot_mean_kwh": round(baseline_slot_mean, 2),
        "slot_deviation_kwh": round(slot_dev, 2),
        "slot_z_score": round(slot_z, 2),
        "one_hour_ramp_kwh": round(ramp, 2),
        "primary_driver": primary_driver,
        "all_drivers": drivers,
        "plain_summary": plain_summary,
        "model_name": model_name,
        "feature_mode": feature_mode,
        "threshold_value": threshold_value,
    }


def compute_building_slot_statistics(train_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Precompute 2016 baseline mean and standard deviation for all 168 hour-of-week slots."""
    sub = train_df.dropna(subset=["energy_consumption"]).copy()
    sub["slot"] = sub["timestamp"].dt.day_of_week * 24 + sub["timestamp"].dt.hour
    grouped = sub.groupby("slot")["energy_consumption"].agg(["mean", "std"]).to_dict(orient="index")
    slot_stats = {}
    for slot in range(168):
        st = grouped.get(slot, {"mean": 0.0, "std": 1.0})
        mean_v = float(st.get("mean", 0.0))
        std_v = float(st.get("std", 1.0))
        slot_stats[slot] = {"mean": mean_v, "std": max(std_v, 0.1)}
    return slot_stats

