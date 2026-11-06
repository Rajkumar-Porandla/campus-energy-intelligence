"""Defensible energy operational risk scoring engine for campus facilities.

Calculates the Energy Risk Index (ERI, 0-100) based on observable consumption volatility,
peak demand stress, zero-shift exposure, and multi-step forecasting uncertainty.

DISCLAIMER: The Energy Risk Index evaluates energy demand volatility and forecasting
uncertainty for facility management prioritization. It does NOT represent a probability
of physical equipment failure or electrical infrastructure breakdown.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def compute_building_risk_score(
    building_id: str,
    load_factor: float,
    coeff_variation: float,
    train_zero_per_1000h: float,
    seasonal_lof_alert_rate: float,
    daily_rf_mase: float
) -> Dict[str, Any]:
    """Compute the multi-factor Energy Risk Index (ERI) for a single building.

    Parameters
    ----------
    building_id : str
        Building identifier.
    load_factor : float
        Ratio of mean load to peak demand in [0, 1].
    coeff_variation : float
        Ratio of standard deviation to mean (sigma / mu).
    train_zero_per_1000h : float
        2016 valid zero rate per 1,000 observed hours.
    seasonal_lof_alert_rate : float
        H1 2017 validation alert rate from LOF in seasonal_deviation mode (alerts/1,000h).
    daily_rf_mase : float
        Multi-step daily Random Forest MASE (H=24, lookback=168h).

    Returns
    -------
    dict
        Composite ERI score (0-100), risk tier, sub-component breakdowns, and methodology disclaimer.
    """
    # 1. Peak Demand Stress (0 - 20 points): lower load factor means spikier load
    peak_stress_pts = float(np.clip((1.0 - load_factor) * 20.0, 0.0, 20.0))

    # 2. Operational Dispersion (0 - 20 points): normalized coefficient of variation
    volatility_pts = float(np.clip((coeff_variation / 1.25) * 20.0, 0.0, 20.0))

    # 3. 2016 Zero-Regime Exposure (0 - 20 points): vulnerability to 2016 zero metering shift
    zero_shift_pts = float(np.clip((train_zero_per_1000h / 400.0) * 20.0, 0.0, 20.0))

    # 4. Anomaly Alert Frequency (0 - 20 points): geometric/statistical departure frequency
    anomaly_pts = float(np.clip((seasonal_lof_alert_rate / 35.0) * 20.0, 0.0, 20.0))

    # 5. Forecast Uncertainty (0 - 20 points): daily profile recursive forecasting difficulty
    forecast_pts = float(np.clip((daily_rf_mase / 2.5) * 20.0, 0.0, 20.0))

    total_score = round(peak_stress_pts + volatility_pts + zero_shift_pts + anomaly_pts + forecast_pts, 1)

    if total_score < 30.0:
        tier = "Low"
        tier_color = "#568B87"  # Teal
    elif total_score < 60.0:
        tier = "Moderate"
        tier_color = "#D9AE54"  # Mustard
    elif total_score < 80.0:
        tier = "Elevated"
        tier_color = "#C76B43"  # Burnt Orange
    else:
        tier = "High"
        tier_color = "#f43f5e"  # Crimson

    return {
        "building_id": building_id,
        "energy_risk_index": total_score,
        "risk_tier": tier,
        "tier_color": tier_color,
        "components": {
            "peak_demand_stress": round(peak_stress_pts, 1),
            "load_volatility": round(volatility_pts, 1),
            "zero_shift_exposure": round(zero_shift_pts, 1),
            "anomaly_frequency": round(anomaly_pts, 1),
            "forecast_uncertainty": round(forecast_pts, 1),
        },
        "methodology": {
            "scale": "0 - 100",
            "scale_type": "Composite Operational Volatility & Forecasting Uncertainty",
            "disclaimer": (
                "The Energy Risk Index is a descriptive operational prioritization metric. "
                "It evaluates load volatility and forecasting uncertainty. It does NOT "
                "represent a probability of physical equipment failure or electrical fault."
            ),
        },
    }


def compute_portfolio_risk_scores(
    profiles_df: pd.DataFrame,
    core_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute risk scores for all 29 core buildings."""
    rows = []
    for _, prof in profiles_df.iterrows():
        b_id = prof["building_id"]
        b_series = core_df[core_df["building_id"] == b_id]["energy_consumption"].dropna().values
        if len(b_series) == 0:
            continue

        mean_val = float(np.mean(b_series))
        peak_val = float(np.max(b_series))
        std_val = float(np.std(b_series))
        load_factor = float(mean_val / peak_val) if peak_val > 0 else 1.0
        cv = float(std_val / mean_val) if mean_val > 0 else 0.0

        train_zero = float(prof.get("train_valid_zero_per_1000_hours", 381.0))
        lof_alert = float(prof.get("anom_1pct_seasonal_lof_per_1000h", 15.0))
        daily_mase = float(prof.get("daily_baseline_mase", 1.6))
        if "daily_rf_mase_delta" in prof:
            daily_mase += float(prof["daily_rf_mase_delta"])

        res = compute_building_risk_score(
            building_id=b_id,
            load_factor=load_factor,
            coeff_variation=cv,
            train_zero_per_1000h=train_zero,
            seasonal_lof_alert_rate=lof_alert,
            daily_rf_mase=daily_mase,
        )
        rows.append({
            "building_id": b_id,
            "energy_risk_index": res["energy_risk_index"],
            "risk_tier": res["risk_tier"],
            "peak_stress_pts": res["components"]["peak_demand_stress"],
            "volatility_pts": res["components"]["load_volatility"],
            "zero_shift_pts": res["components"]["zero_shift_exposure"],
            "anomaly_pts": res["components"]["anomaly_frequency"],
            "forecast_pts": res["components"]["forecast_uncertainty"],
        })

    return pd.DataFrame(rows).sort_values("energy_risk_index", ascending=False)

