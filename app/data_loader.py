"""Data loader module for the CEI Interactive Dashboard.

Loads canonical electricity, weather, and building metadata alongside all
independently approved Phase 3A-3D validation reports, syntheses, and
analytics/intelligence modules with caching.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.cei.analytics.peak_detection import (
    compute_diurnal_peak_distribution,
    detect_building_peaks,
    detect_campus_coincident_peaks,
)
from src.cei.intelligence.explainability import (
    compute_building_slot_statistics,
    explain_anomaly_observation,
)
from src.cei.intelligence.risk_scoring import (
    compute_building_risk_score,
    compute_portfolio_risk_scores,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/bdg2/panther"
RAW_DIR = ROOT / "data/raw/bdg2/panther"
REPORTS_DIR = ROOT / "data/reports/bdg2/panther"


@st.cache_data(show_spinner=False)
def load_all_reports() -> Dict[str, Any]:
    """Load all approved Phase 3A-3D JSON reports and manifests."""
    p3a_path = REPORTS_DIR / "phase3a_corrected_validation.json"
    p3b1_path = REPORTS_DIR / "phase3b_baseline_mase_audit_validation_v2.json"
    p3b2_path = REPORTS_DIR / "phase3b2_candidate_recursive_validation.json"
    p3b3_path = REPORTS_DIR / "phase3b3_anomaly_remediated_validation.json"
    p3c_path = REPORTS_DIR / "phase3c_integrated_research_synthesis_remediated.json"
    p3d_path = REPORTS_DIR / "phase3d_reproducibility_manifest_remediated.json"

    return {
        "phase3a": json.loads(p3a_path.read_text(encoding="utf-8")),
        "phase3b1": json.loads(p3b1_path.read_text(encoding="utf-8")),
        "phase3b2": json.loads(p3b2_path.read_text(encoding="utf-8")),
        "phase3b3": json.loads(p3b3_path.read_text(encoding="utf-8")),
        "phase3c": json.loads(p3c_path.read_text(encoding="utf-8")),
        "phase3d": json.loads(p3d_path.read_text(encoding="utf-8")),
    }


@st.cache_data(show_spinner=False)
def load_building_metadata() -> pd.DataFrame:
    """Load building metadata filtered to the 29 UCF Panther educational buildings + Quintin."""
    meta_path = RAW_DIR / "metadata.csv"
    if not meta_path.exists():
        alt_path = DATA_DIR / "metadata.csv"
        if alt_path.exists():
            meta_path = alt_path
    df = pd.read_csv(meta_path)
    reports = load_all_reports()
    core_ids = set(reports["phase3b2"]["canonical_validation"]["building_count"] and reports["phase3c"]["coverage_and_regime_synthesis"]["coverage_macro_summary"]["building_id"] if "building_id" in reports["phase3c"]["coverage_and_regime_synthesis"]["coverage_macro_summary"] else [])
    if not core_ids:
        core_ids = set(load_electricity_core()["building_id"].unique())
    valid_ids = core_ids | {"Panther_education_Quintin"}
    df = df[df["building_id"].isin(valid_ids)].copy()
    return df.set_index("building_id")


@st.cache_data(show_spinner=False)
def load_electricity_core() -> pd.DataFrame:
    """Load canonical electricity records for the 29 core buildings (508,776 rows)."""
    core_path = DATA_DIR / "electricity_core.csv"
    if not core_path.exists():
        gz_path = DATA_DIR / "electricity_core.csv.gz"
        if gz_path.exists():
            core_path = gz_path
    df = pd.read_csv(core_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_electricity_quintin() -> pd.DataFrame:
    """Load canonical electricity records for the isolated sensitivity building Panther_education_Quintin."""
    q_path = DATA_DIR / "electricity_sensitivity_quintin.csv"
    df = pd.read_csv(q_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_weather() -> pd.DataFrame:
    """Load site-level weather observations."""
    weather_path = DATA_DIR / "weather.csv"
    df = pd.read_csv(weather_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_building_series(
    building_id: str,
    include_quintin_if_selected: bool = True
) -> pd.DataFrame:
    """Retrieve full electricity time series for a single building."""
    if "Quintin" in building_id:
        if not include_quintin_if_selected:
            raise ValueError("Panther_education_Quintin is isolated for sensitivity analysis only")
        q_df = load_electricity_quintin()
        return q_df[q_df["building_id"] == building_id].sort_values("timestamp").copy()

    core_df = load_electricity_core()
    return core_df[core_df["building_id"] == building_id].sort_values("timestamp").copy()


def get_all_building_ids() -> List[str]:
    """Return sorted list of 29 core buildings."""
    meta = load_building_metadata()
    core_ids = [b for b in sorted(meta.index.tolist()) if "Quintin" not in b]
    return core_ids


def get_partition_dates() -> Dict[str, Tuple[str, str]]:
    """Approved chronological experiment partitions."""
    return {
        "train": ("2016-01-01 00:00:00", "2016-12-31 23:00:00"),
        "validation": ("2017-01-01 00:00:00", "2017-06-30 23:00:00"),
        "test_held_out": ("2017-07-01 00:00:00", "2017-12-31 23:00:00"),
    }


def get_forecasting_models_data(task: str = "daily_profile", lookback_hours: int = 168) -> pd.DataFrame:
    """Extract combined baseline and candidate forecasting rows for all 29 buildings."""
    reports = load_all_reports()
    p3b2 = reports["phase3b2"]

    rows = []
    # 1. Baseline rows
    for r in p3b2["results"]["baseline"]["rows"]:
        if r["task"] == task and r["lookback_hours"] == lookback_hours:
            rows.append({
                "building_id": r["building_id"],
                "model": r.get("method") or r.get("model"),
                "model_category": "Baseline",
                "task": r["task"],
                "lookback_hours": r["lookback_hours"],
                "mae": r["mae"],
                "rmse": r["rmse"],
                "mean_error": r["mean_error"],
                "mase": r["mase_training_daily_naive"],
                "eligible_origins": r["eligible_origins"],
                "eligible_target_hours": r["eligible_target_hours"],
            })

    # 2. Candidate rows
    for r in p3b2["results"]["candidates"]["rows"]:
        if r["task"] == task and r["lookback_hours"] == lookback_hours:
            rows.append({
                "building_id": r["building_id"],
                "model": r.get("model") or r.get("method"),
                "model_category": "ML Candidate",
                "task": r["task"],
                "lookback_hours": r["lookback_hours"],
                "mae": r["mae"],
                "rmse": r["rmse"],
                "mean_error": r["mean_error"],
                "mase": r["mase_training_daily_naive"],
                "eligible_origins": r["eligible_origins"],
                "eligible_target_hours": r["eligible_target_hours"],
            })

    return pd.DataFrame(rows)


def get_forecasting_macro_summary() -> pd.DataFrame:
    """Extract macro-averaged forecasting performance across all task/lookback combinations."""
    reports = load_all_reports()
    p3c = reports["phase3c"]
    return pd.DataFrame(p3c["forecasting_synthesis"]["task_lookback_summary"])


def get_anomaly_results_data(
    feature_mode: str = "seasonal_deviation",
    operating_point: str = "1.0pct",
    include_quintin: bool = False
) -> pd.DataFrame:
    """Extract anomaly detection evaluation rows for the given mode and operating point."""
    reports = load_all_reports()
    p3b3 = reports["phase3b3"]

    rows = []
    for r in p3b3["results"]["rows"]:
        if r["feature_mode"] == feature_mode and r["operating_point"] == operating_point:
            rows.append(r)

    if include_quintin:
        for r in p3b3["results"]["sensitivity_quintin"]["rows"]:
            if r["feature_mode"] == feature_mode and r["operating_point"] == operating_point:
                rows.append(r)

    return pd.DataFrame(rows)


def get_anomaly_macro_summary() -> pd.DataFrame:
    """Extract macro-averaged anomaly detection metrics."""
    reports = load_all_reports()
    p3c = reports["phase3c"]
    return pd.DataFrame(p3c["anomaly_detection_synthesis"]["macro_and_coincidence_summary"])


def get_outlier_correlation_data() -> Dict[str, Any]:
    """Extract the cross-phase correlation and influential outlier diagnostic evidence."""
    reports = load_all_reports()
    p3c = reports["phase3c"]
    corr = p3c["cross_building_heterogeneity_synthesis"]["cross_phase_correlations"][
        "rf_daily_mase_delta_vs_seasonal_lof_rate"
    ]
    profiles = p3c["cross_building_heterogeneity_synthesis"]["building_profiles"]
    df = pd.DataFrame(profiles)
    return {
        "stats": corr,
        "buildings_df": df,
    }


def compute_diurnal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average 24-hour diurnal profile split by 2016 vs H1 2017 and Weekday vs Weekend."""
    sub = df.copy()
    sub["year"] = sub["timestamp"].dt.year
    sub["hour"] = sub["timestamp"].dt.hour
    sub["dayofweek"] = sub["timestamp"].dt.day_of_week
    sub["is_weekend"] = sub["dayofweek"] >= 5
    sub["day_type"] = sub["is_weekend"].map({True: "Weekend", False: "Weekday"})

    val_cutoff = pd.Timestamp("2017-07-01 00:00:00")
    sub["partition"] = "Held-Out Test (H2 2017)"
    sub.loc[sub["year"] == 2016, "partition"] = "Training (2016)"
    sub.loc[(sub["year"] == 2017) & (sub["timestamp"] < val_cutoff), "partition"] = "Validation (H1 2017)"

    grouped = (
        sub[sub["partition"].isin(["Training (2016)", "Validation (H1 2017)"])]
        .groupby(["partition", "day_type", "hour"])["energy_consumption"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped


# --- NEW ANALYTICS & INTELLIGENCE INTEGRATION HELPERS ---

@st.cache_data(show_spinner=False)
def get_building_peak_analysis(building_id: str) -> Dict[str, Any]:
    """Retrieve peak detection metrics and diurnal peak distribution for a building."""
    df = get_building_series(building_id)
    peak_res = detect_building_peaks(df, threshold_percentile=99.0)
    diurnal_peaks = compute_diurnal_peak_distribution(df)
    return {
        "peaks_df": peak_res["peak_hours_df"],
        "diurnal_peaks": diurnal_peaks,
        "peak_count": peak_res["peak_event_count"],
        "max_peak_kwh": peak_res["peak_kwh"],
        "threshold_kwh": peak_res["threshold_kwh"],
        "load_factor": peak_res["load_factor"],
    }


@st.cache_data(show_spinner=False)
def get_campus_coincident_peaks(top_k: int = 10) -> pd.DataFrame:
    """Retrieve coincident peak demand events across all 29 core buildings."""
    core_df = load_electricity_core()
    return detect_campus_coincident_peaks(core_df, top_n=top_k)


@st.cache_data(show_spinner=False)
def get_portfolio_risk_scores(
    feature_mode: str = "seasonal_deviation",
    operating_point: str = "1.0pct"
) -> pd.DataFrame:
    """Compute and rank composite Energy Risk Index (0-100) across all 29 core buildings."""
    core_df = load_electricity_core()
    reports = load_all_reports()
    profiles_df = pd.DataFrame(reports["phase3c"]["cross_building_heterogeneity_synthesis"]["building_profiles"])
    return compute_portfolio_risk_scores(profiles_df=profiles_df, core_df=core_df)


@st.cache_data(show_spinner=False)
def get_anomaly_explanations(
    building_id: str,
    operating_point: str = "1.0pct",
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """Explain top anomaly observations for a building using seasonal slot decomposition."""
    df = get_building_series(building_id)
    slot_stats = compute_building_slot_statistics(df)
    anom_rows = get_anomaly_results_data(feature_mode="seasonal_deviation", operating_point=operating_point)
    b_rows = anom_rows[anom_rows["building_id"] == building_id]

    threshold = 2.5
    if not b_rows.empty and "training_threshold" in b_rows.columns:
        threshold = float(b_rows.iloc[0]["training_threshold"])

    # Find candidate anomalous points in validation set (H1 2017)
    val_df = df[(df["timestamp"] >= "2017-01-01") & (df["timestamp"] < "2017-07-01")].copy()
    val_df = val_df.sort_values("timestamp")
    val_df["prior_consumption"] = val_df["energy_consumption"].shift(1)

    explanations = []
    # Identify high deviation points
    for _, row in val_df.iterrows():
        val = row["energy_consumption"]
        ts = row["timestamp"]
        prior = row["prior_consumption"] if pd.notna(row["prior_consumption"]) else None
        slot_idx = ts.day_of_week * 24 + ts.hour
        s_mean = slot_stats[slot_idx]["mean"]
        s_std = slot_stats[slot_idx]["std"]
        
        # Check if significantly deviant
        if s_std > 0 and abs(val - s_mean) / s_std >= 2.0:
            exp = explain_anomaly_observation(
                building_id=building_id,
                timestamp=ts,
                consumption=val,
                baseline_slot_mean=s_mean,
                baseline_slot_std=s_std,
                prev_hour_consumption=prior,
                threshold_value=threshold,
                model_name="isolation_forest",
                feature_mode="seasonal_deviation",
            )
            explanations.append(exp)
            if len(explanations) >= top_k:
                break

    return explanations


def get_validation_forecast_sample(
    building_id: str,
    start_date: str = "2017-03-01",
    days: int = 7
) -> pd.DataFrame:
    """Generate actual vs Daily Seasonal Naive baseline forecast over a representative validation window."""
    df = get_building_series(building_id)
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + pd.Timedelta(days=days)

    window_df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].sort_values("timestamp").copy()

    # Build daily seasonal naive (lag 24h) and persistence (lag 1h)
    lookup = dict(zip(df["timestamp"], df["energy_consumption"]))

    window_df["forecast_daily_seasonal_naive"] = [
        lookup.get(ts - pd.Timedelta(hours=24), None) for ts in window_df["timestamp"]
    ]
    window_df["forecast_persistence"] = [
        lookup.get(ts - pd.Timedelta(hours=1), None) for ts in window_df["timestamp"]
    ]

    return window_df
