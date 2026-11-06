"""Tests for peak detection, defensible energy risk scoring, and explainability modules."""
import pandas as pd
import pytest

from app.data_loader import (
    get_building_series,
    load_all_reports,
    load_electricity_core,
)
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


def test_detect_building_peaks():
    df = get_building_series("Panther_education_Alecia")
    res = detect_building_peaks(df, threshold_percentile=99.0)
    assert res["peak_kwh"] > 0
    assert res["mean_kwh"] > 0
    assert 0.0 <= res["load_factor"] <= 1.0
    assert res["threshold_kwh"] > 0
    assert res["peak_event_count"] > 0
    peaks_df = res["peak_hours_df"]
    assert not peaks_df.empty
    assert (peaks_df["energy_consumption"] >= res["threshold_kwh"] - 0.05).all()

    diurnal = compute_diurnal_peak_distribution(df)
    assert len(diurnal) == 24
    assert "peak_count" in diurnal.columns


def test_detect_campus_coincident_peaks():
    core_df = load_electricity_core()
    coincident = detect_campus_coincident_peaks(core_df, top_n=10)
    assert len(coincident) == 10
    assert "campus_total_kwh" in coincident.columns
    assert "active_buildings_count" in coincident.columns
    # Total demand should be monotonically non-increasing
    assert coincident["campus_total_kwh"].is_monotonic_decreasing


def test_compute_building_and_portfolio_risk_score():
    risk = compute_building_risk_score(
        building_id="Panther_education_Alecia",
        load_factor=0.65,
        coeff_variation=0.45,
        train_zero_per_1000h=381.0,
        seasonal_lof_alert_rate=15.2,
        daily_rf_mase=1.85,
    )
    assert 0.0 <= risk["energy_risk_index"] <= 100.0
    assert risk["risk_tier"] in ("Low", "Moderate", "Elevated", "High")
    assert 0.0 <= risk["components"]["peak_demand_stress"] <= 20.0
    assert 0.0 <= risk["components"]["load_volatility"] <= 20.0
    assert 0.0 <= risk["components"]["zero_shift_exposure"] <= 20.0
    assert 0.0 <= risk["components"]["anomaly_frequency"] <= 20.0
    assert 0.0 <= risk["components"]["forecast_uncertainty"] <= 20.0

    core_df = load_electricity_core()
    reports = load_all_reports()
    profiles_df = pd.DataFrame(reports["phase3c"]["cross_building_heterogeneity_synthesis"]["building_profiles"])
    portfolio = compute_portfolio_risk_scores(profiles_df=profiles_df, core_df=core_df)
    assert len(portfolio) == 29
    assert (portfolio["energy_risk_index"] >= 0.0).all()
    assert (portfolio["energy_risk_index"] <= 100.0).all()


def test_explain_anomaly_observation():
    df = get_building_series("Panther_education_Alecia")
    slot_stats = compute_building_slot_statistics(df)
    assert len(slot_stats) == 168  # 7 days * 24 hours

    # Test an observation
    ts = pd.Timestamp("2017-03-01 14:00:00")
    val = 150.0
    slot_idx = ts.day_of_week * 24 + ts.hour
    s_mean = slot_stats[slot_idx]["mean"]
    s_std = slot_stats[slot_idx]["std"]

    explanation = explain_anomaly_observation(
        building_id="Panther_education_Alecia",
        timestamp=ts,
        consumption=val,
        baseline_slot_mean=s_mean,
        baseline_slot_std=s_std,
        prev_hour_consumption=110.0,
        threshold_value=2.5,
        model_name="isolation_forest",
        feature_mode="seasonal_deviation",
    )

    assert "primary_driver" in explanation
    assert "plain_summary" in explanation
    assert explanation["consumption_kwh"] == 150.0
    assert explanation["one_hour_ramp_kwh"] == 40.0
