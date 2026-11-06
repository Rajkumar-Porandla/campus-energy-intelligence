"""Automated tests for CEI interactive dashboard data loaders, aggregations, and metrics."""
from pathlib import Path
import pandas as pd
import pytest

from app.data_loader import (
    compute_diurnal_profile,
    get_all_building_ids,
    get_anomaly_macro_summary,
    get_anomaly_results_data,
    get_building_series,
    get_forecasting_macro_summary,
    get_forecasting_models_data,
    get_outlier_correlation_data,
    load_all_reports,
    load_building_metadata,
    load_electricity_core,
    load_electricity_quintin,
    load_weather,
)


def test_dashboard_loads_all_six_approved_reports():
    reports = load_all_reports()
    assert set(reports.keys()) == {"phase3a", "phase3b1", "phase3b2", "phase3b3", "phase3c", "phase3d"}
    assert reports["phase3a"]["canonical_validation"]["status"] == "passed"
    assert reports["phase3b1"]["status"] == "validation_only"
    assert reports["phase3b2"]["status"] == "validation_only"
    assert reports["phase3b3"]["report_version"] == "3B.3-remediated-v1"
    assert reports["phase3c"]["report_version"] == "3C-remediated-v1"
    assert reports["phase3d"]["report_version"] == "3D-remediated-v1"


def test_dashboard_metadata_and_core_buildings():
    meta = load_building_metadata()
    assert len(meta) == 30  # 29 core + 1 sensitivity (Quintin)
    assert "Panther_education_Quintin" in meta.index

    core_ids = get_all_building_ids()
    assert len(core_ids) == 29
    assert "Panther_education_Quintin" not in core_ids


def test_dashboard_electricity_data_shapes_and_sensitivity_isolation():
    core_df = load_electricity_core()
    assert len(core_df) == 508776
    assert core_df["building_id"].nunique() == 29
    assert core_df["timestamp"].min() == pd.Timestamp("2016-01-01 00:00:00")
    assert core_df["timestamp"].max() == pd.Timestamp("2017-12-31 23:00:00")

    q_df = load_electricity_quintin()
    assert len(q_df) == 17544
    assert q_df["building_id"].unique().tolist() == ["Panther_education_Quintin"]

    # Verify get_building_series handles regular vs Quintin
    alecia_df = get_building_series("Panther_education_Alecia")
    assert len(alecia_df) == 17544

    with pytest.raises(ValueError, match="Panther_education_Quintin is isolated"):
        get_building_series("Panther_education_Quintin", include_quintin_if_selected=False)

    q_series = get_building_series("Panther_education_Quintin", include_quintin_if_selected=True)
    assert len(q_series) == 17544


def test_dashboard_forecasting_data_and_macros():
    # Test primary task (daily_profile, 168h lookback)
    fc_df = get_forecasting_models_data(task="daily_profile", lookback_hours=168)
    assert len(fc_df) == 29 * 4  # 4 models evaluated on daily_profile: daily_seasonal_naive, sarima, random_forest, xgboost
    models = sorted(fc_df["model"].unique().tolist())
    assert "daily_seasonal_naive" in models
    assert "random_forest" in models
    assert "sarima" in models
    assert "xgboost" in models

    macro_df = get_forecasting_macro_summary()
    assert len(macro_df) == 36  # 3 tasks x 3 lookbacks x 4 models


def test_dashboard_anomaly_data_and_guardrails():
    anom_df = get_anomaly_results_data(
        feature_mode="seasonal_deviation",
        operating_point="1.0pct",
        include_quintin=False
    )
    assert len(anom_df) == 29 * 3  # 29 buildings x 3 models = 87 rows
    # Verify zero valid zero readings flagged
    assert (anom_df["validation_flagged_zero_hours"] == 0).all()

    # Include Quintin
    anom_q = get_anomaly_results_data(
        feature_mode="seasonal_deviation",
        operating_point="1.0pct",
        include_quintin=True
    )
    assert len(anom_q) == 30 * 3  # 90 rows

    macro_anom = get_anomaly_macro_summary()
    assert len(macro_anom) == 18  # 2 modes x 3 operating points x 3 models


def test_dashboard_outlier_correlation_data():
    outlier = get_outlier_correlation_data()
    stats = outlier["stats"]
    assert stats["influential_outlier_building_id"] == "Panther_education_Vincent"
    assert round(stats["pearson_r"], 4) == 0.6906
    assert round(stats["spearman_rho"], 4) == -0.1466
    assert round(stats["excluding_vincent_pearson_r"], 4) == 0.2518
    assert round(stats["influential_outlier_leverage_h"], 4) == 0.8507
    assert round(stats["influential_outlier_cooks_d"], 4) == 1.2558

    bldgs = outlier["buildings_df"]
    assert len(bldgs) == 29


def test_dashboard_diurnal_profile_calculation():
    alecia_df = get_building_series("Panther_education_Alecia")
    diurnal = compute_diurnal_profile(alecia_df)
    assert set(diurnal["partition"].unique()) == {"Training (2016)", "Validation (H1 2017)"}
    assert set(diurnal["day_type"].unique()) == {"Weekday", "Weekend"}
    assert set(diurnal["hour"].unique()) == set(range(24))
