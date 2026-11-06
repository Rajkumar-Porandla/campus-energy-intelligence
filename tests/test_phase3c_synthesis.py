"""Tests for Phase 3C integrated evaluation and research synthesis."""
import copy
import json
import math
from pathlib import Path

from cei.synthesis.phase3c_synthesis import (
    EXPECTED_DATASET_CHECKSUMS,
    EXPECTED_PRIOR_PHASE3C_CHECKSUMS,
    EXPECTED_SOURCE_REPORT_CHECKSUMS,
    build_phase3c_synthesis,
    render_markdown_report,
    verify_prior_phase3c_checksums,
    verify_protected_checksums,
    verify_source_report_checksums,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUPERVISED_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "auc"}


def test_phase3c_verifies_all_protected_dataset_and_source_report_checksums():
    datasets = verify_protected_checksums(ROOT)
    reports = verify_source_report_checksums(ROOT)
    prior_p3c = verify_prior_phase3c_checksums(ROOT)
    assert datasets == EXPECTED_DATASET_CHECKSUMS
    assert reports == EXPECTED_SOURCE_REPORT_CHECKSUMS
    assert prior_p3c == EXPECTED_PRIOR_PHASE3C_CHECKSUMS


def test_phase3c_synthesis_reproduces_approved_forecasting_and_anomaly_metrics():
    syn = build_phase3c_synthesis(ROOT)
    assert syn["phase"] == "3C"
    assert syn["report_version"] == "3C-remediated-v1"
    assert syn["status"] == "validation_synthesis_only"
    assert syn["test_evaluation"] == "not_run"
    assert syn["ground_truth_anomaly_labels"] == "none"

    # Verify JSON serialization with allow_nan=False
    serialized = json.dumps(syn, allow_nan=False)
    assert len(serialized) > 1000

    # Verify no test partition rows in coverage macro summary
    cov_rows = syn["coverage_and_regime_synthesis"]["coverage_macro_summary"]
    assert len(cov_rows) == 2 * 3 * 3
    assert all(r["partition"] in ("train", "validation") for r in cov_rows)
    assert all(r["building_count"] == 29 for r in cov_rows)

    # Verify 2016-to-2017 valid-zero shift numbers
    zs = syn["coverage_and_regime_synthesis"]["zero_frequency_shift_summary"]
    assert zs["core_building_count"] == 29
    assert math.isclose(zs["core_train_valid_zero_per_1000_macro_mean"], 381.01637747168775, rel_tol=1e-12)
    assert zs["core_validation_valid_zero_per_1000_macro_mean"] == 0.0
    assert "sensitivity_quintin" in zs

    # Verify forecasting task-lookback summary (3 tasks x 3 lookbacks x 4 models = 36 rows)
    fc_rows = syn["forecasting_synthesis"]["task_lookback_summary"]
    assert len(fc_rows) == 36
    rf_h_168 = next(
        r for r in fc_rows
        if r["task"] == "hourly" and r["lookback_hours"] == 168 and r["model"] == "random_forest"
    )
    assert math.isclose(rf_h_168["mae_macro_mean"], 6.730917559716886, rel_tol=1e-12)
    assert rf_h_168["buildings_improved_mae"] == 18
    assert rf_h_168["buildings_improved_rmse"] == 19

    dn_d_168 = next(
        r for r in fc_rows
        if r["task"] == "daily_profile" and r["lookback_hours"] == 168 and r["model"] == "daily_seasonal_naive"
    )
    assert math.isclose(dn_d_168["mae_macro_mean"], 12.158284129739434, rel_tol=1e-12)

    # Verify win counts sum to 29 for every task and lookback
    for w in syn["forecasting_synthesis"]["building_win_count_summary"]:
        assert w["building_count"] == 29
        assert sum(w["best_mae_model_counts"].values()) == 29
        assert sum(w["best_rmse_model_counts"].values()) == 29

    # Verify anomaly synthesis (18 macro rows) and conditional durations >= 1.0h
    anom_rows = syn["anomaly_detection_synthesis"]["macro_and_coincidence_summary"]
    assert len(anom_rows) == 18
    for m in anom_rows:
        assert m["building_count"] == 29
        assert m["buildings_with_events_count"] + m["zero_event_buildings_count"] == 29
        assert m["mean_event_duration_macro_mean"] >= 1.0
        assert m["total_flagged_zero_hours"] == 0

    # Verify cross-building profiles exclude Quintin and contain 29 buildings
    cb = syn["cross_building_heterogeneity_synthesis"]
    assert cb["building_count"] == 29
    assert len(cb["building_profiles"]) == 29
    assert all("Quintin" not in p["building_id"] for p in cb["building_profiles"])

    corr = cb["cross_phase_correlations"]
    assert corr["status"] == "exploratory_outlier_sensitive_diagnostic"
    assert corr["is_robust_general_relationship"] is False
    assert corr["causal_claim_permitted"] is False
    assert math.isclose(
        corr["pearson_rf_daily_mase_delta_vs_seasonal_lof_rate"],
        0.6905667547821152,
        rel_tol=1e-12,
    )
    assert math.isclose(
        corr["pearson_rf_daily_abs_bias_vs_mae_delta"],
        0.5801605146661392,
        rel_tol=1e-12,
    )

    c_lof = corr["rf_daily_mase_delta_vs_seasonal_lof_rate"]
    assert c_lof["n_buildings"] == 29
    assert math.isclose(c_lof["pearson_r"], 0.6905667547821152, rel_tol=1e-12)
    assert math.isclose(c_lof["pearson_p_value"], 3.380160022712965e-05, rel_tol=1e-12)
    assert math.isclose(c_lof["spearman_rho"], -0.14658783396332514, rel_tol=1e-12)
    assert math.isclose(c_lof["spearman_p_value"], 0.4479787272617319, rel_tol=1e-12)
    assert c_lof["influential_outlier_building_id"] == "Panther_education_Vincent"
    assert math.isclose(c_lof["influential_outlier_leverage_h"], 0.8506615952605636, rel_tol=1e-12)
    assert math.isclose(c_lof["influential_outlier_cooks_d"], 1.2558116307862284, rel_tol=1e-12)
    assert c_lof["excluding_vincent_n"] == 28
    assert math.isclose(c_lof["excluding_vincent_pearson_r"], 0.25176125453061515, rel_tol=1e-12)
    assert math.isclose(c_lof["excluding_vincent_pearson_p_value"], 0.19622099736829424, rel_tol=1e-12)

    c_zero = corr["train_valid_zero_rate_vs_rf_daily_mase_delta"]
    assert math.isclose(c_zero["pearson_r"], 0.23295201229829743, rel_tol=1e-12)
    assert math.isclose(c_zero["pearson_p_value"], 0.2239389157512947, rel_tol=1e-12)
    assert math.isclose(c_zero["spearman_rho"], 0.32862147983686346, rel_tol=1e-12)
    assert math.isclose(c_zero["spearman_p_value"], 0.08176274257103006, rel_tol=1e-12)

    # Verify no forbidden supervised metric keys exist anywhere in the synthesis payload
    def _collect_keys(obj):
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(str(k).lower())
                keys.update(_collect_keys(v))
        elif isinstance(obj, list):
            for item in obj:
                keys.update(_collect_keys(item))
        return keys

    assert FORBIDDEN_SUPERVISED_KEYS.isdisjoint(_collect_keys(syn))


def test_phase3c_deterministic_markdown_rendering_and_qualified_correlation_reporting():
    syn = build_phase3c_synthesis(ROOT)
    md_in_memory = render_markdown_report(syn)

    # Round-trip through sort_keys=True JSON and reverse win-count dict key order
    syn_sorted = json.loads(json.dumps(syn, sort_keys=True))
    md_from_sorted_json = render_markdown_report(syn_sorted)
    assert md_in_memory == md_from_sorted_json

    syn_reversed = copy.deepcopy(syn)
    for w in syn_reversed["forecasting_synthesis"]["building_win_count_summary"]:
        w["best_mae_model_counts"] = dict(reversed(list(w["best_mae_model_counts"].items())))
        w["best_rmse_model_counts"] = dict(reversed(list(w["best_rmse_model_counts"].items())))
    assert render_markdown_report(syn_reversed) == md_in_memory

    # Verify required correlation qualifications and forecasting mechanism explanations in Markdown
    assert "0.6906" in md_in_memory
    assert "-0.1466" in md_in_memory
    assert "0.4480" in md_in_memory
    assert "0.2518" in md_in_memory
    assert "0.1962" in md_in_memory
    assert "0.8507" in md_in_memory
    assert "1.2558" in md_in_memory
    assert "(1,0,0)x(0,0,0,0)" in md_in_memory
    assert "n_estimators=40" in md_in_memory
    assert "learning_rate=0.05" in md_in_memory
    assert "lag < lookback" in md_in_memory
    assert "demonstrating that both tasks are governed by" not in md_in_memory
