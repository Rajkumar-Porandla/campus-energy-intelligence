"""Read-only Phase 3C synthesis of approved Phase 3A, 3B.1, 3B.2, and 3B.3 validation reports."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy import stats

EXPECTED_DATASET_CHECKSUMS = {
    "data/processed/bdg2/panther/electricity_core.csv": (
        "13cb0556da3a56fbd0eaacc6177c9e078ad42bc02e092f40df4b5dabc2881c8d"
    ),
    "data/processed/bdg2/panther/electricity_sensitivity_quintin.csv": (
        "302076f87ff6fd7423ac97cf3783575584a28b055d047d0c323ead892307a14e"
    ),
    "data/processed/bdg2/panther/weather.csv": (
        "bf34dd1fb1fbc5c00754d1869c4388c9353dac8d19094288dcac7739510f50cb"
    ),
    "data/raw/bdg2/panther/electricity.csv": (
        "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"
    ),
    "data/raw/bdg2/panther/metadata.csv": (
        "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"
    ),
    "data/raw/bdg2/panther/weather.csv": (
        "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd"
    ),
}

EXPECTED_SOURCE_REPORT_CHECKSUMS = {
    "data/reports/bdg2/panther/phase3a_final_validation.json": (
        "845c73e429fa3bd38e50f53d71961e2e57be5167246a04e7de968647e7e141c3"
    ),
    "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json": (
        "95c33e83b1dc374bed4e25a6cfc85c55e8415a88ff12c2a5fd8afeeeb63a978e"
    ),
    "data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json": (
        "d15f06e47c163642934484361600d010d28401624b7c9070494bd53ba311e24f"
    ),
    "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json": (
        "e400244a4db4209dd4fdd4ab544c2df48427dd876b0b04d06c4a3a44c44e2575"
    ),
}

EXPECTED_PRIOR_PHASE3C_CHECKSUMS = {
    "data/reports/bdg2/panther/phase3c_integrated_research_synthesis.json": (
        "4599210c9deb9206d5e6c1b82dd2700d4a0e926ac8227ed7d105ad849d830552"
    ),
    "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS.md": (
        "b78f2c23708e22eff7724f7d70b9721b3d7aa87a3c9921e52f7bf6470aedc05e"
    ),
}

TASKS = ("hourly", "daily_profile", "weekly_profile")
LOOKBACKS = (24, 48, 168)
CANDIDATE_MODELS = ("sarima", "random_forest", "xgboost")
TASK_BASELINES = {
    "hourly": "persistence",
    "daily_profile": "daily_seasonal_naive",
    "weekly_profile": "weekly_seasonal_naive",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protected_checksums(root):
    root_path = Path(root)
    verified = {}
    for rel_path, expected in EXPECTED_DATASET_CHECKSUMS.items():
        actual = sha256_file(root_path / rel_path)
        if actual != expected:
            raise ValueError(f"Protected dataset checksum mismatch for {rel_path}: {actual} != {expected}")
        verified[rel_path] = actual
    return verified


def verify_source_report_checksums(root):
    root_path = Path(root)
    verified = {}
    for rel_path, expected in EXPECTED_SOURCE_REPORT_CHECKSUMS.items():
        actual = sha256_file(root_path / rel_path)
        if actual != expected:
            raise ValueError(f"Source report checksum mismatch for {rel_path}: {actual} != {expected}")
        verified[rel_path] = actual
    return verified


def verify_prior_phase3c_checksums(root):
    root_path = Path(root)
    verified = {}
    for rel_path, expected in EXPECTED_PRIOR_PHASE3C_CHECKSUMS.items():
        actual = sha256_file(root_path / rel_path)
        if actual != expected:
            raise ValueError(f"Prior Phase 3C report checksum mismatch for {rel_path}: {actual} != {expected}")
        verified[rel_path] = actual
    return verified


def load_approved_reports(root):
    root_path = Path(root)
    verify_protected_checksums(root_path)
    verify_source_report_checksums(root_path)
    reports = {}
    for key, rel_path in (
        ("phase3a", "data/reports/bdg2/panther/phase3a_final_validation.json"),
        ("phase3b1", "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json"),
        ("phase3b2", "data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json"),
        ("phase3b3", "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json"),
    ):
        with (root_path / rel_path).open(encoding="utf-8") as source:
            reports[key] = json.load(source)
    return reports


def synthesize_coverage_and_regime_shift(p3a, p3b3):
    coverage_rows = [
        r for r in p3a["coverage"]
        if r["partition"] in ("train", "validation")
    ]
    if any("Quintin" in r["building_id"] for r in coverage_rows):
        raise ValueError("Sensitivity building Quintin must not appear in Phase 3A core coverage")

    coverage_macro = []
    for partition in ("train", "validation"):
        for task in TASKS:
            for lb in LOOKBACKS:
                sub = [
                    r for r in coverage_rows
                    if r["partition"] == partition and r["task"] == task and r["lookback_hours"] == lb
                ]
                cand_origins = [r["candidate_origins"] for r in sub]
                elig_origins = [r["eligible_origins"] for r in sub]
                elig_targets = [r["eligible_target_hours"] for r in sub]
                hist_excl = [r["insufficient_history_exclusions"] for r in sub]
                miss_excl = [r["missing_target_exclusions"] for r in sub]
                coverage_macro.append({
                    "partition": partition,
                    "task": task,
                    "horizon_hours": int(sub[0]["horizon_hours"]),
                    "lookback_hours": lb,
                    "building_count": len(sub),
                    "candidate_origins_per_building": int(cand_origins[0]),
                    "eligible_origins_macro_mean": float(np.mean(elig_origins)),
                    "eligible_origins_range": [int(min(elig_origins)), int(max(elig_origins))],
                    "eligible_target_hours_macro_mean": float(np.mean(elig_targets)),
                    "insufficient_history_exclusions_macro_mean": float(np.mean(hist_excl)),
                    "missing_target_exclusions_macro_mean": float(np.mean(miss_excl)),
                    "origin_coverage_fraction_macro_mean": float(
                        np.mean([e / c for e, c in zip(elig_origins, cand_origins)])
                    ),
                })

    return {
        "coverage_macro_summary": coverage_macro,
        "zero_frequency_shift_summary": p3b3["results"]["zero_frequency_shift_summary"],
        "regime_shift_scope_distinction": {
            "campus_wide_zero_shift_effect": (
                "All 29 core buildings experience a campus-wide drop from 381.02 valid zero hours per 1,000 "
                "observed hours in 2016 (median 384.34/1,000h, range [366.98, 384.79]) to 0.00/1,000h in H1 2017. "
                "This campus-wide shift depresses 2016 unconditional training means and shifts 2016 training score "
                "quantiles relative to H1 2017."
            ),
            "cross_building_variation_effect": (
                "Because 2016 valid-zero rates are nearly uniform across the 29 core buildings (20/29 buildings "
                "lie within 384.0 to 384.8 per 1,000h), building-to-building differences in 2016 zero frequency "
                "do not significantly explain cross-building differences in forecasting degradation (Pearson r = 0.2330, "
                "p = 0.2239; Spearman rho = 0.3286, p = 0.0818). Cross-building variation is driven by building-specific "
                "changes in positive active-load distributions and diurnal ramp profiles between 2016 and H1 2017."
            ),
        },
    }


def synthesize_forecasting(p3b1, p3b2):
    if p3b1["results"]["macro_summary"] != p3b2["results"]["baseline"]["macro_summary"]:
        raise ValueError("Phase 3B.1 and Phase 3B.2 baseline macro summaries do not match")

    base_rows = p3b2["results"]["baseline"]["rows"]
    cand_rows = p3b2["results"]["candidates"]["rows"]
    comp_rows = p3b2["results"]["comparisons"]
    if any("Quintin" in r["building_id"] for r in base_rows + cand_rows):
        raise ValueError("Sensitivity building Quintin must not appear in core forecasting rows")

    task_lookback_summary = []
    win_count_summary = []

    for task in TASKS:
        base_method = TASK_BASELINES[task]
        for lb in LOOKBACKS:
            b_sub = [r for r in base_rows if r["task"] == task and r["lookback_hours"] == lb]
            b_maes = [r["mae"] for r in b_sub]
            b_rmses = [r["rmse"] for r in b_sub]
            b_biases = [r["mean_error"] for r in b_sub]
            b_mases = [r["mase_training_daily_naive"] for r in b_sub]
            task_lookback_summary.append({
                "task": task,
                "lookback_hours": lb,
                "model": base_method,
                "model_role": "task_baseline",
                "building_count": len(b_sub),
                "mae_macro_mean": float(np.mean(b_maes)),
                "mae_median": float(np.median(b_maes)),
                "mae_range": [float(min(b_maes)), float(max(b_maes))],
                "rmse_macro_mean": float(np.mean(b_rmses)),
                "rmse_median": float(np.median(b_rmses)),
                "rmse_range": [float(min(b_rmses)), float(max(b_rmses))],
                "mean_error_macro_mean": float(np.mean(b_biases)),
                "mean_error_median": float(np.median(b_biases)),
                "mean_error_range": [float(min(b_biases)), float(max(b_biases))],
                "mase_macro_mean": float(np.mean(b_mases)),
                "mase_median": float(np.median(b_mases)),
                "mase_range": [float(min(b_mases)), float(max(b_mases))],
                "buildings_improved_mae": None,
                "buildings_improved_rmse": None,
                "buildings_improved_mase": None,
                "mae_delta_vs_baseline_macro_mean": 0.0,
                "mae_delta_vs_baseline_median": 0.0,
                "mase_delta_vs_baseline_macro_mean": 0.0,
                "mase_delta_vs_baseline_median": 0.0,
            })

            for model in CANDIDATE_MODELS:
                c_sub = [
                    r for r in cand_rows
                    if r["task"] == task and r["lookback_hours"] == lb and r["model"] == model
                ]
                cmp_sub = [
                    c for c in comp_rows
                    if c["task"] == task and c["lookback_hours"] == lb and c["model"] == model
                ]
                c_maes = [r["mae"] for r in c_sub]
                c_rmses = [r["rmse"] for r in c_sub]
                c_biases = [r["mean_error"] for r in c_sub]
                c_mases = [r["mase_training_daily_naive"] for r in c_sub]
                mae_deltas = [c["mae_delta"] for c in cmp_sub]
                mase_deltas = [c["mase_delta"] for c in cmp_sub]
                task_lookback_summary.append({
                    "task": task,
                    "lookback_hours": lb,
                    "model": model,
                    "model_role": "candidate_recursive",
                    "building_count": len(c_sub),
                    "mae_macro_mean": float(np.mean(c_maes)),
                    "mae_median": float(np.median(c_maes)),
                    "mae_range": [float(min(c_maes)), float(max(c_maes))],
                    "rmse_macro_mean": float(np.mean(c_rmses)),
                    "rmse_median": float(np.median(c_rmses)),
                    "rmse_range": [float(min(c_rmses)), float(max(c_rmses))],
                    "mean_error_macro_mean": float(np.mean(c_biases)),
                    "mean_error_median": float(np.median(c_biases)),
                    "mean_error_range": [float(min(c_biases)), float(max(c_biases))],
                    "mase_macro_mean": float(np.mean(c_mases)),
                    "mase_median": float(np.median(c_mases)),
                    "mase_range": [float(min(c_mases)), float(max(c_mases))],
                    "buildings_improved_mae": sum(1 for c in cmp_sub if c["mae_improved"]),
                    "buildings_improved_rmse": sum(1 for c in cmp_sub if c["rmse_delta"] < 0.0),
                    "buildings_improved_mase": sum(1 for c in cmp_sub if c["mase_delta"] < 0.0),
                    "mae_delta_vs_baseline_macro_mean": float(np.mean(mae_deltas)),
                    "mae_delta_vs_baseline_median": float(np.median(mae_deltas)),
                    "mase_delta_vs_baseline_macro_mean": float(np.mean(mase_deltas)),
                    "mase_delta_vs_baseline_median": float(np.median(mase_deltas)),
                })

            b_by_id = {r["building_id"]: r for r in b_sub}
            c_by_id_model = {
                (r["building_id"], r["model"]): r
                for r in cand_rows
                if r["task"] == task and r["lookback_hours"] == lb
            }
            mae_wins = {base_method: 0, "sarima": 0, "random_forest": 0, "xgboost": 0}
            rmse_wins = {base_method: 0, "sarima": 0, "random_forest": 0, "xgboost": 0}
            for b_id, b_row in sorted(b_by_id.items()):
                mae_scores = {base_method: b_row["mae"]}
                rmse_scores = {base_method: b_row["rmse"]}
                for m in CANDIDATE_MODELS:
                    mae_scores[m] = c_by_id_model[(b_id, m)]["mae"]
                    rmse_scores[m] = c_by_id_model[(b_id, m)]["rmse"]
                mae_wins[min(mae_scores, key=mae_scores.get)] += 1
                rmse_wins[min(rmse_scores, key=rmse_scores.get)] += 1
            win_count_summary.append({
                "task": task,
                "lookback_hours": lb,
                "building_count": len(b_by_id),
                "best_mae_model_counts": mae_wins,
                "best_rmse_model_counts": rmse_wins,
            })

    return {
        "multi_step_protocol": "frozen_origin_recursive",
        "mechanistic_and_configuration_diagnostics": {
            "sarima_non_seasonal_ar1_behavior": (
                "In phase3b2_candidate_experiment.json, SARIMA is configured with order=(1, 0, 0) and "
                "seasonal_order=(0, 0, 0, 0) with a constant intercept (trend='c') across all three tasks. "
                "As a non-seasonal AR(1), its multi-step recursive forecasts y_hat_{t+h|t} = mu_2016 + phi_1^h * (y_t - mu_2016) "
                "decay geometrically toward the unconditional 2016 training mean mu_2016 (which is depressed by the 38.10% "
                "2016 valid-zero regime), producing large negative biases at H=24 (-21.232 kWh) and H=168 (-33.498 kWh). "
                "Furthermore, SARIMA fitting does not use lookback_hours for feature construction; lookback_hours only changes "
                "the common-origin eligibility mask max(lookback, period), so weekly_profile (period=168) SARIMA results are "
                "identical across lookbacks 24h, 48h, and 168h (Macro MAE 36.566 kWh, MASE 4.929)."
            ),
            "xgboost_h1_shrinkage_and_recursive_compounding": (
                "XGBoost is configured with n_estimators=40 and learning_rate=0.05. Even at H=1 (hourly, where no recursive "
                "rollout occurs), XGBoost exhibits substantial negative bias (-7.671 kWh at lb=24h, -7.927 kWh at lb=48h, "
                "-7.965 kWh at lb=168h; MASE 1.609, 1.498, 1.482) because (1 - 0.05)^40 = 0.1285 of the initial global "
                "base_score = mean(y_train) (diluted by 38.10% 2016 zeros) remains unclosed on positive 2017 inputs. "
                "During 24-step and 168-step recursive rollouts, this 1-step underprediction compounds as underestimated "
                "predictions are fed back as lag features (Macro Bias -21.350 kWh at H=24, -28.332 kWh at H=168 for lb=168h)."
            ),
            "random_forest_tree_partitioning": (
                "Random Forest (n_estimators=20, max_depth=10) averages un-shrunk bootstrap regression trees that partition "
                "positive lag vectors away from zero lag vectors without global base_score shrinkage, achieving near-zero bias "
                "at H=1 (+0.374 kWh, MASE 0.937 at lb=168h) and moderate negative drift under 24-step (-2.827 kWh, MASE 1.909) "
                "and 168-step (-6.238 kWh, MASE 2.491) recursion."
            ),
            "strict_lag_less_than_lookback_selection": (
                "In phase3b_candidates.py, feature_lags(config, lookback) filters feature_lags_hours = [1, 2, 3, 6, 12, 24, 48, 168] "
                "using strict inequality (lag < lookback). Consequently, lookback=24h uses only lags [1, 2, 3, 6, 12] (excluding "
                "the diurnal lag_24), lookback=48h uses [1, 2, 3, 6, 12, 24] (adding lag_24 and reducing Random Forest daily_profile "
                "Macro MAE from 18.365 to 14.424 kWh and hourly Macro MAE from 7.492 to 6.772 kWh), and lookback=168h uses "
                "[1, 2, 3, 6, 12, 24, 48] (excluding lag_168)."
            ),
        },
        "task_lookback_summary": task_lookback_summary,
        "building_win_count_summary": win_count_summary,
    }


def synthesize_anomaly_detection(p3b3):
    core_rows = p3b3["results"]["rows"]
    if any("Quintin" in r["building_id"] for r in core_rows):
        raise ValueError("Sensitivity building Quintin must not appear in core anomaly rows")
    coinc_map = {
        (c["feature_mode"], c["model"], c["operating_point"]): c
        for c in p3b3["results"]["coincidence_summary"]
    }
    enriched_macro = []
    for m in p3b3["results"]["macro_summary"]:
        key = (m["feature_mode"], m["model"], m["operating_point"])
        sub = [
            r for r in core_rows
            if (r["feature_mode"], r["model"], r["operating_point"]) == key
        ]
        tr_rates = [r["training_flagged_per_1000_hours"] for r in sub]
        ties = [r["training_tied_at_threshold_hours"] for r in sub]
        coinc = coinc_map[key]
        enriched_macro.append({
            **m,
            "nominal_training_flagged_per_1000_hours": float(sub[0]["nominal_training_flagged_per_1000_hours"]),
            "training_flagged_per_1000_macro_mean": float(np.mean(tr_rates)),
            "training_flagged_per_1000_median": float(np.median(tr_rates)),
            "training_flagged_per_1000_range": [float(min(tr_rates)), float(max(tr_rates))],
            "buildings_with_threshold_ties_count": sum(1 for t in ties if t > 0),
            "max_training_tied_at_threshold_hours": int(max(ties)),
            "hours_with_any_building_flagged": coinc["hours_with_any_building_flagged"],
            "hours_with_multi_building_coincidence": coinc["hours_with_multi_building_coincidence"],
            "hours_with_5plus_buildings_flagged": coinc["hours_with_5plus_buildings_flagged"],
            "max_simultaneous_buildings_flagged": coinc["max_simultaneous_buildings_flagged"],
        })

    return {
        "report_version": p3b3["report_version"],
        "feature_space_semantics": p3b3["feature_space_semantics"],
        "operating_point_semantics": p3b3["operating_point_semantics"],
        "macro_and_coincidence_summary": enriched_macro,
        "coincidence_details": p3b3["results"]["coincidence_summary"],
        "sensitivity_quintin": p3b3["results"]["sensitivity_quintin"],
    }


def _ols_leverage_and_cooks_d(x_arr, y_arr):
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    mse = float(np.sum(residuals ** 2) / (len(x) - 2))
    hat_diag = np.diag(design @ np.linalg.inv(design.T @ design) @ design.T)
    cooks_d = (residuals ** 2 / (2.0 * mse)) * (hat_diag / ((1.0 - hat_diag) ** 2))
    return hat_diag, cooks_d


def synthesize_cross_building_heterogeneity(p3b1, p3b2, p3b3):
    base_168 = {
        (r["building_id"], r["task"]): r
        for r in p3b2["results"]["baseline"]["rows"]
        if r["lookback_hours"] == 168
    }
    cand_168 = {
        (r["building_id"], r["task"], r["model"]): r
        for r in p3b2["results"]["candidates"]["rows"]
        if r["lookback_hours"] == 168
    }
    comp_168 = {
        (c["building_id"], c["task"], c["model"]): c
        for c in p3b2["results"]["comparisons"]
        if c["lookback_hours"] == 168
    }
    anom_1pct = {
        (r["building_id"], r["feature_mode"], r["model"]): r
        for r in p3b3["results"]["rows"]
        if r["operating_point"] == "1.0pct"
    }
    zero_shift = {
        r["building_id"]: r
        for r in p3b3["results"]["zero_frequency_shift_summary"]["core_buildings"]
    }
    buildings = sorted(zero_shift)
    if len(buildings) != 29 or any("Quintin" in b for b in buildings):
        raise ValueError("Cross-building core synthesis must contain exactly 29 non-Quintin buildings")

    building_profiles = []
    for b_id in buildings:
        zs = zero_shift[b_id]
        h_base = base_168[(b_id, "hourly")]
        d_base = base_168[(b_id, "daily_profile")]
        w_base = base_168[(b_id, "weekly_profile")]

        def _best_for_task(task):
            b_row = base_168[(b_id, task)]
            options = {b_row["method"]: b_row}
            for m in CANDIDATE_MODELS:
                options[m] = cand_168[(b_id, task, m)]
            winner = min(options, key=lambda k: options[k]["mae"])
            return winner, options[winner]["mae"], options[winner]["mase_training_daily_naive"]

        h_win, h_best_mae, h_best_mase = _best_for_task("hourly")
        d_win, d_best_mae, d_best_mase = _best_for_task("daily_profile")
        w_win, w_best_mae, w_best_mase = _best_for_task("weekly_profile")

        rf_h_cmp = comp_168[(b_id, "hourly", "random_forest")]
        rf_d_cmp = comp_168[(b_id, "daily_profile", "random_forest")]
        rf_w_cmp = comp_168[(b_id, "weekly_profile", "random_forest")]

        building_profiles.append({
            "building_id": b_id,
            "mase_scaling_denominator_kwh": float(d_base["mase_scaling_denominator"]),
            "train_valid_zero_per_1000_hours": float(zs["train_valid_zero_per_1000_hours"]),
            "validation_valid_zero_per_1000_hours": float(zs["validation_valid_zero_per_1000_hours"]),
            "hourly_baseline_mae": float(h_base["mae"]),
            "hourly_baseline_mase": float(h_base["mase_training_daily_naive"]),
            "hourly_best_model": h_win,
            "hourly_best_mae": float(h_best_mae),
            "hourly_best_mase": float(h_best_mase),
            "hourly_rf_mae_delta": float(rf_h_cmp["mae_delta"]),
            "hourly_rf_mase_delta": float(rf_h_cmp["mase_delta"]),
            "daily_baseline_mae": float(d_base["mae"]),
            "daily_baseline_mase": float(d_base["mase_training_daily_naive"]),
            "daily_best_model": d_win,
            "daily_best_mae": float(d_best_mae),
            "daily_best_mase": float(d_best_mase),
            "daily_rf_mae_delta": float(rf_d_cmp["mae_delta"]),
            "daily_rf_mase_delta": float(rf_d_cmp["mase_delta"]),
            "daily_rf_mean_error": float(rf_d_cmp["candidate_mean_error"]),
            "weekly_baseline_mae": float(w_base["mae"]),
            "weekly_baseline_mase": float(w_base["mase_training_daily_naive"]),
            "weekly_best_model": w_win,
            "weekly_best_mae": float(w_best_mae),
            "weekly_best_mase": float(w_best_mase),
            "weekly_rf_mae_delta": float(rf_w_cmp["mae_delta"]),
            "weekly_rf_mase_delta": float(rf_w_cmp["mase_delta"]),
            "anom_1pct_raw_zscore_per_1000h": float(
                anom_1pct[(b_id, "raw_consumption", "statistical_zscore")]["validation_flagged_per_1000_hours"]
            ),
            "anom_1pct_raw_iforest_per_1000h": float(
                anom_1pct[(b_id, "raw_consumption", "isolation_forest")]["validation_flagged_per_1000_hours"]
            ),
            "anom_1pct_raw_lof_per_1000h": float(
                anom_1pct[(b_id, "raw_consumption", "lof_novelty")]["validation_flagged_per_1000_hours"]
            ),
            "anom_1pct_seasonal_zscore_per_1000h": float(
                anom_1pct[(b_id, "seasonal_deviation", "statistical_zscore")]["validation_flagged_per_1000_hours"]
            ),
            "anom_1pct_seasonal_iforest_per_1000h": float(
                anom_1pct[(b_id, "seasonal_deviation", "isolation_forest")]["validation_flagged_per_1000_hours"]
            ),
            "anom_1pct_seasonal_lof_per_1000h": float(
                anom_1pct[(b_id, "seasonal_deviation", "lof_novelty")]["validation_flagged_per_1000_hours"]
            ),
        })

    b_ids = [p["building_id"] for p in building_profiles]
    rf_d_mase_deltas = np.asarray([p["daily_rf_mase_delta"] for p in building_profiles], dtype=float)
    rf_d_abs_biases = np.asarray([abs(p["daily_rf_mean_error"]) for p in building_profiles], dtype=float)
    rf_d_mae_deltas = np.asarray([p["daily_rf_mae_delta"] for p in building_profiles], dtype=float)
    lof_seas_rates = np.asarray([p["anom_1pct_seasonal_lof_per_1000h"] for p in building_profiles], dtype=float)
    zero_tr_rates = np.asarray([p["train_valid_zero_per_1000_hours"] for p in building_profiles], dtype=float)

    r_full = float(np.corrcoef(rf_d_mase_deltas, lof_seas_rates)[0, 1])
    _, p_full = stats.pearsonr(rf_d_mase_deltas, lof_seas_rates)
    rho_full, prho_full = stats.spearmanr(rf_d_mase_deltas, lof_seas_rates)
    tau_full, ptau_full = stats.kendalltau(rf_d_mase_deltas, lof_seas_rates)

    vincent_idx = b_ids.index("Panther_education_Vincent")
    mask_no_vincent = np.arange(len(b_ids)) != vincent_idx
    r_no_v, p_no_v = stats.pearsonr(rf_d_mase_deltas[mask_no_vincent], lof_seas_rates[mask_no_vincent])
    rho_no_v, prho_no_v = stats.spearmanr(rf_d_mase_deltas[mask_no_vincent], lof_seas_rates[mask_no_vincent])
    hat_diag, cooks_d = _ols_leverage_and_cooks_d(rf_d_mase_deltas, lof_seas_rates)

    rf_win_mask = rf_d_mase_deltas < 0.0
    r_zero, p_zero = stats.pearsonr(zero_tr_rates, rf_d_mase_deltas)
    rho_zero, prho_zero = stats.spearmanr(zero_tr_rates, rf_d_mase_deltas)

    r_bias = float(np.corrcoef(rf_d_abs_biases, rf_d_mae_deltas)[0, 1])
    _, p_bias = stats.pearsonr(rf_d_abs_biases, rf_d_mae_deltas)
    rho_bias, prho_bias = stats.spearmanr(rf_d_abs_biases, rf_d_mae_deltas)

    return {
        "reference_lookback_hours": 168,
        "reference_anomaly_operating_point": "1.0pct",
        "building_count": len(building_profiles),
        "cross_phase_correlations": {
            "status": "exploratory_outlier_sensitive_diagnostic",
            "is_robust_general_relationship": False,
            "causal_claim_permitted": False,
            "pearson_rf_daily_mase_delta_vs_seasonal_lof_rate": r_full,
            "pearson_rf_daily_abs_bias_vs_mae_delta": r_bias,
            "rf_daily_mase_delta_vs_seasonal_lof_rate": {
                "n_buildings": int(len(building_profiles)),
                "pearson_r": r_full,
                "pearson_p_value": float(p_full),
                "spearman_rho": float(rho_full),
                "spearman_p_value": float(prho_full),
                "kendall_tau": float(tau_full),
                "kendall_p_value": float(ptau_full),
                "influential_outlier_building_id": "Panther_education_Vincent",
                "influential_outlier_mase_delta": float(rf_d_mase_deltas[vincent_idx]),
                "influential_outlier_seasonal_lof_per_1000h": float(lof_seas_rates[vincent_idx]),
                "influential_outlier_leverage_h": float(hat_diag[vincent_idx]),
                "influential_outlier_cooks_d": float(cooks_d[vincent_idx]),
                "excluding_vincent_n": int(mask_no_vincent.sum()),
                "excluding_vincent_pearson_r": float(r_no_v),
                "excluding_vincent_pearson_p_value": float(p_no_v),
                "excluding_vincent_spearman_rho": float(rho_no_v),
                "excluding_vincent_spearman_p_value": float(prho_no_v),
                "rf_winning_buildings_count": int(rf_win_mask.sum()),
                "rf_winning_buildings_seasonal_lof_median_per_1000h": float(
                    np.median(lof_seas_rates[rf_win_mask])
                ),
                "baseline_winning_buildings_count": int((~rf_win_mask).sum()),
                "baseline_winning_buildings_seasonal_lof_median_per_1000h": float(
                    np.median(lof_seas_rates[~rf_win_mask])
                ),
                "interpretation": (
                    "Exploratory and non-robust: the apparent N=29 positive linear correlation "
                    "(Pearson r = 0.6906, p = 3.38e-05) is dominated by one extreme high-leverage "
                    "building (Panther_education_Vincent, daily_rf_mase_delta = +3.4928, seasonal LOF = "
                    "306.69/1,000h, leverage h = 0.8507, Cook's D = 1.2558). Across all 29 buildings, "
                    "non-parametric rank correlation is negative and statistically indistinguishable from "
                    "zero (Spearman rho = -0.1466, p = 0.4480; Kendall tau = -0.1284, p = 0.3292), and "
                    "excluding Panther_education_Vincent (N=28) reduces Pearson r to 0.2518 (p = 0.1962, "
                    "non-significant). No general cross-building linear relationship or causal coupling "
                    "is supported."
                ),
            },
            "train_valid_zero_rate_vs_rf_daily_mase_delta": {
                "n_buildings": int(len(building_profiles)),
                "pearson_r": float(r_zero),
                "pearson_p_value": float(p_zero),
                "spearman_rho": float(rho_zero),
                "spearman_p_value": float(prho_zero),
                "interpretation": (
                    "2016 valid-zero rates are nearly uniform across the 29 core buildings (366.98 to "
                    "384.79 per 1,000h; 20/29 buildings within 384.0 to 384.8 per 1,000h) and do not "
                    "significantly explain cross-building differences in daily Random Forest MASE "
                    "degradation (Pearson r = 0.2330, p = 0.2239; Spearman rho = 0.3286, p = 0.0818)."
                ),
            },
            "rf_daily_abs_bias_vs_mae_delta": {
                "n_buildings": int(len(building_profiles)),
                "pearson_r": r_bias,
                "pearson_p_value": float(p_bias),
                "spearman_rho": float(rho_bias),
                "spearman_p_value": float(prho_bias),
                "interpretation": (
                    "Descriptive error-decomposition diagnostic: per-building |candidate_mean_error| "
                    "associates positively with daily_profile MAE delta (Pearson r = 0.5802, Spearman "
                    "rho = 0.7537), reflecting the mathematical bound |mean_error| <= MAE and "
                    "building-specific positive active-load level shifts between 2016 and H1 2017."
                ),
            },
        },
        "building_profiles": building_profiles,
    }


def build_phase3c_synthesis(root):
    reports = load_approved_reports(root)
    p3a = reports["phase3a"]
    p3b1 = reports["phase3b1"]
    p3b2 = reports["phase3b2"]
    p3b3 = reports["phase3b3"]

    coverage_and_regime = synthesize_coverage_and_regime_shift(p3a, p3b3)
    forecasting = synthesize_forecasting(p3b1, p3b2)
    anomaly = synthesize_anomaly_detection(p3b3)
    cross_building = synthesize_cross_building_heterogeneity(p3b1, p3b2, p3b3)

    return {
        "phase": "3C",
        "report_version": "3C-remediated-v1",
        "title": "Integrated Evaluation & Research Synthesis",
        "status": "validation_synthesis_only",
        "test_evaluation": "not_run",
        "ground_truth_anomaly_labels": "none",
        "execution": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "provenance": {
            "protected_dataset_checksums": verify_protected_checksums(root),
            "source_report_checksums": verify_source_report_checksums(root),
            "prior_phase3c_report_checksums": verify_prior_phase3c_checksums(root),
        },
        "coverage_and_regime_synthesis": coverage_and_regime,
        "forecasting_synthesis": forecasting,
        "anomaly_detection_synthesis": anomaly,
        "cross_building_heterogeneity_synthesis": cross_building,
        "model_selection_conclusions": {
            "hourly_1h_ahead": {
                "recommended_primary": "random_forest (lookback=168h or 48h)",
                "recommended_reference": "persistence",
                "evidence": (
                    "At lookback=168h, Random Forest achieves the lowest macro mean MAE (6.731 kWh vs "
                    "7.100 kWh for persistence), lowest macro mean RMSE (10.813 kWh vs 12.566 kWh), "
                    "and lowest macro mean MASE (0.937 vs 1.017), improving MAE on 18/29 buildings and "
                    "RMSE on 19/29 buildings. SARIMA also improves MAE slightly on 19/29 buildings "
                    "(macro MAE 7.081 kWh, MASE 1.013). By contrast, XGBoost (n_estimators=40, "
                    "learning_rate=0.05) already underpredicts at H=1 without recursion (macro bias "
                    "-7.965 kWh, MASE 1.482) because 40 boosting steps at lr=0.05 do not overcome the "
                    "zero-diluted 2016 global base_score on positive 2017 inputs."
                ),
            },
            "daily_profile_24h_ahead": {
                "recommended_primary": "daily_seasonal_naive (period=24h)",
                "recommended_ml_comparator": "random_forest (lookback=168h)",
                "rejected_for_multistep_rollout": ["sarima", "xgboost"],
                "evidence": (
                    "Daily seasonal naive achieves the lowest macro mean MAE (12.158 kWh), lowest "
                    "median MAE (10.551 kWh), lowest macro mean RMSE (19.841 kWh), near-zero bias "
                    "(+0.009 kWh), and lowest macro mean MASE (1.644), winning 19/29 buildings on MAE. "
                    "Random Forest (lookback=168h) is the strongest ML candidate (macro MAE 14.557 kWh, "
                    "MASE 1.909), winning 10/29 buildings on MAE and 14/29 on RMSE where 2016-to-2017 "
                    "positive active-load distributions remain aligned. SARIMA (configured as a "
                    "non-seasonal AR(1) (1,0,0)x(0,0,0,0) with intercept; bias -21.232 kWh, MASE 3.801) "
                    "mean-reverts geometrically toward the unconditional 2016 mean, while XGBoost "
                    "(bias -21.350 kWh, MASE 3.403) compounds its H=1 shrinkage underprediction across "
                    "the 24 recursive steps."
                ),
            },
            "weekly_profile_168h_ahead": {
                "recommended_primary": "weekly_seasonal_naive (period=168h)",
                "recommended_ml_comparator": "random_forest (lookback=168h)",
                "rejected_for_multistep_rollout": ["sarima", "xgboost"],
                "evidence": (
                    "Weekly seasonal naive achieves the lowest macro mean MAE (14.135 kWh), lowest "
                    "median MAE (12.504 kWh), lowest macro mean RMSE (22.337 kWh), and lowest macro "
                    "mean MASE (1.937), winning 20/29 buildings on MAE. Random Forest (lookback=168h) "
                    "wins 7/29 buildings on MAE (11/29 on RMSE) with macro MASE 2.491, whereas non-seasonal "
                    "AR(1) SARIMA (MASE 4.929, bias -33.498 kWh, identical across lookbacks 24h/48h/168h "
                    "because max(lookback, 168) == 168) and XGBoost (MASE 4.355, bias -28.332 kWh) "
                    "suffer severe multi-step underprediction."
                ),
            },
            "unsupervised_anomaly_detection": {
                "recommended_primary_mode": "seasonal_deviation (3D multivariate: isolation_forest and lof_novelty)",
                "recommended_univariate_reference": "statistical_zscore (1D slot_zscore and raw_consumption)",
                "evidence": (
                    "In 3D seasonal_deviation mode, Isolation Forest (1.0pct median validation rate "
                    "15.52/1,000h, macro mean 21.82/1,000h, conditional mean event duration 1.41h) and "
                    "full-multiplicity LOF novelty (1.0pct median 19.69/1,000h, macro mean 46.00/1,000h, "
                    "conditional mean event duration 1.84h) capture joint hour-of-week and ramp "
                    "departures across all 29 buildings without flagging any valid zero readings "
                    "(validation_flagged_zero_hours == 0)."
                ),
            },
        },
        "scientific_limitations_and_threats_to_validity": [
            "No confirmed ground-truth anomaly labels exist; all anomaly metrics describe unlabeled statistical or geometric departures from 2016 training references, and supervised metrics (precision, recall, F1, ROC-AUC, accuracy) are strictly prohibited.",
            "Campus-wide vs. building-specific regime shifts: all 29 core buildings average 381.02 valid zero hours per 1,000 observed hours in 2016 (median 384.34/1,000h, range [366.98, 384.79]) versus 0.00/1,000h in H1 2017. While this campus-wide shift depresses 2016 unconditional means and shifts training quantiles, 2016 zero rates are nearly uniform across the 29 buildings and do not significantly explain cross-building differences in forecasting degradation (Pearson r = 0.2330, p = 0.2239; Spearman rho = 0.3286, p = 0.0818), which instead reflect building-specific changes in positive active-load distributions.",
            "Model-configuration vs. recursion effects: multi-step forecasting errors in Phase 3B.2 reflect both recursive error propagation and specific model configurations—namely SARIMA's non-seasonal AR(1) (1,0,0)x(0,0,0,0) specification (which decays geometrically to the 2016 unconditional mean), XGBoost's under-converged boosting schedule (n_estimators=40, learning_rate=0.05, which already underpredicts by -7.965 kWh at H=1 without recursion), and strict lag < lookback feature filtering (which excludes lag_24 at lookback=24h and lag_168 at lookback=168h).",
            "Outlier sensitivity of cross-phase correlation: the apparent N=29 positive Pearson correlation between Random Forest daily_profile MASE delta and seasonal LOF alert rate (r = 0.6906) is an exploratory, non-robust artifact dominated by a single extreme high-leverage building (Panther_education_Vincent, leverage h = 0.8507, Cook's D = 1.2558; excluding Vincent gives N=28 Pearson r = 0.2518, p = 0.1962, and N=29 Spearman rho = -0.1466, p = 0.4480).",
            "Feature-space dimensionality in seasonal_deviation anomaly detection intentionally compares a 1D univariate statistical baseline (|slot_zscore|) against 3D multivariate detectors ([slot_zscore, lag_1h_scaled_diff, lag_24h_scaled_diff]) over identical eligible timestamps.",
            "Weather-exogenous forecasting was specified for complete-case evaluation in Phase 2.7/3A, but Phases 3B.1 and 3B.2 evaluated electricity-only baselines and candidates; weather-augmented forecasting remains an unexecuted sensitivity extension.",
            "The held-out test partition (2017-07-01 00:00:00 through 2017-12-31 23:00:00) remains untouched (not_run); findings reflect H1 2017 validation performance on 29 UCF Panther education buildings and do not support broad multi-year or cross-campus generalization claims.",
        ],
    }


def render_markdown_report(syn):
    cov = syn["coverage_and_regime_synthesis"]["coverage_macro_summary"]
    zs = syn["coverage_and_regime_synthesis"]["zero_frequency_shift_summary"]
    fc_rows = syn["forecasting_synthesis"]["task_lookback_summary"]
    fc_wins = {
        (w["task"], w["lookback_hours"]): w
        for w in syn["forecasting_synthesis"]["building_win_count_summary"]
    }
    anom_rows = syn["anomaly_detection_synthesis"]["macro_and_coincidence_summary"]
    quintin_anom = syn["anomaly_detection_synthesis"]["sensitivity_quintin"]["rows"]
    cb = syn["cross_building_heterogeneity_synthesis"]
    corr = cb["cross_phase_correlations"]
    c_lof = corr["rf_daily_mase_delta_vs_seasonal_lof_rate"]
    c_zero = corr["train_valid_zero_rate_vs_rf_daily_mase_delta"]
    c_bias = corr["rf_daily_abs_bias_vs_mae_delta"]

    lines = [
        "# Phase 3C — Integrated Evaluation & Research Synthesis (Remediated)",
        "",
        f"**Report Version:** `{syn['report_version']}` | **Report Status:** `{syn['status']}` | **Held-Out Test Partition:** `{syn['test_evaluation']}` | **Ground-Truth Anomaly Labels:** `{syn['ground_truth_anomaly_labels']}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This document synthesizes the completed and independently audited validation experiments across **Phase 3A** (canonical validation and lookback coverage), **Phase 3B.1** (seasonal-naive and persistence baselines), **Phase 3B.2** (frozen-origin recursive multi-step candidate forecasting: SARIMA, Random Forest, XGBoost), and **Phase 3B.3** (remediated training-calibrated unsupervised anomaly detection: Statistical Z-Score, Isolation Forest, and full-empirical-multiplicity LOF novelty detection) for the 29 core UCF Panther education buildings (`2016` training, `H1 2017` validation) and the isolated sensitivity population (`Panther_education_Quintin`).",
        "",
        "### Core Scientific Findings",
        "1. **Horizon-Dependent Forecasting Performance (No Universal Winner):**",
        "   - **1-Hour Ahead (`hourly`, secondary task):** `random_forest` (`lookback=48h` and `168h`) and `sarima` outperform the `persistence` baseline. At `lookback=168h`, `random_forest` achieves the lowest macro mean MAE (**`6.731 kWh`** vs. `7.100 kWh`), lowest macro mean RMSE (**`10.813 kWh`** vs. `12.566 kWh`), and lowest macro mean MASE (**`0.937`** vs. `1.017`), improving MAE on **`18/29`** buildings (`19/29` on RMSE).",
        "   - **24-Hour Ahead (`daily_profile`, primary task):** `daily_seasonal_naive` outperforms all frozen-origin recursive candidates in aggregate macro metrics (Macro MAE **`12.158 kWh`**, Median MAE **`10.551 kWh`**, Macro RMSE **`19.841 kWh`**, Bias **`+0.009 kWh`**, Macro MASE **`1.644`** at `lookback=168h`), winning **`19/29`** buildings on MAE. `random_forest` (`lookback=168h`) is the strongest ML candidate (Macro MAE **`14.557 kWh`**, Macro MASE **`1.909`**), winning **`10/29`** buildings on MAE and **`14/29`** on RMSE where 2016-to-2017 positive active-load distributions remain aligned.",
        "   - **168-Hour Ahead (`weekly_profile`, exploratory task):** `weekly_seasonal_naive` leads in macro metrics (Macro MAE **`14.135 kWh`**, Median MAE **`12.504 kWh`**, Macro RMSE **`22.337 kWh`**, Macro MASE **`1.937`**), winning **`20/29`** buildings on MAE (`random_forest` wins `7/29`, `sarima` `1/29`, `xgboost` `1/29`).",
        "2. **Model-Configuration Mechanisms vs. Recursive Rollout & Regime Shift:**",
        "   - Multi-step forecasting errors in Phase 3B.2 cannot be attributed solely to the 2016 valid-zero regime shift; they reflect specific model-configuration mechanisms interacting with recursion:",
        "     - **SARIMA Non-Seasonal AR(1) Specification:** Configured as `(1,0,0)x(0,0,0,0)` with constant intercept (`trend='c'`), SARIMA lacks seasonal terms ($s=24$ or $s=168$), so its multi-step forecasts $\\hat{y}_{t+h|t} = \\mu_{2016} + \\phi_1^h(y_t - \\mu_{2016})$ mean-revert geometrically toward the zero-diluted 2016 unconditional mean $\\mu_{2016}$ (`daily_profile` Macro Bias `-21.232 kWh`, `weekly_profile` `-33.498 kWh`).",
        "     - **XGBoost Under-Convergence Already Present at $H=1$:** With `n_estimators=40` and `learning_rate=0.05`, $(1 - 0.05)^{40} = 0.1285$ of the gap from the global `base_score = mean(y_train)` remains unclosed even on strictly positive 2017 inputs, causing severe underprediction at $H=1$ without recursion (`hourly` Macro Bias `-7.965 kWh`, MASE `1.482` at `lb=168h`) that compounds during 24-step (`-21.350 kWh`) and 168-step (`-28.332 kWh`) recursive rollouts.",
        "     - **Strict `lag < lookback` Feature Filtering:** Because `feature_lags` uses strict inequality (`lag < lookback`) over `[1, 2, 3, 6, 12, 24, 48, 168]`, `lookback=24h` excludes `lag_24` (`[1, 2, 3, 6, 12]`), `lookback=48h` introduces `lag_24` (cutting `random_forest` `daily_profile` Macro MAE from `18.365` to `14.424 kWh`), and `lookback=168h` uses `[1, 2, 3, 6, 12, 24, 48]` (excluding `lag_168`).",
        "3. **Campus-Wide Valid-Zero Shift vs. Building-Specific Positive Load Shifts:**",
        f"   - Across all 29 core buildings, valid `0.0 kWh` readings drop from a campus-wide mean of **`{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}` per 1,000 observed hours** (`38.10%`, median `{zs['core_train_valid_zero_per_1000_median']:.2f}/1,000h`, range `[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]`) in 2016 (`train`) to **`{zs['core_validation_valid_zero_per_1000_macro_mean']:.2f}` per 1,000 observed hours** in H1 2017 (`validation`).",
        f"   - Because 2016 zero rates are nearly uniform across the 29 buildings (`20/29` within `384.0–384.8/1,000h`), 2016 zero frequency does **not** significantly explain cross-building differences in forecasting degradation (Pearson $r = {c_zero['pearson_r']:.4f}$, $p = {c_zero['pearson_p_value']:.4f}$; Spearman $\\rho = {c_zero['spearman_rho']:.4f}$, $p = {c_zero['spearman_p_value']:.4f}$). Cross-building variation is driven by building-specific shifts in positive active-load distributions.",
        "4. **Unsupervised Anomaly Detection & Outlier-Sensitive Exploratory Diagnostics:**",
        "   - Remediated full-empirical-multiplicity `lof_novelty` and `isolation_forest` preserve all valid zeros as inliers (**`validation_flagged_zero_hours == 0`** across all 522 core rows and 18 `Quintin` rows) and exhibit strictly monotonic threshold operating points (`0.5pct > 1.0pct > 2.0pct` in `29/29` buildings).",
        f"   - Cross-phase correlation between `random_forest` `daily_profile` `mase_delta` (`lb=168h`) and `seasonal_deviation` `lof_novelty` validation alert rate (`1.0pct`) is **not** a robust general relationship: while $N=29$ Pearson $r = {c_lof['pearson_r']:.4f}$, it is dominated by a single extreme high-leverage outlier (`Panther_education_Vincent`, leverage $h = {c_lof['influential_outlier_leverage_h']:.4f}$, Cook's $D = {c_lof['influential_outlier_cooks_d']:.4f}$). Across all 29 buildings, Spearman rank correlation is negative and null ($\\rho = {c_lof['spearman_rho']:.4f}$, $p = {c_lof['spearman_p_value']:.4f}$), and excluding `Panther_education_Vincent` ($N=28$) reduces Pearson $r$ to `{c_lof['excluding_vincent_pearson_r']:.4f}` ($p = {c_lof['excluding_vincent_pearson_p_value']:.4f}$).",
        "",
        "---",
        "",
        "## 2. Coverage Accounting & Valid-Zero Regime Shift (Phase 3A & Phase 3B.3)",
        "",
        "### Table 1: Chronological Partition Coverage & Common-Lookback Eligibility (29 Core Buildings)",
        "",
        "| Partition | Task ($H$) | Lookback (h) | Candidate Origins / Bldg | Eligible Origins / Bldg (Mean) | Eligible Target Hours / Bldg | Insufficient History Exclusions | Missing Target Exclusions | Origin Coverage (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in cov:
        lines.append(
            f"| `{r['partition']}` | `{r['task']}` ({r['horizon_hours']}h) | `{r['lookback_hours']}` | "
            f"{r['candidate_origins_per_building']} | {r['eligible_origins_macro_mean']:.1f} | "
            f"{r['eligible_target_hours_macro_mean']:.1f} | {r['insufficient_history_exclusions_macro_mean']:.1f} | "
            f"{r['missing_target_exclusions_macro_mean']:.1f} | {100.0 * r['origin_coverage_fraction_macro_mean']:.2f}% |"
        )

    q_zs = zs["sensitivity_quintin"]
    lines.extend([
        "",
        "### Table 2: Training (2016) vs. Validation (H1 2017) Valid-Zero Frequency Shift",
        "",
        "| Population | Buildings | 2016 Train Observed Hours | 2016 Valid Zero / 1,000h (Mean / Median / Range) | H1 2017 Val Observed Hours | H1 2017 Valid Zero / 1,000h | Shift $\\Delta$ (Val $-$ Train / 1,000h) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Core Panther Education** | 29 | 8,784 | `{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}` / `{zs['core_train_valid_zero_per_1000_median']:.2f}` / `[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]` | 4,332 | `0.00` | **`{zs['core_delta_per_1000_macro_mean']:.2f}`** |",
        f"| **Sensitivity (`Quintin`)** | 1 | {q_zs['train_observed_hours']:,} | `{q_zs['train_valid_zero_per_1000_hours']:.2f}` (`{q_zs['train_valid_zero_hours']}` zero hours) | {q_zs['validation_observed_hours']:,} | `0.00` | **`{q_zs['valid_zero_per_1000_delta_validation_minus_train']:.2f}`** |",
        "",
        "### Zero-Regime Interpretation: Campus-Wide Shift vs. Cross-Building Active-Load Variation",
        f"- **Campus-Wide Effect:** All 29 core buildings experience the shift from ~38.1% valid zeros in 2016 (`{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}/1,000h`) to `0.00/1,000h` in H1 2017. This campus-wide regime change depresses 2016 unconditional training means ($\\mu_{{2016}}$) and lowers 2016 training score quantiles relative to active H1 2017 operations.",
        f"- **Cross-Building Variation:** Because 2016 valid-zero rates are tightly clustered across the 29 buildings (`[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]` per 1,000h; `20/29` buildings between `384.0` and `384.8/1,000h`), 2016 zero frequency does **not** significantly explain cross-building differences in daily Random Forest MASE degradation (Pearson $r = {c_zero['pearson_r']:.4f}$, $p = {c_zero['pearson_p_value']:.4f}$; Spearman $\\rho = {c_zero['spearman_rho']:.4f}$, $p = {c_zero['spearman_p_value']:.4f}$). Instead, building-to-building differences in forecasting accuracy and anomaly alert rates reflect building-specific changes in **positive active-load distributions** and diurnal ramp amplitude between 2016 and H1 2017.",
        "",
        "---",
        "",
        "## 3. Multi-Horizon Forecasting Comparison (Phases 3B.1 & 3B.2)",
        "",
        "### Table 3: Complete Forecasting Macro Synthesis Across Tasks and Lookbacks (29 Core Buildings)",
        "",
        "| Task | Lookback (h) | Model | Role | Macro MAE (kWh) | Median MAE (kWh) | MAE Range (kWh) | Macro RMSE (kWh) | Macro Bias (kWh) | Macro MASE | Bldgs Improved vs. Baseline (MAE / RMSE / MASE) |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in fc_rows:
        imp_str = (
            "—"
            if r["buildings_improved_mae"] is None
            else f"`{r['buildings_improved_mae']}/29` / `{r['buildings_improved_rmse']}/29` / `{r['buildings_improved_mase']}/29`"
        )
        lines.append(
            f"| `{r['task']}` | `{r['lookback_hours']}` | `{r['model']}` | `{r['model_role']}` | "
            f"`{r['mae_macro_mean']:.3f}` | `{r['mae_median']:.3f}` | "
            f"`[{r['mae_range'][0]:.2f}, {r['mae_range'][1]:.2f}]` | `{r['rmse_macro_mean']:.3f}` | "
            f"`{r['mean_error_macro_mean']:+.3f}` | `{r['mase_macro_mean']:.3f}` | {imp_str} |"
        )

    lines.extend([
        "",
        "### Table 4: Best-Performing Model Counts Across the 29 Core Buildings by Task and Lookback",
        "",
        "| Task | Lookback (h) | Best MAE Model Counts (29 Buildings) | Best RMSE Model Counts (29 Buildings) |",
        "| :--- | :---: | :--- | :--- |",
    ])
    for (task, lb), w in sorted(fc_wins.items(), key=lambda x: (TASKS.index(x[0][0]), x[0][1])):
        ordered_models = (TASK_BASELINES[task],) + CANDIDATE_MODELS
        mae_s = ", ".join(
            f"`{k}`: {w['best_mae_model_counts'][k]}"
            for k in ordered_models
            if w["best_mae_model_counts"].get(k, 0) > 0
        )
        rmse_s = ", ".join(
            f"`{k}`: {w['best_rmse_model_counts'][k]}"
            for k in ordered_models
            if w["best_rmse_model_counts"].get(k, 0) > 0
        )
        lines.append(f"| `{task}` | `{lb}` | {mae_s} | {rmse_s} |")

    lines.extend([
        "",
        "### Mechanistic Interpretation of Candidate Model & Lookback Behavior",
        "1. **SARIMA Non-Seasonal AR(1) Specification (`(1,0,0)x(0,0,0,0)` with `trend='c'`):**",
        "   - In `phase3b2_candidate_experiment.json`, SARIMA is configured with `order = [1, 0, 0]` and `seasonal_order = [0, 0, 0, 0]` across all three tasks (`hourly`, `daily_profile`, `weekly_profile`). Because it has no seasonal terms ($s=24$ or $s=168$), its multi-step recursive predictions $\\hat{y}_{t+h|t} = \\mu_{2016} + \\phi_1^h(y_t - \\mu_{2016})$ decay geometrically toward the unconditional 2016 training mean $\\mu_{2016}$ (which is depressed by the 38.10% 2016 zero regime), causing severe multi-step underprediction (`daily_profile` Macro Bias `-21.232 kWh`, `weekly_profile` `-33.498 kWh` at `lb=168h`).",
        "   - Furthermore, `_fit_sarima` fits on the full 2016 series without using `lookback_hours` for feature selection; `lookback_hours` only modifies the common-origin eligibility mask `max(lookback, period)`. Consequently, for `weekly_profile` (`period = 168`, where `max(lookback, 168) == 168` for all three lookbacks), SARIMA's evaluated origins and metrics are **identical across `lookback = 24, 48, 168`** (`Macro MAE = 36.566 kWh`, `MASE = 4.929`).",
        "2. **XGBoost Under-Convergence at $H=1$ (`n_estimators=40, learning_rate=0.05`) vs. Recursive Compounding:**",
        "   - Even at `hourly` ($H=1$, where **no recursive rollout occurs**), XGBoost already exhibits a large negative Macro Bias (`-7.671 kWh` at `lb=24h`, `-7.927 kWh` at `lb=48h`, `-7.965 kWh` at `lb=168h`; Macro MASE `1.609`, `1.498`, `1.482`). Because `XGBRegressor` initializes at `base_score = mean(y_train)` (diluted by 38.10% 2016 zeros) and executes `n_estimators = 40` boosting trees at `learning_rate = 0.05`, $(1 - 0.05)^{40} = 0.1285$ of the initial downward offset remains unclosed even on strictly positive 2017 lag vectors.",
        "   - During 24-step (`daily_profile`) and 168-step (`weekly_profile`) recursive rollouts, this 1-step shrinkage bias compounds as underestimated predictions are fed back into subsequent lag features (`Macro Bias -21.350 kWh` at $H=24$ and `-28.332 kWh` at $H=168$ for `lb=168h`).",
        "   - By contrast, `random_forest` (`n_estimators=20, max_depth=10`) averages un-shrunk bootstrap trees that isolate positive lag vectors from zero lag vectors without global `base_score` shrinkage (`hourly` bias `+0.374 kWh`, `daily_profile` bias `-2.827 kWh` at `lb=168h`).",
        "3. **Strict `lag < lookback` Feature Selection (`feature_lags`):**",
        "   - `feature_lags(config, lookback)` filters `feature_lags_hours = [1, 2, 3, 6, 12, 24, 48, 168]` using strict inequality (`lag < lookback`):",
        "     - At `lookback = 24h`, active tabular lags are `[1, 2, 3, 6, 12]` (**excluding `lag_24`**), which explains why `random_forest` underperforms `persistence` at $H=1$ (`MAE 7.492` vs. `7.191 kWh`) and has higher $H=24$ error (`MAE 18.365 kWh`).",
        "     - At `lookback = 48h`, active lags expand to `[1, 2, 3, 6, 12, 24]` (including diurnal `lag_24`), driving a sharp improvement in `random_forest` (`hourly` MAE `6.772 kWh`, `daily_profile` MAE `14.424 kWh`, `weekly_profile` MAE `19.069 kWh`).",
        "     - At `lookback = 168h`, active lags expand to `[1, 2, 3, 6, 12, 24, 48]` (**excluding `lag_168`** because `168 < 168` is false).",
        "",
        "---",
        "",
        "## 4. Unsupervised Anomaly Detection Synthesis (Phase 3B.3 Remediated)",
        "",
        "### Table 5: Remediated Anomaly Detection Macro & Coincidence Summary (29 Core Buildings)",
        "",
        "| Feature Mode | Detector (Dim / Scope) | Operating Point | Nominal Train / 1,000h | Realized Train / 1,000h (Mean / Med) | Val Flagged / 1,000h (Mean / Med / Range) | Bldgs w/ Events (`Zero-Ev`) | Conditional Mean Event Duration (h) | Coincidence Hours ($\\ge 2$ / $\\ge 5$ / Max Simul Bldgs) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m in anom_rows:
        lines.append(
            f"| `{m['feature_mode']}` | `{m['model']}` (`{m['detector_feature_dimension']}D` `{m['detector_feature_scope']}`) | "
            f"`{m['operating_point']}` | `{m['nominal_training_flagged_per_1000_hours']:.1f}` | "
            f"`{m['training_flagged_per_1000_macro_mean']:.2f}` / `{m['training_flagged_per_1000_median']:.2f}` | "
            f"`{m['flagged_per_1000_macro_mean']:.2f}` / `{m['flagged_per_1000_median']:.2f}` / `[{m['flagged_per_1000_range'][0]:.2f}, {m['flagged_per_1000_range'][1]:.2f}]` | "
            f"`{m['buildings_with_events_count']}/29` (`{m['zero_event_buildings_count']}`) | "
            f"`{m['mean_event_duration_macro_mean']:.3f}` | "
            f"`{m['hours_with_multi_building_coincidence']}` / `{m['hours_with_5plus_buildings_flagged']}` / `{m['max_simultaneous_buildings_flagged']}` |"
        )

    lines.extend([
        "",
        "### Table 6: Separate Sensitivity Population (`Panther_education_Quintin`, `1.0pct` Operating Point)",
        "",
        "| Feature Mode | Detector (`Dim`) | Train Eligible Hours | Val Eligible Hours | Threshold ($\\tau_{0.99}$) | Train Flagged / 1,000h | Val Flagged / 1,000h | Event Count | Conditional Mean Duration (h) | Val Flagged Zero Hours |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in quintin_anom:
        if r["operating_point"] == "1.0pct":
            lines.append(
                f"| `{r['feature_mode']}` | `{r['model']}` (`{r['detector_feature_dimension']}D`) | "
                f"{r['train_eligible_hours']:,} | {r['validation_eligible_hours']:,} | "
                f"`{r['threshold_value']:.4f}` | `{r['training_flagged_per_1000_hours']:.2f}` | "
                f"`{r['validation_flagged_per_1000_hours']:.2f}` | {r['event_count']} | "
                f"`{r['mean_event_duration_hours']:.3f}` | {r['validation_flagged_zero_hours']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Cross-Building Heterogeneity & Exploratory Correlation Diagnostics (`lookback=168h`, `1.0pct` Operating Point)",
        "",
        "### Exploratory Correlation & Outlier-Leverage Audit",
        f"- **Daily Random Forest MASE Degradation vs. Seasonal-Deviation LOF Alert Rate (`exploratory_outlier_sensitive_diagnostic`):**",
        f"  - **$N=29$ Linear Correlation:** Pearson $r = {c_lof['pearson_r']:.4f}$ ($p = {c_lof['pearson_p_value']:.4e}$).",
        f"  - **$N=29$ Non-Parametric Rank Correlation:** Spearman $\\rho = {c_lof['spearman_rho']:.4f}$ ($p = {c_lof['spearman_p_value']:.4f}$); Kendall $\\tau = {c_lof['kendall_tau']:.4f}$ ($p = {c_lof['kendall_p_value']:.4f}$).",
        f"  - **Single-Outlier Leverage (`Panther_education_Vincent`):** `Panther_education_Vincent` (`daily_rf_mase_delta = +{c_lof['influential_outlier_mase_delta']:.4f}`, `seasonal_lof = {c_lof['influential_outlier_seasonal_lof_per_1000h']:.2f}/1,000h`) exhibits OLS hat-matrix leverage **$h = {c_lof['influential_outlier_leverage_h']:.4f}$** and **Cook's $D = {c_lof['influential_outlier_cooks_d']:.4f}$**.",
        f"  - **Excluding `Panther_education_Vincent` ($N={c_lof['excluding_vincent_n']}$):** Pearson $r = {c_lof['excluding_vincent_pearson_r']:.4f}$ ($p = {c_lof['excluding_vincent_pearson_p_value']:.4f}$, non-significant); Spearman $\\rho = {c_lof['excluding_vincent_spearman_rho']:.4f}$ ($p = {c_lof['excluding_vincent_spearman_p_value']:.4f}$).",
        f"  - **Diagnostic Conclusion:** Pearson's apparent positive association is dominated by one influential building (`Panther_education_Vincent`) and is **not** a robust general relationship across the 29 buildings (where the 10 RF-winning buildings have a median seasonal LOF rate of `{c_lof['rf_winning_buildings_seasonal_lof_median_per_1000h']:.2f}/1,000h` vs. `{c_lof['baseline_winning_buildings_seasonal_lof_median_per_1000h']:.2f}/1,000h` for the 19 baseline-winning buildings). No causal relationship is implied or supported.",
        f"- **2016 Valid-Zero Rate vs. Daily Random Forest MASE Degradation ($N=29$):** Pearson $r = {c_zero['pearson_r']:.4f}$ ($p = {c_zero['pearson_p_value']:.4f}$); Spearman $\\rho = {c_zero['spearman_rho']:.4f}$ ($p = {c_zero['spearman_p_value']:.4f}$) — non-significant because 2016 zero frequency is nearly uniform across the 29 core buildings.",
        f"- **Daily Random Forest Absolute Bias vs. MAE Degradation ($N=29$):** Pearson $r = {c_bias['pearson_r']:.4f}$ ($p = {c_bias['pearson_p_value']:.4f}$); Spearman $\\rho = {c_bias['spearman_rho']:.4f}$ ($p = {c_bias['spearman_p_value']:.4e}$) — descriptive error-decomposition diagnostic reflecting the mathematical bound $|\\text{{mean\\_error}}| \\le \\text{{MAE}}$.",
        "",
        "### Table 7: Per-Building Integrated Diagnostic Profile (29 Core Buildings)",
        "",
        "| Building ID | Train Scale Denom (kWh) | 2016 Zero / 1,000h | `hourly` Best Model (MASE) | `daily_profile` Best Model (MASE) | RF Daily $\\Delta$MASE (Bias kWh) | `weekly_profile` Best Model (MASE) | Seasonal `1.0pct` Val Flagged / 1,000h (`Z` / `IF` / `LOF`) |",
        "| :--- | :---: | :---: | :--- | :--- | :---: | :--- | :---: |",
    ])
    for p in cb["building_profiles"]:
        lines.append(
            f"| `{p['building_id']}` | `{p['mase_scaling_denominator_kwh']:.2f}` | "
            f"`{p['train_valid_zero_per_1000_hours']:.1f}` | "
            f"`{p['hourly_best_model']}` (`{p['hourly_best_mase']:.3f}`) | "
            f"`{p['daily_best_model']}` (`{p['daily_best_mase']:.3f}`) | "
            f"`{p['daily_rf_mase_delta']:+.3f}` (`{p['daily_rf_mean_error']:+.2f}`) | "
            f"`{p['weekly_best_model']}` (`{p['weekly_best_mase']:.3f}`) | "
            f"`{p['anom_1pct_seasonal_zscore_per_1000h']:.1f}` / `{p['anom_1pct_seasonal_iforest_per_1000h']:.1f}` / `{p['anom_1pct_seasonal_lof_per_1000h']:.1f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Recommended Research Figures & Visual Diagnostics",
        "",
        "1. **Figure 1 — Horizon-by-Model MASE Distribution (Box + Strip Plot across 29 Core Buildings):**",
        "   - Displays per-building `mase_training_daily_naive` at `lookback=168h` across `hourly`, `daily_profile`, and `weekly_profile` for the task baseline, `sarima`, `random_forest`, and `xgboost`, illustrating the shift from `random_forest` superiority at $H=1$ (`MASE = 0.937`) to seasonal-naive superiority at $H=24$ (`MASE = 1.644`) and $H=168$ (`MASE = 1.937`).",
        "2. **Figure 2 — Multi-Step Error Decomposition Scatter Plot (`daily_profile`, `lookback=168h`):**",
        f"   - Plots per-building $|\\text{{mean\\_error}}|$ (kWh) against `mae_delta` (kWh) for `random_forest`, `sarima`, and `xgboost` (Pearson $r = {c_bias['pearson_r']:.4f}$, Spearman $\\rho = {c_bias['spearman_rho']:.4f}$ for `random_forest`), separating $H=1$ shrinkage (`xgboost`) and AR(1) mean-reversion (`sarima`) from building-specific positive load level shifts.",
        "3. **Figure 3 — Training vs. Validation Alert-Rate Operating Curves (`0.5pct`, `1.0pct`, `2.0pct`):**",
        "   - Compares nominal training rates (`5.0`, `10.0`, `20.0` per 1,000h), realized training rates, and validation alert-rate distributions (median and interquartile range across 29 buildings) for `raw_consumption` (`1D`) and `seasonal_deviation` (`1D` `statistical_zscore` vs. `3D` `isolation_forest` and `lof_novelty`).",
        "4. **Figure 4 — Exploratory Cross-Phase Outlier-Leverage Diagnostic Plot:**",
        f"   - Plots `random_forest` `daily_profile` `mase_delta` against `seasonal_deviation` `lof_novelty` validation alert rate (`1.0pct`) with both $N=29$ OLS ($r = {c_lof['pearson_r']:.4f}$) and $N=28$ leave-one-out (`excluding Vincent`, $r = {c_lof['excluding_vincent_pearson_r']:.4f}$, $p = {c_lof['excluding_vincent_pearson_p_value']:.4f}$; Spearman $\\rho = {c_lof['spearman_rho']:.4f}$, $p = {c_lof['spearman_p_value']:.4f}$) overlays, visually demonstrating how a single high-leverage outlier (`Panther_education_Vincent`, $h = {c_lof['influential_outlier_leverage_h']:.4f}$, Cook's $D = {c_lof['influential_outlier_cooks_d']:.4f}$) distorts parametric correlation.",
        "",
        "---",
        "",
        "## 7. Model Selection Conclusions & Threats to Validity",
        "",
        "### Model Selection Conclusions",
        f"- **1-Hour Ahead (`hourly`):** {syn['model_selection_conclusions']['hourly_1h_ahead']['evidence']}",
        f"- **24-Hour Daily Profile (`daily_profile`, Primary Task):** {syn['model_selection_conclusions']['daily_profile_24h_ahead']['evidence']}",
        f"- **168-Hour Weekly Profile (`weekly_profile`, Exploratory Task):** {syn['model_selection_conclusions']['weekly_profile_168h_ahead']['evidence']}",
        f"- **Unsupervised Anomaly Detection:** {syn['model_selection_conclusions']['unsupervised_anomaly_detection']['evidence']}",
        "",
        "### Scientific Limitations & Threats to Validity",
    ])
    for idx, lim in enumerate(syn["scientific_limitations_and_threats_to_validity"], start=1):
        lines.append(f"{idx}. {lim}")
    lines.append("")
    return "\n".join(lines)
