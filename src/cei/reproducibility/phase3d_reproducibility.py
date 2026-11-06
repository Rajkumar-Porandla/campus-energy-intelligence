"""Read-only Phase 3D reproducibility verification, provenance ledger, and deterministic parity audit."""
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from cei.forecasting.phase3a import coverage_report, load_experiment_config, validate_canonical
from cei.forecasting.phase3b_baselines import evaluate_validation, load_baseline_config
from cei.synthesis.phase3c_synthesis import (
    CANDIDATE_MODELS,
    TASK_BASELINES,
    TASKS,
    build_phase3c_synthesis,
    render_markdown_report,
)

PROTECTED_DATASET_CHECKSUMS = {
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

PROTECTED_CONFIG_CHECKSUMS = {
    "src/cei/config/dataset_schema.json": (
        "4d2af6c3a74bd11ff0bf287ca80769cbec2356f777af32fa689000d12cda8c55"
    ),
    "src/cei/config/bdg2_panther.json": (
        "7796fdd7e117f4f78615b49858bcf5ec0a41d00db1e00e04845ac65115010c1c"
    ),
    "src/cei/config/phase3a_experiment.json": (
        "a6583b89556848f01a249f9dd6f5c4b225aea49a7fcd484a2081c0afe1615856"
    ),
    "src/cei/config/phase3b_baseline_experiment.json": (
        "cc67f112d05b8b58188c27e94569840bb1ad5e47978ca0e36a263dc128f687ad"
    ),
    "src/cei/config/phase3b2_candidate_experiment.json": (
        "299f7314907ef4c10a2fe17fb0fd9fa8069ae66eae59bc8ad7d12ce6b186dab2"
    ),
    "src/cei/config/phase3b3_anomaly_experiment.json": (
        "e1a158c7846ec71a199eb794a2c1ce86724bdfeb7529cb850efae9af7a2a1607"
    ),
}

ACCEPTED_REPORT_CHECKSUMS = {
    "data/reports/bdg2/panther/dataset_manifest.json": (
        "4265e095d450937e880f4c8c643722d461cb5050ae6ca0d3ec05030f8fddde5f"
    ),
    "data/reports/bdg2/panther/migration_report.json": (
        "0885019fef2f6823b0ae47edfd5145598218cead7714b4d26aeaa685d56cdd42"
    ),
    "data/reports/bdg2/panther/phase3a_final_validation.json": (
        "845c73e429fa3bd38e50f53d71961e2e57be5167246a04e7de968647e7e141c3"
    ),
    "data/reports/bdg2/panther/phase3a_corrected_validation.json": (
        "017e0c7c3dbd9e1fbd7a882c8833b46975c7f03b26873d9e8c99b38acbc38228"
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
    "data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json": (
        "66fa3fdceb99545cc8fa479aa3a7144075674f794f90fee823881832e681bcd1"
    ),
}

HISTORICAL_REPORT_CHECKSUMS = {
    "data/reports/bdg2/panther/phase3b_baseline_validation.json": (
        "386be9922c24829c795d21851b667675ca9f96f657dc1338930f108665392e29"
    ),
    "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation.json": (
        "061c6662963a8ff4b9b945f6c0ca39ed19194eebe3e182de6dde834147700520"
    ),
    "data/reports/bdg2/panther/phase3b2_candidate_validation.json": (
        "e0fa9784703e72cca658947777c9966ceb211be779eead4979b442890c91faa0"
    ),
    "data/reports/bdg2/panther/phase3b3_anomaly_validation.json": (
        "be9f621fd1fc3851d1e8fab69bc56a69a70304ff604ebee053eddae3f9fdb309"
    ),
    "data/reports/bdg2/panther/phase3c_integrated_research_synthesis.json": (
        "4599210c9deb9206d5e6c1b82dd2700d4a0e926ac8227ed7d105ad849d830552"
    ),
}

PROTECTED_DOC_CHECKSUMS = {
    "docs/PHASE_2_7_ML_EXPERIMENTATION_SPEC.md": (
        "d3246d3ede70323564cdb49059448c7ee94425c2dcc270cb3555b01ff0c39ce6"
    ),
    "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS.md": (
        "b78f2c23708e22eff7724f7d70b9721b3d7aa87a3c9921e52f7bf6470aedc05e"
    ),
    "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md": (
        "9d94cd49e9e030c8a268769bef11e0b84895dc198f1a87132fa725704b05d880"
    ),
}

PRIOR_PHASE3D_CHECKSUMS = {
    "data/reports/bdg2/panther/phase3d_reproducibility_manifest.json": (
        "e18101a19f9f6af64603e4c2da832aac27fe56d671f45ad2c3bbdd96010e24ee"
    ),
    "docs/PHASE_3D_FINAL_RESEARCH_REPORT.md": (
        "2a6ab0323c33010026597aa058fd80602de169714b3cc1c604f0d7aed9819d8c"
    ),
    "docs/REPRODUCIBILITY_PLAYBOOK.md": (
        "74de2982415e9e61f3bc33ef6063cafdf7fb8387481fc88524003afc1ae21a85"
    ),
}

FORBIDDEN_SUPERVISED_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "auc"}

ANOMALY_MASK_KEYS = (
    "train_partition_hours",
    "train_observed_hours",
    "train_missing_hours",
    "train_insufficient_history_exclusions",
    "train_eligible_hours",
    "train_valid_zero_hours",
    "validation_partition_hours",
    "validation_observed_hours",
    "validation_missing_hours",
    "validation_insufficient_history_exclusions",
    "validation_eligible_hours",
    "validation_valid_zero_hours",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_group(root_path, expected_map, group_label):
    verified = {}
    for rel_path, expected in expected_map.items():
        target = root_path / rel_path
        if not target.exists():
            raise FileNotFoundError(f"Missing {group_label} artifact: {rel_path}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"{group_label} checksum mismatch for {rel_path}: {actual} != {expected}")
        verified[rel_path] = actual
    return verified


def verify_all_protected_checksums(root):
    root_path = Path(root)
    return {
        "protected_datasets": _verify_group(root_path, PROTECTED_DATASET_CHECKSUMS, "Protected dataset"),
        "protected_configs": _verify_group(root_path, PROTECTED_CONFIG_CHECKSUMS, "Protected config"),
        "accepted_reports": _verify_group(root_path, ACCEPTED_REPORT_CHECKSUMS, "Accepted report"),
        "historical_reports": _verify_group(root_path, HISTORICAL_REPORT_CHECKSUMS, "Historical report"),
        "protected_docs": _verify_group(root_path, PROTECTED_DOC_CHECKSUMS, "Protected doc"),
    }


def verify_prior_phase3d_checksums(root):
    root_path = Path(root)
    return _verify_group(root_path, PRIOR_PHASE3D_CHECKSUMS, "Prior Phase 3D artifact")


def collect_environment_manifest():
    packages = (
        "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "xgboost", "pytest", "joblib",
    )
    versions = {pkg: importlib.metadata.version(pkg) for pkg in packages}
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": versions,
    }


def verify_in_memory_reproduction(root):
    root_path = Path(root)
    phase3a_cfg = load_experiment_config(root_path / "src/cei/config/phase3a_experiment.json")
    baseline_cfg = load_baseline_config(root_path / "src/cei/config/phase3b_baseline_experiment.json")

    core_path = root_path / "data/processed/bdg2/panther/electricity_core.csv"
    weather_path = root_path / "data/processed/bdg2/panther/weather.csv"
    readings, timestamps = validate_canonical(core_path, weather_path, phase3a_cfg)

    # 1. Verify Phase 3A coverage in memory against both corrected and final reports
    cov_rows = coverage_report(readings, phase3a_cfg)
    p3a_corr = json.loads((root_path / "data/reports/bdg2/panther/phase3a_corrected_validation.json").read_text())
    p3a_final = json.loads((root_path / "data/reports/bdg2/panther/phase3a_final_validation.json").read_text())
    if len(cov_rows) != 783 or len(p3a_corr["coverage"]) != 783 or len(p3a_final["coverage"]) != 783:
        raise ValueError("Phase 3A coverage row count must equal 783 (29 buildings x 3 partitions x 3 tasks x 3 lookbacks)")
    if cov_rows != p3a_corr["coverage"]:
        raise ValueError("In-memory Phase 3A coverage does not match phase3a_corrected_validation.json")
    for r_mem, r_fin in zip(cov_rows, p3a_final["coverage"], strict=True):
        for k, v in r_fin.items():
            if r_mem[k] != v:
                raise ValueError(f"Phase 3A final coverage mismatch on key {k}")
    train_val_cov_count = sum(1 for r in cov_rows if r["partition"] in ("train", "validation"))
    test_cov_count = sum(1 for r in cov_rows if r["partition"] == "test")
    if train_val_cov_count != 522 or test_cov_count != 261:
        raise ValueError("Phase 3A coverage partition split must be 522 train/validation rows and 261 test coverage-only rows")

    # 2. Verify Phase 3B.1 baseline evaluation in memory
    base_res = evaluate_validation(readings, phase3a_cfg, baseline_cfg)
    p3b1_disk = json.loads(
        (root_path / "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json").read_text()
    )
    if base_res != p3b1_disk["results"]:
        raise ValueError("In-memory Phase 3B.1 baseline results do not match phase3b_baseline_mase_audit_validation_v2.json")

    # 3. Verify Phase 3C synthesis in memory
    p3c_mem = build_phase3c_synthesis(root_path)
    p3c_disk = json.loads(
        (root_path / "data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json").read_text()
    )
    p3c_mem_no_ts = {k: v for k, v in p3c_mem.items() if k != "execution"}
    p3c_disk_no_ts = {k: v for k, v in p3c_disk.items() if k != "execution"}
    if p3c_mem_no_ts != p3c_disk_no_ts:
        raise ValueError("In-memory Phase 3C synthesis does not match phase3c_integrated_research_synthesis_remediated.json")

    return {
        "phase3a_in_memory_parity": "passed",
        "phase3a_coverage_rows_verified": len(cov_rows),
        "phase3a_train_validation_coverage_rows": train_val_cov_count,
        "phase3a_test_coverage_only_rows": test_cov_count,
        "phase3a_timestamp_count": len(timestamps),
        "phase3a_core_building_count": len(readings),
        "phase3b1_in_memory_parity": "passed",
        "phase3b1_baseline_rows_verified": len(base_res["rows"]),
        "phase3c_in_memory_parity": "passed",
        "phase3c_report_version_verified": p3c_disk["report_version"],
    }


def _collect_all_keys_lower(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys.update(_collect_all_keys_lower(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys_lower(item))
    return keys


def verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3):
    # Phase 3B.2 structural, determinism, and shared-mask invariants
    if p3b2["status"] != "validation_only" or p3b2["configuration"]["test_evaluation"] != "not_run":
        raise ValueError("Phase 3B.2 report violates validation-only or test_evaluation guardrails")
    if p3b2["configuration"]["multi_step_strategy"] != "frozen_origin_recursive":
        raise ValueError("Phase 3B.2 report does not use frozen_origin_recursive strategy")
    if p3b2["configuration"].get("random_seed") != 42:
        raise ValueError("Phase 3B.2 configuration random_seed must equal 42")
    p3b2_models = p3b2["configuration"].get("models", {})
    for model_name in ("random_forest", "xgboost"):
        if p3b2_models.get(model_name, {}).get("n_jobs") != 1:
            raise ValueError(f"Phase 3B.2 model {model_name} must enforce n_jobs == 1")
    if p3b2["results"]["baseline"] != p3b1["results"]:
        raise ValueError("Phase 3B.2 embedded baseline results do not match Phase 3B.1 accepted report")
    base_by_cell = {
        (r["building_id"], r["task"], r["lookback_hours"]): r
        for r in p3b2["results"]["baseline"]["rows"]
    }
    cand_rows = p3b2["results"]["candidates"]["rows"]
    if len(cand_rows) != 783 or any("Quintin" in r["building_id"] for r in cand_rows):
        raise ValueError("Phase 3B.2 candidate row count or Quintin exclusion failed")
    for c_row in cand_rows:
        b_row = base_by_cell[(c_row["building_id"], c_row["task"], c_row["lookback_hours"])]
        if (
            c_row["candidate_origins"] != b_row["candidate_origins"]
            or c_row["eligible_origins"] != b_row["eligible_origins"]
            or c_row["eligible_target_hours"] != b_row["eligible_target_hours"]
            or c_row["missing_target_exclusions"] != b_row["missing_target_exclusions"]
            or c_row["insufficient_history_exclusions"] != b_row["insufficient_history_exclusions"]
        ):
            raise ValueError("Phase 3B.2 candidate and baseline origin/target masks diverged")

    # Phase 3B.3 structural, determinism, shared-mask, and threshold monotonicity invariants
    if (
        p3b3["report_version"] != "3B.3-remediated-v1"
        or p3b3["status"] != "validation_only"
        or p3b3["ground_truth_labels"] != "none"
        or p3b3["configuration"]["test_evaluation"] != "not_run"
    ):
        raise ValueError("Phase 3B.3 report violates validation-only or ground-truth guardrails")
    if p3b3["configuration"].get("random_seed") != 42:
        raise ValueError("Phase 3B.3 configuration random_seed must equal 42")
    p3b3_models = p3b3["configuration"].get("models", {})
    for model_name in ("isolation_forest", "lof_novelty"):
        if p3b3_models.get(model_name, {}).get("n_jobs") != 1:
            raise ValueError(f"Phase 3B.3 model {model_name} must enforce n_jobs == 1")

    anom_rows = p3b3["results"]["rows"]
    quintin_rows = p3b3["results"]["sensitivity_quintin"]["rows"]
    if len(anom_rows) != 522 or len(quintin_rows) != 18:
        raise ValueError("Phase 3B.3 core or sensitivity row count mismatch")
    if any("Quintin" in r["building_id"] for r in anom_rows):
        raise ValueError("Phase 3B.3 core rows must exclude Quintin")

    all_anom_rows = anom_rows + quintin_rows
    by_bldg_mode = {}
    by_bldg_mode_model = {}
    for r in all_anom_rows:
        if r["validation_flagged_zero_hours"] != 0:
            raise ValueError("Phase 3B.3 flagged a valid zero reading")
        if r["event_count"] == 0 and r["mean_event_duration_hours"] is not None:
            raise ValueError("Phase 3B.3 zero-event building must have null mean_event_duration_hours")
        if r["event_count"] > 0 and (r["mean_event_duration_hours"] is None or r["mean_event_duration_hours"] < 1.0):
            raise ValueError("Phase 3B.3 positive-event building must have mean_event_duration_hours >= 1.0")
        by_bldg_mode.setdefault((r["building_id"], r["feature_mode"]), []).append(r)
        by_bldg_mode_model.setdefault((r["building_id"], r["feature_mode"], r["model"]), {})[
            r["operating_point"]
        ] = r

    # Verify cross-detector eligibility-mask parity per (building_id, feature_mode)
    if len(by_bldg_mode) != 60:
        raise ValueError("Phase 3B.3 must have 60 (building_id, feature_mode) cells (30 buildings x 2 modes)")
    for cell_key, cell_rows in by_bldg_mode.items():
        if len(cell_rows) != 9:
            raise ValueError(f"Phase 3B.3 cell {cell_key} must have 9 rows (3 models x 3 operating points)")
        ref = {k: cell_rows[0][k] for k in ANOMALY_MASK_KEYS}
        for row in cell_rows[1:]:
            cur = {k: row[k] for k in ANOMALY_MASK_KEYS}
            if cur != ref:
                raise ValueError(f"Phase 3B.3 eligibility mask mismatch across detectors in {cell_key}")

    # Verify threshold and flagged-hour monotonicity across 0.5pct, 1.0pct, 2.0pct
    if len(by_bldg_mode_model) != 180:
        raise ValueError("Phase 3B.3 must have 180 (building_id, feature_mode, model) series")
    for series_key, op_map in by_bldg_mode_model.items():
        if set(op_map.keys()) != {"0.5pct", "1.0pct", "2.0pct"}:
            raise ValueError(f"Phase 3B.3 series {series_key} is missing operating points")
        r05, r10, r20 = op_map["0.5pct"], op_map["1.0pct"], op_map["2.0pct"]
        if not (r05["threshold_value"] >= r10["threshold_value"] >= r20["threshold_value"]):
            raise ValueError(f"Phase 3B.3 threshold monotonicity violated in {series_key}")
        if series_key[2] in ("isolation_forest", "lof_novelty") and not (
            r05["threshold_value"] > r10["threshold_value"] > r20["threshold_value"]
        ):
            raise ValueError(f"Phase 3B.3 strict threshold monotonicity violated for {series_key}")
        if not (r05["training_flagged_hours"] <= r10["training_flagged_hours"] <= r20["training_flagged_hours"]):
            raise ValueError(f"Phase 3B.3 training flagged hours monotonicity violated in {series_key}")
        if not (r05["validation_flagged_hours"] <= r10["validation_flagged_hours"] <= r20["validation_flagged_hours"]):
            raise ValueError(f"Phase 3B.3 validation flagged hours monotonicity violated in {series_key}")

    if not FORBIDDEN_SUPERVISED_KEYS.isdisjoint(_collect_all_keys_lower(p3b3)):
        raise ValueError("Phase 3B.3 contains forbidden supervised metric keys")

    return {
        "phase3b2_invariants": "passed",
        "phase3b2_candidate_rows_verified": len(cand_rows),
        "phase3b2_random_seed": p3b2["configuration"]["random_seed"],
        "phase3b2_n_jobs": {
            "random_forest": p3b2_models["random_forest"]["n_jobs"],
            "xgboost": p3b2_models["xgboost"]["n_jobs"],
        },
        "phase3b3_invariants": "passed",
        "phase3b3_core_rows_verified": len(anom_rows),
        "phase3b3_sensitivity_rows_verified": len(quintin_rows),
        "phase3b3_mask_parity_cells_verified": len(by_bldg_mode),
        "phase3b3_threshold_monotonicity_series_verified": len(by_bldg_mode_model),
        "phase3b3_random_seed": p3b3["configuration"]["random_seed"],
        "phase3b3_n_jobs": {
            "isolation_forest": p3b3_models["isolation_forest"]["n_jobs"],
            "lof_novelty": p3b3_models["lof_novelty"]["n_jobs"],
        },
    }


def verify_phase3b2_and_3b3_invariants(root):
    root_path = Path(root)
    p3b1 = json.loads(
        (root_path / "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json").read_text()
    )
    p3b2 = json.loads(
        (root_path / "data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json").read_text()
    )
    p3b3 = json.loads(
        (root_path / "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json").read_text()
    )
    return verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3)


def build_phase3d_manifest(root):
    root_path = Path(root)
    checksums = verify_all_protected_checksums(root_path)
    prior_p3d = verify_prior_phase3d_checksums(root_path)
    in_memory = verify_in_memory_reproduction(root_path)
    invariants = verify_phase3b2_and_3b3_invariants(root_path)
    after_checksums = verify_all_protected_checksums(root_path)
    after_prior_p3d = verify_prior_phase3d_checksums(root_path)
    if checksums != after_checksums or prior_p3d != after_prior_p3d:
        raise RuntimeError("Protected checksums changed during Phase 3D verification")

    return {
        "phase": "3D",
        "report_version": "3D-remediated-v1",
        "title": "Reproducibility Manifest & Provenance Ledger (Remediated)",
        "status": "reproducibility_verified_validation_only",
        "test_evaluation": "not_run",
        "ground_truth_anomaly_labels": "none",
        "execution": {
            **collect_environment_manifest(),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "protected_artifact_counts": {
            "protected_datasets": len(checksums["protected_datasets"]),
            "protected_configs": len(checksums["protected_configs"]),
            "accepted_reports": len(checksums["accepted_reports"]),
            "historical_reports": len(checksums["historical_reports"]),
            "protected_docs": len(checksums["protected_docs"]),
            "prior_phase3d_outputs": len(prior_p3d),
            "total_protected_files": sum(len(v) for v in checksums.values()),
        },
        "checksum_ledger": {
            **checksums,
            "prior_phase3d_outputs": prior_p3d,
        },
        "deterministic_reproduction_checks": {
            **in_memory,
            **invariants,
        },
        "report_lineage": [
            {
                "phase": "2.6",
                "accepted_reports": [
                    "data/reports/bdg2/panther/dataset_manifest.json",
                    "data/reports/bdg2/panther/migration_report.json",
                ],
                "superseded_reports": [],
                "script": "scripts/migrate_bdg2.py",
                "config": "src/cei/config/bdg2_panther.json",
            },
            {
                "phase": "3A",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3a_corrected_validation.json",
                    "data/reports/bdg2/panther/phase3a_final_validation.json",
                ],
                "superseded_reports": [],
                "script": "scripts/run_phase3a_validation.py",
                "config": "src/cei/config/phase3a_experiment.json",
                "notes": (
                    "phase3a_final_validation.json and phase3a_corrected_validation.json have identical "
                    "core coverage counts across all 783 rows (522 train/validation rows + 261 test "
                    "coverage-only rows); phase3a_corrected_validation.json adds shortest-lookback "
                    "coverage reduction fields and exclusion precedence documentation."
                ),
            },
            {
                "phase": "3B.1",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json",
                ],
                "superseded_reports": [
                    "data/reports/bdg2/panther/phase3b_baseline_validation.json",
                    "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation.json",
                ],
                "script": "scripts/run_phase3b_baselines.py",
                "config": "src/cei/config/phase3b_baseline_experiment.json",
            },
            {
                "phase": "3B.2",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json",
                ],
                "superseded_reports": [
                    "data/reports/bdg2/panther/phase3b2_candidate_validation.json",
                ],
                "script": "scripts/run_phase3b2_candidates.py",
                "config": "src/cei/config/phase3b2_candidate_experiment.json",
            },
            {
                "phase": "3B.3",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json",
                ],
                "superseded_reports": [
                    "data/reports/bdg2/panther/phase3b3_anomaly_validation.json",
                ],
                "script": "scripts/run_phase3b3_anomaly.py",
                "config": "src/cei/config/phase3b3_anomaly_experiment.json",
            },
            {
                "phase": "3C",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json",
                    "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md",
                ],
                "superseded_reports": [
                    "data/reports/bdg2/panther/phase3c_integrated_research_synthesis.json",
                    "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS.md",
                ],
                "script": "scripts/run_phase3c_synthesis.py",
                "config": None,
            },
            {
                "phase": "3D",
                "accepted_reports": [
                    "data/reports/bdg2/panther/phase3d_reproducibility_manifest_remediated.json",
                    "docs/PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md",
                    "docs/REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md",
                ],
                "superseded_reports": [
                    "data/reports/bdg2/panther/phase3d_reproducibility_manifest.json",
                    "docs/PHASE_3D_FINAL_RESEARCH_REPORT.md",
                    "docs/REPRODUCIBILITY_PLAYBOOK.md",
                ],
                "script": "scripts/run_phase3d_reproducibility.py",
                "config": None,
            },
        ],
        "unused_phase1_placeholder_packages": [
            "src/cei/analytics",
            "src/cei/intelligence",
            "src/cei/visualization",
        ],
    }


def render_final_research_report(syn, manifest):
    cov = syn["coverage_and_regime_synthesis"]["coverage_macro_summary"]
    zs = syn["coverage_and_regime_synthesis"]["zero_frequency_shift_summary"]
    q_zs = zs["sensitivity_quintin"]
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
    env = manifest["execution"]
    pkg = env["package_versions"]
    counts = manifest["protected_artifact_counts"]

    lines = [
        "# Campus Energy Consumption Intelligence & Unsupervised Anomaly Detection: Final Research Report (Phase 3D Remediated)",
        "",
        f"**Report Version:** `{manifest['report_version']}` | **Synthesis Baseline:** `{syn['report_version']}` | **Evaluation Scope:** `validation_only` (`2016` Train $\\to$ `H1 2017` Validation) | **Held-Out Test Partition (`H2 2017`):** `{manifest['test_evaluation']}` | **Ground-Truth Anomaly Labels:** `{manifest['ground_truth_anomaly_labels']}`",
        "",
        "---",
        "",
        "## 1. Abstract & Executive Summary",
        "",
        "This report presents the complete, cryptographically verified experimental methodology and empirical findings of the **Campus Energy Consumption Intelligence (CEI)** research program across Phases 1 through 3D. Following a rigorous data-quality audit that rejected the initial 300-row synthetic Kaggle dataset (`synthetic_campus_energy_usage.csv`, spanning `5` buildings—`Admin Office`, `Classroom Block`, `Computer Lab`, `Hostel`, and `Library`—with only `52–60` observed dates per building across observed date spans of `355–360 days`, `17` conflicting duplicate `(building, date)` groups differing by `4–385 kWh`, and undocumented three categorical classes for `Usage_Level`—`High` (`114` rows), `Medium` (`99` rows), and `Low` (`87` rows)), the study migrated in **Phase 2.6** to the immutable hourly electricity and weather records of the **University of Central Florida (UCF) Panther** site from the **Building Data Genome Project 2 (BDG2)** repository (Miller et al., 2020, *Scientific Data* 7, 368).",
        "",
        "Across the **29 core Panther educational buildings** (`8,784` training hours in 2016; `4,344` validation hours in H1 2017) and the strictly isolated sensitivity building (`Panther_education_Quintin`, `56.61%` missing in 2016), we evaluate:",
        "1. **Multi-Horizon Electricity Forecasting (Phases 3A, 3B.1, 3B.2):** Persistence and seasonal-naive baselines (`period=24h`, `period=168h`) versus frozen-origin recursive **SARIMA**, **Random Forest**, and **XGBoost** across three horizons (`hourly` $H=1$, secondary; `daily_profile` $H=24$, primary; `weekly_profile` $H=168$, exploratory) and three historical lookback windows (`24h`, `48h`, `168h`) over identical common-lookback origin and target masks.",
        "2. **Training-Calibrated Unsupervised Anomaly Detection (Phase 3B.3 Remediated):** Per-building **Statistical Z-Score**, **Isolation Forest**, and full-empirical-multiplicity **Local Outlier Factor (LOF) novelty detection** across `raw_consumption` (`1D`) and `seasonal_deviation` (`1D` univariate reference vs. `3D` joint slot-deviation and ramp vector) at frozen 2016 training upper-tail score quantiles (`99.5th`, `99th`, and `98th` percentiles; nominal `0.5%`, `1.0%`, and `2.0%` training alert rates).",
        "",
        "### Principal Empirical Findings",
        "- **Horizon-Dependent Forecasting Superiority:** No single model dominates all horizons. At **$H=1$ (`hourly`)**, `random_forest` (`lookback=168h`) achieves the lowest macro mean MAE (**`6.731 kWh`** vs. `7.100 kWh` for `persistence`), lowest macro mean RMSE (**`10.813 kWh`** vs. `12.566 kWh`), and lowest macro mean MASE (**`0.937`** vs. `1.017`), winning **`18/29`** buildings on MAE (`19/29` on RMSE). At **$H=24$ (`daily_profile`, primary task)** and **$H=168$ (`weekly_profile`, exploratory task)**, `daily_seasonal_naive` (Macro MAE **`12.158 kWh`**, Median **`10.551 kWh`**, Macro RMSE **`19.841 kWh`**, Bias **`+0.009 kWh`**, Macro MASE **`1.644`**) and `weekly_seasonal_naive` (Macro MAE **`14.135 kWh`**, Macro MASE **`1.937`**) outperform all frozen-origin recursive candidates in aggregate macro metrics, winning **`19/29`** and **`20/29`** buildings on MAE, respectively. `random_forest` (`lookback=168h`) is the sole competitive ML candidate on multi-step profiles (`daily_profile` Macro MAE **`14.557 kWh`**, Macro MASE **`1.909`**, winning **`10/29`** buildings on MAE and **`14/29`** on RMSE).",
        "- **Mechanistic Separation of Model Configuration vs. Recursive Drift:** Multi-step degradation in Phase 3B.2 reflects distinct architectural and configuration mechanisms rather than regime shift alone: (a) `sarima` is configured as a non-seasonal AR(1) `(1,0,0)x(0,0,0,0)` with constant intercept (`trend='c'`), causing multi-step forecasts $\\hat{y}_{t+h|t} = \\mu_{2016} + \\phi_1^h(y_t - \\mu_{2016})$ to decay geometrically toward the zero-diluted 2016 unconditional mean (`daily_profile` Bias `-21.232 kWh`, `weekly_profile` `-33.498 kWh`); (b) `xgboost` (`n_estimators=40`, `learning_rate=0.05`) retains $(1 - 0.05)^{40} = 0.1285$ of its zero-diluted global `base_score = mean(y_train)`, producing a **`-7.965 kWh`** negative bias (`MASE = 1.482`) **even at $H=1$ without recursion** that compounds across 24-step (`-21.350 kWh`) and 168-step (`-28.332 kWh`) rollouts; and (c) strict `lag < lookback` filtering in `feature_lags` excludes `lag_24` at `lookback=24h` (`[1, 2, 3, 6, 12]`) and includes `lag_24` at `lookback=48h` (`[1, 2, 3, 6, 12, 24]`), cutting `random_forest` `daily_profile` Macro MAE from `18.365` to `14.424 kWh`.",
        f"- **Campus-Wide Valid-Zero Shift vs. Building-Specific Active-Load Shifts:** All 29 core buildings shift from a campus-wide mean of **`{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}` valid zero hours per 1,000 observed hours** (`38.10%`, median `{zs['core_train_valid_zero_per_1000_median']:.2f}/1,000h`, range `[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]`) in 2016 to **`0.00/1,000h`** in H1 2017. While this campus-wide shift depresses 2016 unconditional means and lowers training score quantiles, 2016 zero rates are nearly uniform across the 29 buildings (`20/29` within `384.0–384.8/1,000h`) and do **not** significantly explain cross-building differences in forecasting degradation (Pearson $r = {c_zero['pearson_r']:.4f}$, $p = {c_zero['pearson_p_value']:.4f}$; Spearman $\\rho = {c_zero['spearman_rho']:.4f}$, $p = {c_zero['spearman_p_value']:.4f}$).",
        f"- **Unsupervised Anomaly Detection & Outlier-Leverage Diagnostics:** Full-empirical-multiplicity `lof_novelty` and `isolation_forest` in 3D `seasonal_deviation` mode flag **zero valid `0.0 kWh` readings** (`validation_flagged_zero_hours == 0` across all `522` core and `18` `Quintin` rows) and maintain strict threshold monotonicity (`0.5pct > 1.0pct > 2.0pct` in `29/29` buildings, while `statistical_zscore` satisfies non-increasing threshold monotonicity `0.5pct >= 1.0pct >= 2.0pct`). Furthermore, an exploratory cross-phase correlation audit proves that the apparent $N=29$ linear correlation between `random_forest` `daily_profile` `mase_delta` and `seasonal_deviation` `lof_novelty` alert rate (Pearson $r = {c_lof['pearson_r']:.4f}$) is a **non-robust single-outlier artifact** driven by `Panther_education_Vincent` (hat leverage $h = {c_lof['influential_outlier_leverage_h']:.4f}$, Cook's $D = {c_lof['influential_outlier_cooks_d']:.4f}$): across all 29 buildings, Spearman rank correlation is negative and null ($\\rho = {c_lof['spearman_rho']:.4f}$, $p = {c_lof['spearman_p_value']:.4f}$), and excluding `Panther_education_Vincent` ($N=28$) reduces Pearson $r$ to `{c_lof['excluding_vincent_pearson_r']:.4f}` ($p = {c_lof['excluding_vincent_pearson_p_value']:.4f}$).",
        "",
        "---",
        "",
        "## 2. Dataset Provenance, Quality Audit & Canonical Migration (Phases 1–2.6)",
        "",
        "### 2.1 Rejection of Initial Synthetic Dataset (Phases 1 & 2)",
        "In Phases 1 and 2, the supplied 300-row Kaggle dataset (`synthetic_campus_energy_usage.csv`) was profiled (`data/processed/dataset_profile.json`) and conservatively preprocessed (`data/processed/preprocessing_report.json`) without imputation or deletion. Inspection revealed:",
        "- Only `300` total records across `5` buildings (`Admin Office`, `Classroom Block`, `Computer Lab`, `Hostel`, and `Library`) spanning `2025-01-01` to `2025-12-28`, yielding only `52–60` distinct observation dates per building across observed date spans of `355–360 days` (`295–307` unobserved days per building; `14.5%–16.9%` daily coverage).",
        "- `17` repeated `(building_id, date)` keys (`34` rows) with conflicting electricity consumption values differing by `4–385 kWh` (and `9` of the `17` pairs also assigning conflicting `Usage_Level` categories).",
        "- Undocumented `Usage_Level` with three categorical classes — `High` (`114` rows), `Medium` (`99` rows), and `Low` (`87` rows)—whose derivation is not specified by the source and does not represent anomaly ground truth.",
        "",
        "Because irregular 83%–85%-missing daily series cannot support valid daily or weekly seasonal lags, the project migrated in **Phase 2.6** to the BDG2 UCF Panther dataset while preserving all Phase 1–2 scripts, reports, and tests intact.",
        "",
        "### 2.2 BDG2 UCF Panther Canonical Ingestion (Phase 2.6)",
        "- **Source Repository:** Building Data Genome Project 2 (`buds-lab/building-data-genome-project-2`, Miller et al., 2020, [doi:10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x); license recorded conservatively as `CC BY-SA 4.0`).",
        "- **Core Population (`electricity_core.csv`):** `29` UCF Panther educational buildings (`primaryspaceusage == 'Education'`), spanning `17,544` hourly timestamps (`2016-01-01 00:00:00` through `2017-12-31 23:00:00`, timezone-naive source clock) with `508,776` canonical building-hour records (`meter_reading` in kWh).",
        "- **Sensitivity Population (`electricity_sensitivity_quintin.csv`):** `Panther_education_Quintin` (`17,544` timestamps, `4,973` missing hours in 2016 = `56.61%` missing) is stored and evaluated in strict isolation so its prolonged 2016 meter outage cannot distort core macro summaries.",
        "- **Weather (`weather.csv`):** `17,544` site-level hourly records for `site_id=Panther` (`airTemperature`, `dewTemperature`, `windSpeed`, `precipDepth1HR`, `seaLvlPressure`), kept separate and aligned strictly by exact timestamp.",
        "",
        "---",
        "",
        "## 3. Experimental Methodology & Leakage Prevention Architecture (Phases 2.7 & 3A)",
        "",
        "### 3.1 Chronological Partitioning & Non-Overlapping Forecast Blocks",
        "| Partition | Inclusive Timestamp Range | Total Hours | Experimental Role & Status |",
        "| :--- | :---: | :---: | :--- |",
        "| **Train** | `2016-01-01 00:00:00` – `2016-12-31 23:00:00` | `8,784` | Fit model parameters, seasonal slot references, scalers, MASE denominators, and quantile thresholds |",
        "| **Validation** | `2017-01-01 00:00:00` – `2017-06-30 23:00:00` | `4,344` | Evaluate frozen 2016 forecasting models and frozen anomaly operating points |",
        "| **Test** | `2017-07-01 00:00:00` – `2017-12-31 23:00:00` | `4,416` | **Held-out (`not_run`)** — untouched across all model evaluations |",
        "",
        "- **Block Origin Definition:** For horizon $H \\in \\{24, 168\\}$, non-overlapping target blocks begin at $\\text{block\\_start} = \\text{partition\\_start} + k H\\text{ hours}$ ($k \\ge 0$) while the full $H$-hour block lies within the partition, with forecast origin $o = \\text{block\\_start} - 1\\text{ hour}$. For $H=1$ (`hourly`), every target timestamp $t$ in the partition has origin $o = t - 1\\text{ hour}$.",
        "- **Common Lookback & Shared Eligibility Mask:** For each task with seasonal baseline period $P \\in \\{1, 24, 168\\}$ and candidate lookback $L \\in \\{24, 48, 168\\}$, the common required history is $M = \\max(L, P)\\text{ hours}$. An origin $o$ is eligible if and only if all $M$ historical observations $\\{o - (M-1)\\text{h}, \\dots, o\\}$ and all $H$ target observations $\\{o+1\\text{h}, \\dots, o+H\\text{h}\\}$ are non-missing. Every baseline and candidate model is scored on the identical set of eligible origins and targets.",
        "- **Phase 3A Coverage Accounting (`783` Total Rows vs. `522` Train/Validation Rows):** Phase 3A's `coverage_report` audits timestamp/missingness eligibility across all `783` building-partition-task-lookback cells (`29 buildings x 3 partitions x 3 tasks x 3 lookbacks`), comprising `522` `train`/`validation` rows (summarized in Table 1 below) and `261` `test` partition timestamp-coverage-only rows (no model fitting or test-partition scoring is ever performed; `test_evaluation == 'not_run'`).",
        "- **MASE Scaling:** For each building $b$, $\\text{MASE}_{b} = \\text{MAE}_{b,\\text{val}} / d_b$, where $d_b = \\frac{1}{|\\mathcal{P}_{b,\\text{train}}|} \\sum_{(t, t-24) \\in \\mathcal{P}_{b,\\text{train}}} |y_{b,t} - y_{b,t-24}|$ is computed strictly over non-missing period-24 training pairs in 2016.",
        "",
        "### Table 1: Chronological Partition Coverage & Common-Lookback Eligibility (`train` and `validation` Partitions; 29 Core Buildings)",
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

    lines.extend([
        "",
        "### Table 2: Training (2016) vs. Validation (H1 2017) Valid-Zero Frequency Shift",
        "",
        "| Population | Buildings | 2016 Train Observed Hours | 2016 Valid Zero / 1,000h (Mean / Median / Range) | H1 2017 Val Observed Hours | H1 2017 Valid Zero / 1,000h | Shift $\\Delta$ (Val $-$ Train / 1,000h) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Core Panther Education** | 29 | 8,784 | `{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}` / `{zs['core_train_valid_zero_per_1000_median']:.2f}` / `[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]` | 4,332 | `0.00` | **`{zs['core_delta_per_1000_macro_mean']:.2f}`** |",
        f"| **Sensitivity (`Quintin`)** | 1 | {q_zs['train_observed_hours']:,} | `{q_zs['train_valid_zero_per_1000_hours']:.2f}` (`{q_zs['train_valid_zero_hours']}` zero hours) | {q_zs['validation_observed_hours']:,} | `0.00` | **`{q_zs['valid_zero_per_1000_delta_validation_minus_train']:.2f}`** |",
        "",
        "---",
        "",
        "## 4. Multi-Horizon Forecasting Evaluation & Mechanistic Analysis (Phases 3B.1 & 3B.2)",
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
        "### 4.1 Mechanistic Separation of Configuration Effects vs. Recursive Error Propagation",
        "1. **SARIMA Non-Seasonal AR(1) Specification (`(1,0,0)x(0,0,0,0)` with `trend='c'`):**",
        "   - In `src/cei/config/phase3b2_candidate_experiment.json`, SARIMA is configured with `order = [1, 0, 0]` and `seasonal_order = [0, 0, 0, 0]` (`trend='c'`) across `hourly`, `daily_profile`, and `weekly_profile`. Because no seasonal autoregressive or moving-average polynomials ($s=24, 168$) are present, multi-step recursive predictions $\\hat{y}_{t+h|t} = \\mu_{2016} + \\phi_1^h(y_t - \\mu_{2016})$ decay geometrically toward the unconditional 2016 training mean $\\mu_{2016}$ (depressed by the 38.10% 2016 valid-zero regime), producing large negative biases at $H=24$ (`-21.232 kWh`, MASE `3.801`) and $H=168$ (`-33.498 kWh`, MASE `4.929`).",
        "   - Furthermore, `_fit_sarima` fits on the full 2016 series without using `lookback_hours` for feature construction; `lookback_hours` only affects the common-origin eligibility mask $\\max(\\text{lookback}, \\text{period})$. Consequently, for `weekly_profile` (`period = 168`, so $\\max(\\text{lookback}, 168) = 168$ for `lookback = 24, 48, 168`), SARIMA's evaluated origins and metrics are **identical across all three lookbacks** (`Macro MAE = 36.566 kWh`, `MASE = 4.929`).",
        "2. **XGBoost $H=1$ Shrinkage Under-Convergence (`n_estimators=40, learning_rate=0.05`) vs. Recursive Compounding:**",
        "   - Even at `hourly` ($H=1$, where **no recursive rollout occurs**), XGBoost exhibits a large negative Macro Bias (`-7.671 kWh` at `lb=24h`, `-7.927 kWh` at `lb=48h`, `-7.965 kWh` at `lb=168h`; Macro MASE `1.609`, `1.498`, `1.482`). Because `XGBRegressor` initializes at `base_score = mean(y_train)` (diluted by 38.10% 2016 zeros) and executes `n_estimators = 40` trees at `learning_rate = 0.05`, $(1 - 0.05)^{40} = 0.1285$ of the initial downward offset remains unclosed even on strictly positive 2017 lag vectors. During 24-step and 168-step recursive rollouts, this 1-step underprediction compounds as underestimated predictions are fed back as subsequent lag features (`daily_profile` Bias `-21.350 kWh`, `weekly_profile` `-28.332 kWh` at `lb=168h`).",
        "   - By contrast, `random_forest` (`n_estimators=20, max_depth=10`) averages un-shrunk bootstrap regression trees that isolate positive lag vectors from zero lag vectors without global `base_score` shrinkage (`hourly` Bias `+0.374 kWh`, `daily_profile` Bias `-2.827 kWh` at `lb=168h`).",
        "3. **Strict `lag < lookback` Feature Selection (`feature_lags`):**",
        "   - `feature_lags(config, lookback)` filters `feature_lags_hours = [1, 2, 3, 6, 12, 24, 48, 168]` using strict inequality (`lag < lookback`):",
        "     - `lookback = 24h` uses only `[1, 2, 3, 6, 12]` (**excluding `lag_24`**), explaining why `random_forest` underperforms `persistence` at $H=1$ (`MAE 7.492` vs. `7.191 kWh`) and has higher $H=24$ error (`MAE 18.365 kWh`).",
        "     - `lookback = 48h` uses `[1, 2, 3, 6, 12, 24]` (adding diurnal `lag_24`), driving a sharp improvement in `random_forest` (`hourly` MAE `6.772 kWh`, `daily_profile` MAE `14.424 kWh`, `weekly_profile` MAE `19.069 kWh`).",
        "     - `lookback = 168h` uses `[1, 2, 3, 6, 12, 24, 48]` (**excluding `lag_168`** because `168 < 168` is false).",
        "",
        "---",
        "",
        "## 5. Unsupervised Anomaly Detection & Remediation Findings (Phase 3B.3)",
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
        "## 6. Regime-Shift Analysis & Cross-Building Heterogeneity (Phase 3C Remediated)",
        "",
        "### 6.1 Campus-Wide Zero Shift vs. Building-Specific Positive Load Shifts",
        f"- **Campus-Wide Effect:** All 29 core buildings drop from `{zs['core_train_valid_zero_per_1000_macro_mean']:.2f}` valid zero hours per 1,000h in 2016 (`median {zs['core_train_valid_zero_per_1000_median']:.2f}/1,000h`, range `[{zs['core_train_valid_zero_per_1000_range'][0]:.2f}, {zs['core_train_valid_zero_per_1000_range'][1]:.2f}]`) to `0.00/1,000h` in H1 2017. This campus-wide shift depresses 2016 unconditional training means ($\\mu_{{2016}}$) and shifts 2016 anomaly score quantiles relative to active H1 2017 operations.",
        f"- **Cross-Building Variation:** Because 2016 valid-zero rates are nearly uniform across the 29 buildings (`20/29` within `384.0–384.8/1,000h`), 2016 zero frequency does **not** significantly explain cross-building differences in daily Random Forest MASE degradation (**Pearson $r = {c_zero['pearson_r']:.4f}$, $p = {c_zero['pearson_p_value']:.4f}$; Spearman $\\rho = {c_zero['spearman_rho']:.4f}$, $p = {c_zero['spearman_p_value']:.4f}$**). Instead, building-to-building differences in forecasting accuracy and anomaly alert rates reflect building-specific changes in positive active-load distributions and diurnal ramp amplitude.",
        "",
        "### 6.2 Exploratory Cross-Phase Correlation & Outlier-Leverage Audit",
        f"- **Daily Random Forest MASE Degradation vs. Seasonal-Deviation LOF Alert Rate (`exploratory_outlier_sensitive_diagnostic`):**",
        f"  - **$N=29$ Linear Correlation:** Pearson $r = {c_lof['pearson_r']:.4f}$ ($p = {c_lof['pearson_p_value']:.4e}$).",
        f"  - **$N=29$ Non-Parametric Rank Correlation:** Spearman $\\rho = {c_lof['spearman_rho']:.4f}$ ($p = {c_lof['spearman_p_value']:.4f}$); Kendall $\\tau = {c_lof['kendall_tau']:.4f}$ ($p = {c_lof['kendall_p_value']:.4f}$).",
        f"  - **Single-Outlier Leverage (`Panther_education_Vincent`):** `Panther_education_Vincent` (`daily_rf_mase_delta = +{c_lof['influential_outlier_mase_delta']:.4f}`, `seasonal_lof = {c_lof['influential_outlier_seasonal_lof_per_1000h']:.2f}/1,000h`) has OLS hat-matrix leverage **$h = {c_lof['influential_outlier_leverage_h']:.4f}$** and **Cook's $D = {c_lof['influential_outlier_cooks_d']:.4f}$**.",
        f"  - **Excluding `Panther_education_Vincent` ($N={c_lof['excluding_vincent_n']}$):** Pearson $r = {c_lof['excluding_vincent_pearson_r']:.4f}$ ($p = {c_lof['excluding_vincent_pearson_p_value']:.4f}$, non-significant); Spearman $\\rho = {c_lof['excluding_vincent_spearman_rho']:.4f}$ ($p = {c_lof['excluding_vincent_spearman_p_value']:.4f}$).",
        f"  - **Diagnostic Conclusion:** Pearson's apparent positive association is dominated by one influential building (`Panther_education_Vincent`) and is **not** a robust general relationship across the 29 buildings (where the 10 RF-winning buildings have a median seasonal LOF rate of `{c_lof['rf_winning_buildings_seasonal_lof_median_per_1000h']:.2f}/1,000h` vs. `{c_lof['baseline_winning_buildings_seasonal_lof_median_per_1000h']:.2f}/1,000h` for the 19 baseline-winning buildings). No causal relationship is implied or supported.",
        f"- **Daily Random Forest Absolute Bias vs. MAE Degradation ($N=29$):** Pearson $r = {c_bias['pearson_r']:.4f}$ ($p = {c_bias['pearson_p_value']:.4f}$); Spearman $\\rho = {c_bias['spearman_rho']:.4f}$ ($p = {c_bias['spearman_p_value']:.4e}$) — descriptive error-decomposition diagnostic reflecting $|\\text{{mean\\_error}}| \\le \\text{{MAE}}$.",
        "",
        "### Table 7: Per-Building Integrated Diagnostic Profile (29 Core Buildings, `lookback=168h`, `1.0pct` Operating Point)",
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
        "## 7. Unsupported Claims, Scientific Limitations & Threats to Validity",
        "",
        "### 7.1 Claims Explicitly Not Supported by the Completed Experiments",
        "1. **No Confirmed Fault Detection or Supervised Anomaly Accuracy:** No ground-truth anomaly labels exist in BDG2 Panther (`ground_truth_anomaly_labels == 'none'`). Flagged hours and contiguous events represent unlabeled statistical or geometric departures from 2016 training distributions, and supervised metrics (precision, recall, F1, ROC-AUC, accuracy) cannot be claimed.",
        "2. **No Universal Forecasting Winner Across Horizons:** Random Forest (`lookback=168h`) improves upon persistence at $H=1$ (`MASE 0.937` vs. `1.017`), whereas daily seasonal naive (`MASE 1.644`) and weekly seasonal naive (`MASE 1.937`) outperform all frozen-origin recursive candidates in aggregate macro metrics at $H=24$ and $H=168$.",
        "3. **No Robust Cross-Building Linear or Causal Coupling Between Forecasting Degradation and LOF Alert Rates:** The $N=29$ Pearson correlation ($r = 0.6906$) between `daily_rf_mase_delta` and `seasonal_deviation` `lof_novelty` validation alert rate is dominated by `Panther_education_Vincent` ($h = 0.8507$, Cook's $D = 1.2558$; $N=28$ Pearson $r = 0.2518, p = 0.1962$; $N=29$ Spearman $\\rho = -0.1466, p = 0.4480$).",
        "4. **No Attribution of All Forecasting Error to the 2016 Zero Regime Alone:** Multi-step errors in Phase 3B.2 are co-determined by model-configuration choices (`SARIMA(1,0,0)x(0,0,0,0)` non-seasonal AR(1) mean-reversion, `XGBoost(n_estimators=40, lr=0.05)` $H=1$ under-convergence, and strict `lag < lookback` lag filtering) and building-specific changes in positive active-load distributions (`2016` zero rate vs. `daily_rf_mase_delta`: Pearson $r = 0.2330, p = 0.2239$).",
        "5. **No Weather-Exogenous Forecasting Uplift Claim:** Although complete-case weather-aware forecasting rules were specified in Phase 2.7/3A, Phases 3B.1 and 3B.2 evaluated electricity-only baselines and candidates; weather-augmented forecasting remains an unexecuted future extension.",
        "6. **No Held-Out Test Partition (`H2 2017`) or Multi-Year Generalization Claim:** The held-out test partition (`2017-07-01 00:00:00` through `2017-12-31 23:00:00`) remains untouched (`not_run`). Findings are descriptive of H1 2017 validation on the 29 UCF Panther educational buildings.",
        "",
        "---",
        "",
        "## 8. Reproducibility & Audit Trail Summary",
        "",
        f"- **Verified Runtime Environment:** Python `{env['python_version']}` on `{env['platform']}` (`numpy=={pkg['numpy']}`, `pandas=={pkg['pandas']}`, `scipy=={pkg['scipy']}`, `scikit-learn=={pkg['scikit-learn']}`, `statsmodels=={pkg['statsmodels']}`, `xgboost=={pkg['xgboost']}`, `joblib=={pkg['joblib']}`, `pytest=={pkg['pytest']}`).",
        f"- **Protected Artifact Ledger:** `{counts['total_protected_files']}` cryptographically verified protected files (`{counts['protected_datasets']}` datasets, `{counts['protected_configs']}` configurations, `{counts['accepted_reports']}` accepted JSON reports, `{counts['historical_reports']}` historical/superseded JSON reports, `{counts['protected_docs']}` specification and synthesis Markdown documents) plus `{counts['prior_phase3d_outputs']}` preserved initial Phase 3D (`3D-final-v1`) outputs.",
        "- **In-Memory Deterministic Reproduction (Phases 3A, 3B.1, 3C):** Verified 100% agreement between source modules executed in memory and accepted reports on disk for **Phase 3A** (`validate_canonical`, `coverage_report`: `783` total coverage rows across `train`, `validation`, and `test`, comprising `522` train/validation rows summarized in Table 1 and `261` test-partition timestamp/missingness coverage-only rows with `test_evaluation == 'not_run'`), **Phase 3B.1** (`evaluate_validation`: `261` baseline validation rows and `9` macro summary rows), and **Phase 3C** (`build_phase3c_synthesis`, `render_markdown_report`: `3C-remediated-v1`).",
        "- **Cross-Report Structural, Mask & Leakage Invariant Verification (Phases 3B.2 & 3B.3):** Phases 3B.2 and 3B.3 were **not** re-trained or re-executed in memory during Phase 3D verification; instead, `verify_phase3b2_and_3b3_invariants()` verifies on the accepted report artifacts (`phase3b2_candidate_recursive_validation.json` with `783` candidate rows; `phase3b3_anomaly_remediated_validation.json` with `522` core + `18` `Quintin` sensitivity rows) that `status == 'validation_only'`, `test_evaluation == 'not_run'`, `random_seed == 42`, `n_jobs == 1` (`random_forest`, `xgboost`, `isolation_forest`, `lof_novelty`), exact baseline-to-candidate origin/target eligibility-mask parity across all `783` Phase 3B.2 rows, exact cross-detector eligibility-mask parity across all `60` `(building_id, feature_mode)` cells (`540` rows) in Phase 3B.3, threshold and flagged-hour monotonicity across `0.5pct`, `1.0pct`, and `2.0pct` across all `180` `(building_id, feature_mode, model)` series (`0.5pct > 1.0pct > 2.0pct` strictly for `isolation_forest` and `lof_novelty`, and `0.5pct >= 1.0pct >= 2.0pct` for `statistical_zscore`), `validation_flagged_zero_hours == 0`, conditional event duration rules, and absence of supervised anomaly metric keys.",
        "- **Verification Command:** `.venv/bin/pytest -v && .venv/bin/python scripts/run_phase3d_reproducibility.py --verify-only`.",
        "",
    ])
    return "\n".join(lines)


def render_reproducibility_playbook(manifest):
    env = manifest["execution"]
    pkg = env["package_versions"]
    ledger = manifest["checksum_ledger"]
    lineage = manifest["report_lineage"]

    lines = [
        "# CEI Reproducibility Playbook & Cryptographic Provenance Guide (Phase 3D Remediated)",
        "",
        f"**Manifest Version:** `{manifest['report_version']}` | **Verification Status:** `{manifest['status']}` | **Held-Out Test Partition:** `{manifest['test_evaluation']}`",
        "",
        "---",
        "",
        "## 1. Environment & Dependency Specification",
        "",
        "### 1.1 Tested Reference Environment",
        f"- **Python Version:** `{env['python_version']}` (`requires-python >= 3.11`)",
        f"- **Operating System / Platform:** `{env['platform']}`",
        "- **Pinned Reference Package Versions:**",
        f"  - `numpy=={pkg['numpy']}`",
        f"  - `pandas=={pkg['pandas']}`",
        f"  - `scipy=={pkg['scipy']}`",
        f"  - `scikit-learn=={pkg['scikit-learn']}`",
        f"  - `statsmodels=={pkg['statsmodels']}`",
        f"  - `xgboost=={pkg['xgboost']}`",
        f"  - `joblib=={pkg['joblib']}`",
        f"  - `pytest=={pkg['pytest']}`",
        "",
        "### 1.2 Environment Setup Commands",
        "```bash",
        "uv python install 3.11",
        "uv venv --python 3.11",
        "uv pip install --python .venv/bin/python -e . -r requirements.txt",
        "```",
        "",
        "---",
        "",
        "## 2. Non-Destructive Reproducibility Verification Workflow (Recommended)",
        "",
        "All phase execution scripts (`run_phase3a_validation.py`, `run_phase3b_baselines.py`, `run_phase3b2_candidates.py`, `run_phase3b3_anomaly.py`, `run_phase3c_synthesis.py`, and `run_phase3d_reproducibility.py`) open their output files in exclusive creation mode (`open('x')`) to guarantee that accepted reports are never overwritten.",
        "",
        "To run the full automated test suite (`~2–3 minutes` measured on reference Apple Silicon) and cryptographically + deterministically verify all **28 protected repository artifacts**, the **3 preserved initial Phase 3D (`3D-final-v1`) outputs**, and the **3 remediated Phase 3D (`3D-remediated-v1`) outputs** without modifying any file on disk (`~2 minutes` for `--verify-only`):",
        "",
        "```bash",
        ".venv/bin/pytest -v",
        ".venv/bin/python scripts/run_phase3d_reproducibility.py --verify-only",
        "```",
        "",
        "### What `--verify-only` Checks:",
        "1. **SHA-256 Checksum Integrity:** Verifies all 6 raw/canonical dataset CSVs, 6 JSON experiment configurations, 8 accepted JSON reports, 5 historical/superseded JSON reports, 3 Markdown specification/synthesis documents, 3 preserved initial Phase 3D (`3D-final-v1`) outputs, and the 3 remediated Phase 3D (`3D-remediated-v1`) outputs.",
        "2. **In-Memory Deterministic Reproduction (Phases 3A, 3B.1, 3C):** Re-executes Phase 3A (`validate_canonical`, `coverage_report`: `783` total coverage rows = `522` train/validation + `261` test coverage-only rows), Phase 3B.1 (`evaluate_validation`: `261` baseline rows), and Phase 3C (`build_phase3c_synthesis`, `render_markdown_report`) in memory and asserts 100% equality with the accepted files on disk.",
        "3. **Cross-Report Structural, Mask & Leakage Invariant Verification (Phases 3B.2 & 3B.3):** Without re-running model training in memory, verifies directly on `phase3b2_candidate_recursive_validation.json` (`783` candidate rows) and `phase3b3_anomaly_remediated_validation.json` (`522` core + `18` `Quintin` rows) that `status == 'validation_only'`, `test_evaluation == 'not_run'`, `random_seed == 42`, `n_jobs == 1` (`random_forest`, `xgboost`, `isolation_forest`, `lof_novelty`), exact baseline-to-candidate origin/target eligibility-mask parity (`783` rows), exact cross-detector eligibility-mask parity across all `60` `(building_id, feature_mode)` cells (`540` rows), threshold and flagged-hour monotonicity across `0.5pct`, `1.0pct`, and `2.0pct` across all `180` `(building_id, feature_mode, model)` series, `Panther_education_Quintin` isolation, `validation_flagged_zero_hours == 0`, conditional event durations, and absence of supervised anomaly metric keys.",
        "",
        "---",
        "",
        "## 3. Clean-Room End-to-End Execution Order (For Fresh Workspace Only)",
        "",
        "If reproducing the pipeline from scratch in a clean workspace where `data/reports/bdg2/panther/` is empty, run the scripts in the following exact sequence:",
        "",
        "| Step | Phase | Command | Determinism Settings | Accepted Output Written (`open('x')`) |",
        "| :---: | :---: | :--- | :--- | :--- |",
        "| 1 | **2.6** | `.venv/bin/python scripts/migrate_bdg2.py` | SHA-256 verified Git LFS stream | `data/processed/bdg2/panther/*.csv`, `dataset_manifest.json`, `migration_report.json` |",
        "| 2 | **3A** | `.venv/bin/python scripts/run_phase3a_validation.py` | Deterministic timestamp scan | `data/reports/bdg2/panther/phase3a_corrected_validation.json` |",
        "| 3 | **3B.1** | `.venv/bin/python scripts/run_phase3b_baselines.py` | Deterministic seasonal naive | `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json` |",
        "| 4 | **3B.2** | `.venv/bin/python scripts/run_phase3b2_candidates.py` | `random_seed=42`, `n_jobs=1` | `data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json` |",
        "| 5 | **3B.3** | `.venv/bin/python scripts/run_phase3b3_anomaly.py` | `random_seed=42`, `n_jobs=1` | `data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json` |",
        "| 6 | **3C** | `.venv/bin/python scripts/run_phase3c_synthesis.py` | Read-only deterministic synthesis | `phase3c_integrated_research_synthesis_remediated.json`, `PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md` |",
        "| 7 | **3D** | `.venv/bin/python scripts/run_phase3d_reproducibility.py` | Read-only provenance & parity audit | `phase3d_reproducibility_manifest_remediated.json`, `PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md`, `REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md` |",
        "",
        "---",
        "",
        "## 4. Report Lineage & Audit Trail",
        "",
        "| Phase | Accepted Canonical Report(s) | Superseded Historical Report(s) Preserved on Disk | Generating Script |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for item in lineage:
        acc = ", ".join(f"`{p}`" for p in item["accepted_reports"])
        sup = "—" if not item["superseded_reports"] else ", ".join(f"`{p}`" for p in item["superseded_reports"])
        lines.append(f"| `{item['phase']}` | {acc} | {sup} | `{item['script']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Complete Protected & Historical SHA-256 Checksum Ledger",
        "",
        "| Category | Relative Path | SHA-256 Checksum |",
        "| :--- | :--- | :--- |",
    ])
    for group_name in (
        "protected_datasets",
        "protected_configs",
        "accepted_reports",
        "historical_reports",
        "protected_docs",
        "prior_phase3d_outputs",
    ):
        for rel_path, digest in sorted(ledger[group_name].items()):
            lines.append(f"| `{group_name}` | `{rel_path}` | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def verify_existing_phase3d_outputs(root_path, manifest_mem=None, syn_mem=None):
    root_path = Path(root_path)
    manifest_output = root_path / "data/reports/bdg2/panther/phase3d_reproducibility_manifest_remediated.json"
    final_report_output = root_path / "docs/PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md"
    playbook_output = root_path / "docs/REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md"

    if manifest_mem is None:
        manifest_mem = build_phase3d_manifest(root_path)
    if syn_mem is None:
        syn_mem = build_phase3c_synthesis(root_path)

    md_3c_mem = render_markdown_report(syn_mem)
    md_3c_disk = (root_path / "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md").read_text(encoding="utf-8")
    if md_3c_mem != md_3c_disk:
        raise ValueError("In-memory Phase 3C Markdown does not match PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md")

    manifest_disk = json.loads(manifest_output.read_text(encoding="utf-8"))
    m_mem_no_ts = {k: v for k, v in manifest_mem.items() if k != "execution"}
    m_disk_no_ts = {k: v for k, v in manifest_disk.items() if k != "execution"}
    if m_mem_no_ts != m_disk_no_ts:
        raise ValueError("In-memory Phase 3D manifest does not match phase3d_reproducibility_manifest_remediated.json")

    final_report_mem = render_final_research_report(syn_mem, manifest_disk)
    final_report_disk = final_report_output.read_text(encoding="utf-8")
    if final_report_mem != final_report_disk:
        raise ValueError("In-memory final research report does not match PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md")

    playbook_mem = render_reproducibility_playbook(manifest_disk)
    playbook_disk = playbook_output.read_text(encoding="utf-8")
    if playbook_mem != playbook_disk:
        raise ValueError("In-memory reproducibility playbook does not match REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md")

    return manifest_disk
