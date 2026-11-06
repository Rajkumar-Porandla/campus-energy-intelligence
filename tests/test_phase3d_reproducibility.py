"""Tests for Phase 3D reproducibility verification, provenance ledger, and final reporting."""
import copy
import json
from pathlib import Path

import pytest

from cei.reproducibility.phase3d_reproducibility import (
    ACCEPTED_REPORT_CHECKSUMS,
    HISTORICAL_REPORT_CHECKSUMS,
    PRIOR_PHASE3D_CHECKSUMS,
    PROTECTED_CONFIG_CHECKSUMS,
    PROTECTED_DATASET_CHECKSUMS,
    PROTECTED_DOC_CHECKSUMS,
    build_phase3d_manifest,
    render_final_research_report,
    render_reproducibility_playbook,
    verify_all_protected_checksums,
    verify_existing_phase3d_outputs,
    verify_phase3b2_and_3b3_report_objects,
    verify_prior_phase3d_checksums,
)
from cei.synthesis.phase3c_synthesis import build_phase3c_synthesis

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_OUTPUT = ROOT / "data/reports/bdg2/panther/phase3d_reproducibility_manifest_remediated.json"
FINAL_REPORT_OUTPUT = ROOT / "docs/PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md"
PLAYBOOK_OUTPUT = ROOT / "docs/REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md"


@pytest.fixture(scope="module")
def phase3d_bundle():
    """Compute Phase 3D manifest and Phase 3C synthesis once per test module to avoid redundant slow runs."""
    manifest = build_phase3d_manifest(ROOT)
    syn = build_phase3c_synthesis(ROOT)
    return {"manifest": manifest, "syn": syn}


def test_phase3d_verifies_all_28_protected_artifacts_prior_3d_and_requirements():
    verified = verify_all_protected_checksums(ROOT)
    assert verified["protected_datasets"] == PROTECTED_DATASET_CHECKSUMS
    assert verified["protected_configs"] == PROTECTED_CONFIG_CHECKSUMS
    assert verified["accepted_reports"] == ACCEPTED_REPORT_CHECKSUMS
    assert verified["historical_reports"] == HISTORICAL_REPORT_CHECKSUMS
    assert verified["protected_docs"] == PROTECTED_DOC_CHECKSUMS
    total = sum(len(v) for v in verified.values())
    assert total == 28

    prior_3d = verify_prior_phase3d_checksums(ROOT)
    assert prior_3d == PRIOR_PHASE3D_CHECKSUMS
    assert len(prior_3d) == 3

    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scipy>=" in req_text


def test_phase3d_in_memory_reproduction_and_cross_report_invariants(phase3d_bundle):
    manifest = phase3d_bundle["manifest"]
    checks = manifest["deterministic_reproduction_checks"]
    assert checks["phase3a_in_memory_parity"] == "passed"
    assert checks["phase3a_coverage_rows_verified"] == 783
    assert checks["phase3a_train_validation_coverage_rows"] == 522
    assert checks["phase3a_test_coverage_only_rows"] == 261
    assert checks["phase3a_timestamp_count"] == 17544
    assert checks["phase3a_core_building_count"] == 29
    assert checks["phase3b1_in_memory_parity"] == "passed"
    assert checks["phase3b1_baseline_rows_verified"] == 261
    assert checks["phase3c_in_memory_parity"] == "passed"
    assert checks["phase3c_report_version_verified"] == "3C-remediated-v1"

    assert checks["phase3b2_invariants"] == "passed"
    assert checks["phase3b2_candidate_rows_verified"] == 783
    assert checks["phase3b2_random_seed"] == 42
    assert checks["phase3b2_n_jobs"] == {"random_forest": 1, "xgboost": 1}
    assert checks["phase3b3_invariants"] == "passed"
    assert checks["phase3b3_core_rows_verified"] == 522
    assert checks["phase3b3_sensitivity_rows_verified"] == 18
    assert checks["phase3b3_mask_parity_cells_verified"] == 60
    assert checks["phase3b3_threshold_monotonicity_series_verified"] == 180
    assert checks["phase3b3_random_seed"] == 42
    assert checks["phase3b3_n_jobs"] == {"isolation_forest": 1, "lof_novelty": 1}


def test_phase3d_invariant_guards_reject_seed_njobs_mask_and_monotonicity_violations():
    p3b1 = json.loads(
        (ROOT / "data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json").read_text(encoding="utf-8")
    )
    p3b2 = json.loads(
        (ROOT / "data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json").read_text(encoding="utf-8")
    )
    p3b3 = json.loads(
        (ROOT / "data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json").read_text(encoding="utf-8")
    )

    # 1. Mutate Phase 3B.2 random_seed
    p3b2_bad_seed = copy.deepcopy(p3b2)
    p3b2_bad_seed["configuration"]["random_seed"] = 99
    with pytest.raises(ValueError, match="Phase 3B.2 configuration random_seed must equal 42"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2_bad_seed, p3b3)

    # 2. Mutate Phase 3B.2 n_jobs
    p3b2_bad_njobs = copy.deepcopy(p3b2)
    p3b2_bad_njobs["configuration"]["models"]["random_forest"]["n_jobs"] = 4
    with pytest.raises(ValueError, match="Phase 3B.2 model random_forest must enforce n_jobs == 1"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2_bad_njobs, p3b3)

    # 3. Mutate Phase 3B.3 n_jobs
    p3b3_bad_njobs = copy.deepcopy(p3b3)
    p3b3_bad_njobs["configuration"]["models"]["lof_novelty"]["n_jobs"] = -1
    with pytest.raises(ValueError, match="Phase 3B.3 model lof_novelty must enforce n_jobs == 1"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3_bad_njobs)

    # 4. Mutate Phase 3B.3 eligibility mask parity across models
    p3b3_bad_mask = copy.deepcopy(p3b3)
    p3b3_bad_mask["results"]["rows"][1]["validation_eligible_hours"] += 1
    with pytest.raises(ValueError, match="Phase 3B.3 eligibility mask mismatch across detectors"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3_bad_mask)

    # 5. Mutate Phase 3B.3 threshold monotonicity
    p3b3_bad_thresh = copy.deepcopy(p3b3)
    # Row 0 is 0.5pct, Row 1 is 1.0pct for first building/mode/detector
    p3b3_bad_thresh["results"]["rows"][0]["threshold_value"] = (
        p3b3_bad_thresh["results"]["rows"][1]["threshold_value"] - 0.5
    )
    with pytest.raises(ValueError, match="Phase 3B.3 threshold monotonicity violated"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3_bad_thresh)

    # 6. Mutate Phase 3B.3 flagged-hour monotonicity
    p3b3_bad_flags = copy.deepcopy(p3b3)
    p3b3_bad_flags["results"]["rows"][0]["validation_flagged_hours"] = (
        p3b3_bad_flags["results"]["rows"][1]["validation_flagged_hours"] + 5
    )
    with pytest.raises(ValueError, match="Phase 3B.3 validation flagged hours monotonicity violated"):
        verify_phase3b2_and_3b3_report_objects(p3b1, p3b2, p3b3_bad_flags)


def test_phase3d_deterministic_report_and_playbook_rendering(phase3d_bundle):
    manifest = phase3d_bundle["manifest"]
    syn = phase3d_bundle["syn"]

    assert manifest["phase"] == "3D"
    assert manifest["report_version"] == "3D-remediated-v1"
    assert manifest["status"] == "reproducibility_verified_validation_only"
    assert manifest["test_evaluation"] == "not_run"
    assert manifest["ground_truth_anomaly_labels"] == "none"
    assert manifest["protected_artifact_counts"]["total_protected_files"] == 28
    assert manifest["protected_artifact_counts"]["prior_phase3d_outputs"] == 3

    # Ensure JSON serializability with allow_nan=False
    serialized = json.dumps(manifest, sort_keys=True, allow_nan=False)
    assert len(serialized) > 1000

    syn_sorted = json.loads(json.dumps(syn, sort_keys=True))
    manifest_sorted = json.loads(serialized)

    report_md_1 = render_final_research_report(syn, manifest)
    report_md_2 = render_final_research_report(syn_sorted, manifest_sorted)
    assert report_md_1 == report_md_2

    playbook_md_1 = render_reproducibility_playbook(manifest)
    playbook_md_2 = render_reproducibility_playbook(manifest_sorted)
    assert playbook_md_1 == playbook_md_2

    # Check required scientific content in final report (including B1, B2, B3, F1, F2)
    for expected_substr in (
        "3D-remediated-v1",
        "3C-remediated-v1",
        "synthetic_campus_energy_usage.csv",
        "Admin Office",
        "Classroom Block",
        "Computer Lab",
        "Hostel",
        "Library",
        "355–360 days",
        "4–385 kWh",
        "`High` (`114` rows), `Medium` (`99` rows), and `Low` (`87` rows)",
        "Cross-Report Structural, Mask & Leakage Invariant Verification",
        "were **not** re-trained or re-executed in memory during Phase 3D",
        "522` train/validation rows",
        "261` test-partition timestamp/missingness coverage-only rows",
        "joblib==",
        "0.6906",
        "-0.1466",
        "0.4480",
        "0.2518",
        "0.1962",
        "0.8507",
        "1.2558",
        "(1,0,0)x(0,0,0,0)",
        "n_estimators=40",
        "learning_rate=0.05",
        "lag < lookback",
        "0.2330",
        "0.2239",
        "0.3286",
        "0.0818",
        "not_run",
    ):
        assert expected_substr in report_md_1

    # Check playbook content (including F2, F3)
    assert "| Step | Phase | Command | Determinism Settings | Accepted Output Written (`open('x')`) |" in playbook_md_1
    assert "joblib==" in playbook_md_1
    assert "~2 minutes" in playbook_md_1
    assert "~2–3 minutes" in playbook_md_1
    assert "< 30 seconds" not in playbook_md_1

    # Ensure fabricated B1/B2 claims are completely absent
    for forbidden_substr in (
        "demonstrating that both tasks are governed by",
        "synthetic_campus_energy.csv",
        "Engineering",
        "Science",
        "Dormitory",
        "Student Center",
        "45 to 62 days",
        "up to 480 kWh",
        "Normal, High, Peak",
        "Re-executing Phases 3A, 3B.1, and 3C in memory and checking all",
    ):
        assert forbidden_substr not in report_md_1

    if MANIFEST_OUTPUT.exists() and FINAL_REPORT_OUTPUT.exists() and PLAYBOOK_OUTPUT.exists():
        verify_existing_phase3d_outputs(ROOT, manifest_mem=manifest, syn_mem=syn)
