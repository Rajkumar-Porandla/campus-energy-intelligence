# CEI Reproducibility Playbook & Cryptographic Provenance Guide (Phase 3D)

**Manifest Version:** `3D-final-v1` | **Verification Status:** `reproducibility_verified_validation_only` | **Held-Out Test Partition:** `not_run`

---

## 1. Environment & Dependency Specification

### 1.1 Tested Reference Environment
- **Python Version:** `3.11.15` (`requires-python >= 3.11`)
- **Operating System / Platform:** `macOS-27.0.1-arm64-arm-64bit`
- **Pinned Reference Package Versions:**
  - `numpy==2.4.6`
  - `pandas==3.1.0`
  - `scipy==1.17.1`
  - `scikit-learn==1.9.1`
  - `statsmodels==0.15.0`
  - `xgboost==3.2.0`
  - `joblib==1.6.0`
  - `pytest==8.4.2`

### 1.2 Environment Setup Commands
```bash
uv python install 3.11
uv venv --python 3.11
uv pip install --python .venv/bin/python -e . -r requirements.txt
```

---

## 2. Non-Destructive Reproducibility Verification Workflow (Recommended)

All phase execution scripts (`run_phase3a_validation.py`, `run_phase3b_baselines.py`, `run_phase3b2_candidates.py`, `run_phase3b3_anomaly.py`, `run_phase3c_synthesis.py`, and `run_phase3d_reproducibility.py`) open their output files in exclusive creation mode (`open('x')`) to guarantee that accepted reports are never overwritten.

To run the full automated test suite and cryptographically + deterministically verify all **28 protected repository artifacts** and the **3 Phase 3D outputs** without modifying any file on disk (`< 15 seconds`):

```bash
.venv/bin/pytest -v
.venv/bin/python scripts/run_phase3d_reproducibility.py --verify-only
```

### What `--verify-only` Checks:
1. **SHA-256 Checksum Integrity:** Verifies all 6 raw/canonical dataset CSVs, 6 JSON experiment configurations, 8 accepted JSON reports, 5 historical/superseded JSON reports, 3 Markdown specification/synthesis documents, and the 3 Phase 3D outputs.
2. **In-Memory Deterministic Reproduction:** Re-executes Phase 3A (`validate_canonical`, `coverage_report`), Phase 3B.1 (`evaluate_validation`), and Phase 3C (`build_phase3c_synthesis`, `render_markdown_report`) in memory and asserts 100% equality with the accepted files on disk.
3. **Cross-Report Invariant & Leakage Guardrails:** Verifies that Phase 3B.2 (`phase3b2_candidate_recursive_validation.json`) and Phase 3B.3 (`phase3b3_anomaly_remediated_validation.json`) preserve `status == 'validation_only'`, `test_evaluation == 'not_run'`, `random_seed == 42`, `n_jobs == 1`, shared origin/target masks, `Panther_education_Quintin` isolation, `validation_flagged_zero_hours == 0`, conditional event durations, and absence of supervised anomaly metric keys.

---

## 3. Clean-Room End-to-End Execution Order (For Fresh Workspace Only)

If reproducing the pipeline from scratch in a clean workspace where `data/reports/bdg2/panther/` is empty, run the scripts in the following exact sequence:

| Step | Phase | Command |Determinism Settings | Accepted Output Written (`open('x')`) |
| :---: | :---: | :--- | :--- | :--- |
| 1 | **2.6** | `.venv/bin/python scripts/migrate_bdg2.py` | SHA-256 verified Git LFS stream | `data/processed/bdg2/panther/*.csv`, `dataset_manifest.json`, `migration_report.json` |
| 2 | **3A** | `.venv/bin/python scripts/run_phase3a_validation.py` | Deterministic timestamp scan | `data/reports/bdg2/panther/phase3a_corrected_validation.json` |
| 3 | **3B.1** | `.venv/bin/python scripts/run_phase3b_baselines.py` | Deterministic seasonal naive | `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json` |
| 4 | **3B.2** | `.venv/bin/python scripts/run_phase3b2_candidates.py` | `random_seed=42`, `n_jobs=1` | `data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json` |
| 5 | **3B.3** | `.venv/bin/python scripts/run_phase3b3_anomaly.py` | `random_seed=42`, `n_jobs=1` | `data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json` |
| 6 | **3C** | `.venv/bin/python scripts/run_phase3c_synthesis.py` | Read-only deterministic synthesis | `phase3c_integrated_research_synthesis_remediated.json`, `PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md` |
| 7 | **3D** | `.venv/bin/python scripts/run_phase3d_reproducibility.py` | Read-only provenance & parity audit | `phase3d_reproducibility_manifest.json`, `PHASE_3D_FINAL_RESEARCH_REPORT.md`, `REPRODUCIBILITY_PLAYBOOK.md` |

---

## 4. Report Lineage & Audit Trail

| Phase | Accepted Canonical Report(s) | Superseded Historical Report(s) Preserved on Disk | Generating Script |
| :---: | :--- | :--- | :--- |
| `2.6` | `data/reports/bdg2/panther/dataset_manifest.json`, `data/reports/bdg2/panther/migration_report.json` | — | `scripts/migrate_bdg2.py` |
| `3A` | `data/reports/bdg2/panther/phase3a_corrected_validation.json`, `data/reports/bdg2/panther/phase3a_final_validation.json` | — | `scripts/run_phase3a_validation.py` |
| `3B.1` | `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json` | `data/reports/bdg2/panther/phase3b_baseline_validation.json`, `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation.json` | `scripts/run_phase3b_baselines.py` |
| `3B.2` | `data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json` | `data/reports/bdg2/panther/phase3b2_candidate_validation.json` | `scripts/run_phase3b2_candidates.py` |
| `3B.3` | `data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json` | `data/reports/bdg2/panther/phase3b3_anomaly_validation.json` | `scripts/run_phase3b3_anomaly.py` |
| `3C` | `data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json`, `docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md` | `data/reports/bdg2/panther/phase3c_integrated_research_synthesis.json`, `docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS.md` | `scripts/run_phase3c_synthesis.py` |

---

## 5. Complete 28-Artifact SHA-256 Checksum Ledger

| Category | Relative Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| `protected_datasets` | `data/processed/bdg2/panther/electricity_core.csv` | `13cb0556da3a56fbd0eaacc6177c9e078ad42bc02e092f40df4b5dabc2881c8d` |
| `protected_datasets` | `data/processed/bdg2/panther/electricity_sensitivity_quintin.csv` | `302076f87ff6fd7423ac97cf3783575584a28b055d047d0c323ead892307a14e` |
| `protected_datasets` | `data/processed/bdg2/panther/weather.csv` | `bf34dd1fb1fbc5c00754d1869c4388c9353dac8d19094288dcac7739510f50cb` |
| `protected_datasets` | `data/raw/bdg2/panther/electricity.csv` | `039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418` |
| `protected_datasets` | `data/raw/bdg2/panther/metadata.csv` | `992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02` |
| `protected_datasets` | `data/raw/bdg2/panther/weather.csv` | `a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd` |
| `protected_configs` | `src/cei/config/bdg2_panther.json` | `7796fdd7e117f4f78615b49858bcf5ec0a41d00db1e00e04845ac65115010c1c` |
| `protected_configs` | `src/cei/config/dataset_schema.json` | `4d2af6c3a74bd11ff0bf287ca80769cbec2356f777af32fa689000d12cda8c55` |
| `protected_configs` | `src/cei/config/phase3a_experiment.json` | `a6583b89556848f01a249f9dd6f5c4b225aea49a7fcd484a2081c0afe1615856` |
| `protected_configs` | `src/cei/config/phase3b2_candidate_experiment.json` | `299f7314907ef4c10a2fe17fb0fd9fa8069ae66eae59bc8ad7d12ce6b186dab2` |
| `protected_configs` | `src/cei/config/phase3b3_anomaly_experiment.json` | `e1a158c7846ec71a199eb794a2c1ce86724bdfeb7529cb850efae9af7a2a1607` |
| `protected_configs` | `src/cei/config/phase3b_baseline_experiment.json` | `cc67f112d05b8b58188c27e94569840bb1ad5e47978ca0e36a263dc128f687ad` |
| `accepted_reports` | `data/reports/bdg2/panther/dataset_manifest.json` | `4265e095d450937e880f4c8c643722d461cb5050ae6ca0d3ec05030f8fddde5f` |
| `accepted_reports` | `data/reports/bdg2/panther/migration_report.json` | `0885019fef2f6823b0ae47edfd5145598218cead7714b4d26aeaa685d56cdd42` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3a_corrected_validation.json` | `017e0c7c3dbd9e1fbd7a882c8833b46975c7f03b26873d9e8c99b38acbc38228` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3a_final_validation.json` | `845c73e429fa3bd38e50f53d71961e2e57be5167246a04e7de968647e7e141c3` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3b2_candidate_recursive_validation.json` | `d15f06e47c163642934484361600d010d28401624b7c9070494bd53ba311e24f` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3b3_anomaly_remediated_validation.json` | `e400244a4db4209dd4fdd4ab544c2df48427dd876b0b04d06c4a3a44c44e2575` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation_v2.json` | `95c33e83b1dc374bed4e25a6cfc85c55e8415a88ff12c2a5fd8afeeeb63a978e` |
| `accepted_reports` | `data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json` | `66fa3fdceb99545cc8fa479aa3a7144075674f794f90fee823881832e681bcd1` |
| `historical_reports` | `data/reports/bdg2/panther/phase3b2_candidate_validation.json` | `e0fa9784703e72cca658947777c9966ceb211be779eead4979b442890c91faa0` |
| `historical_reports` | `data/reports/bdg2/panther/phase3b3_anomaly_validation.json` | `be9f621fd1fc3851d1e8fab69bc56a69a70304ff604ebee053eddae3f9fdb309` |
| `historical_reports` | `data/reports/bdg2/panther/phase3b_baseline_mase_audit_validation.json` | `061c6662963a8ff4b9b945f6c0ca39ed19194eebe3e182de6dde834147700520` |
| `historical_reports` | `data/reports/bdg2/panther/phase3b_baseline_validation.json` | `386be9922c24829c795d21851b667675ca9f96f657dc1338930f108665392e29` |
| `historical_reports` | `data/reports/bdg2/panther/phase3c_integrated_research_synthesis.json` | `4599210c9deb9206d5e6c1b82dd2700d4a0e926ac8227ed7d105ad849d830552` |
| `protected_docs` | `docs/PHASE_2_7_ML_EXPERIMENTATION_SPEC.md` | `d3246d3ede70323564cdb49059448c7ee94425c2dcc270cb3555b01ff0c39ce6` |
| `protected_docs` | `docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS.md` | `b78f2c23708e22eff7724f7d70b9721b3d7aa87a3c9921e52f7bf6470aedc05e` |
| `protected_docs` | `docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md` | `9d94cd49e9e030c8a268769bef11e0b84895dc198f1a87132fa725704b05d880` |
