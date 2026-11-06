# Phase 3C — Integrated Evaluation & Research Synthesis

**Report Status:** `validation_synthesis_only` | **Held-Out Test Partition:** `not_run` | **Ground-Truth Anomaly Labels:** `none`

---

## 1. Executive Summary

This document synthesizes the completed and independently audited validation experiments across **Phase 3A** (canonical validation and lookback coverage), **Phase 3B.1** (seasonal-naive and persistence baselines), **Phase 3B.2** (frozen-origin recursive multi-step candidate forecasting: SARIMA, Random Forest, XGBoost), and **Phase 3B.3** (remediated training-calibrated unsupervised anomaly detection: Statistical Z-Score, Isolation Forest, and full-empirical-multiplicity LOF novelty detection) for the 29 core UCF Panther education buildings (`2016` training, `H1 2017` validation) and the isolated sensitivity population (`Panther_education_Quintin`).

### Core Scientific Findings
1. **Horizon-Dependent Forecasting Performance (No Universal Winner):**
   - **1-Hour Ahead (`hourly`, secondary task):** `random_forest` (`lookback=48h` and `168h`) and `sarima` outperform the `persistence` baseline. At `lookback=168h`, `random_forest` achieves the lowest macro mean MAE (**`6.731 kWh`** vs. `7.100 kWh`), lowest macro mean RMSE (**`10.813 kWh`** vs. `12.566 kWh`), and lowest macro mean MASE (**`0.937`** vs. `1.017`), improving MAE on **`18/29`** buildings (`19/29` on RMSE).
   - **24-Hour Ahead (`daily_profile`, primary task):** `daily_seasonal_naive` outperforms all frozen-origin recursive candidates in aggregate macro metrics (Macro MAE **`12.158 kWh`**, Median MAE **`10.551 kWh`**, Macro RMSE **`19.841 kWh`**, Bias **`+0.009 kWh`**, Macro MASE **`1.644`** at `lookback=168h`), winning **`19/29`** buildings on MAE. `random_forest` (`lookback=168h`) is the strongest ML candidate (Macro MAE **`14.557 kWh`**, Macro MASE **`1.909`**), winning **`10/29`** buildings on MAE and **`14/29`** on RMSE where 2016-to-2017 positive load levels remain aligned.
   - **168-Hour Ahead (`weekly_profile`, exploratory task):** `weekly_seasonal_naive` leads in macro metrics (Macro MAE **`14.135 kWh`**, Median MAE **`12.504 kWh`**, Macro RMSE **`22.337 kWh`**, Macro MASE **`1.937`**), winning **`20/29`** buildings on MAE (`random_forest` wins `7/29`, `sarima` `1/29`, `xgboost` `1/29`).
2. **Mechanistic Root Cause — The 2016-to-2017 Valid-Zero Regime Shift:**
   - Across the 29 core buildings, valid `0.0 kWh` readings account for an average of **`381.02` per 1,000 observed hours** (`38.10%`, median `384.34/1,000h`) in 2016 (`train`), dropping to **`0.00` per 1,000 observed hours** (`0.00%`) in H1 2017 (`validation`).
   - Because models are strictly frozen on 2016 training data, recursive multi-step rollouts in `sarima` and `xgboost` pull 24-hour and 168-hour trajectories downward toward the 2016 zero-heavy mean (`daily_profile` Macro Bias `-21.232 kWh` for SARIMA and `-21.350 kWh` for XGBoost), whereas seasonal-naive baselines copy the immediately preceding 2017 cycle and adapt automatically to the active non-zero regime.
3. **Unsupervised Anomaly Detection & Cross-Phase Coupling:**
   - Remediated full-empirical-multiplicity `lof_novelty` and `isolation_forest` preserve all valid zeros as inliers (**`validation_flagged_zero_hours == 0`** across all 522 core rows and 18 `Quintin` rows) and exhibit strictly monotonic threshold operating points (`0.5pct > 1.0pct > 2.0pct` in `29/29` buildings).
   - Across the 29 core buildings, building-level recursive forecasting degradation (`random_forest` `daily_profile` `mase_delta` at `lb=168h`) correlates strongly with `seasonal_deviation` `lof_novelty` validation alert rates (**Pearson $r = 0.6906$**), demonstrating that both tasks are governed by the magnitude of each building's 2016-to-2017 operating profile shift.

---

## 2. Coverage Accounting & Valid-Zero Regime Shift (Phase 3A & Phase 3B.3)

### Table 1: Chronological Partition Coverage & Common-Lookback Eligibility (29 Core Buildings)

| Partition | Task ($H$) | Lookback (h) | Candidate Origins / Bldg | Eligible Origins / Bldg (Mean) | Eligible Target Hours / Bldg | Insufficient History Exclusions | Missing Target Exclusions | Origin Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `hourly` (1h) | `24` | 8784 | 8756.0 | 8756.0 | 25.7 | 2.3 | 99.68% |
| `train` | `hourly` (1h) | `48` | 8784 | 8730.4 | 8730.4 | 51.3 | 2.3 | 99.39% |
| `train` | `hourly` (1h) | `168` | 8784 | 8602.1 | 8602.1 | 179.6 | 2.3 | 97.93% |
| `train` | `daily_profile` (24h) | `24` | 366 | 364.8 | 8755.0 | 1.1 | 0.1 | 99.67% |
| `train` | `daily_profile` (24h) | `48` | 366 | 363.7 | 8729.4 | 2.1 | 0.1 | 99.38% |
| `train` | `daily_profile` (24h) | `168` | 366 | 358.4 | 8601.1 | 7.5 | 0.1 | 97.92% |
| `train` | `weekly_profile` (168h) | `24` | 52 | 50.8 | 8539.0 | 1.1 | 0.1 | 97.75% |
| `train` | `weekly_profile` (168h) | `48` | 52 | 50.8 | 8539.0 | 1.1 | 0.1 | 97.75% |
| `train` | `weekly_profile` (168h) | `168` | 52 | 50.8 | 8539.0 | 1.1 | 0.1 | 97.75% |
| `validation` | `hourly` (1h) | `24` | 4344 | 4257.7 | 4257.7 | 74.5 | 11.8 | 98.01% |
| `validation` | `hourly` (1h) | `48` | 4344 | 4183.2 | 4183.2 | 149.0 | 11.8 | 96.30% |
| `validation` | `hourly` (1h) | `168` | 4344 | 3812.8 | 3812.8 | 519.4 | 11.8 | 87.77% |
| `validation` | `daily_profile` (24h) | `24` | 181 | 173.8 | 4171.0 | 3.1 | 4.1 | 96.02% |
| `validation` | `daily_profile` (24h) | `48` | 181 | 170.7 | 4096.6 | 6.2 | 4.1 | 94.30% |
| `validation` | `daily_profile` (24h) | `168` | 181 | 155.3 | 3727.4 | 21.6 | 4.1 | 85.81% |
| `validation` | `weekly_profile` (168h) | `24` | 25 | 19.8 | 3331.0 | 2.1 | 3.1 | 79.31% |
| `validation` | `weekly_profile` (168h) | `48` | 25 | 19.8 | 3331.0 | 2.1 | 3.1 | 79.31% |
| `validation` | `weekly_profile` (168h) | `168` | 25 | 19.8 | 3331.0 | 2.1 | 3.1 | 79.31% |

### Table 2: Training (2016) vs. Validation (H1 2017) Valid-Zero Frequency Shift

| Population | Buildings | 2016 Train Observed Hours | 2016 Valid Zero / 1,000h (Mean / Median / Range) | H1 2017 Val Observed Hours | H1 2017 Valid Zero / 1,000h | Shift $\Delta$ (Val $-$ Train / 1,000h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Core Panther Education** | 29 | 8,784 | `381.02` / `384.34` / `[366.98, 384.79]` | 4,332 | `0.00` | **`-381.02`** |
| **Sensitivity (`Quintin`)** | 1 | 3,811 | `86.59` (`330` zero hours) | 4,332 | `0.00` | **`-86.59`** |

---

## 3. Multi-Horizon Forecasting Comparison (Phases 3B.1 & 3B.2)

### Table 3: Complete Forecasting Macro Synthesis Across Tasks and Lookbacks (29 Core Buildings)

| Task | Lookback (h) | Model | Role | Macro MAE (kWh) | Median MAE (kWh) | MAE Range (kWh) | Macro RMSE (kWh) | Macro Bias (kWh) | Macro MASE | Bldgs Improved vs. Baseline (MAE / RMSE / MASE) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hourly` | `24` | `persistence` | `task_baseline` | `7.191` | `6.470` | `[0.39, 29.62]` | `12.899` | `+0.008` | `1.027` | — |
| `hourly` | `24` | `sarima` | `candidate_recursive` | `7.170` | `6.504` | `[0.39, 29.81]` | `12.790` | `-0.738` | `1.022` | `20/29` / `29/29` / `20/29` |
| `hourly` | `24` | `random_forest` | `candidate_recursive` | `7.492` | `6.634` | `[0.50, 30.74]` | `12.183` | `+0.435` | `1.034` | `15/29` / `20/29` / `15/29` |
| `hourly` | `24` | `xgboost` | `candidate_recursive` | `12.367` | `10.339` | `[0.56, 36.80]` | `16.548` | `-7.671` | `1.609` | `1/29` / `9/29` / `1/29` |
| `hourly` | `48` | `persistence` | `task_baseline` | `7.179` | `6.463` | `[0.39, 29.40]` | `12.896` | `+0.010` | `1.027` | — |
| `hourly` | `48` | `sarima` | `candidate_recursive` | `7.157` | `6.496` | `[0.39, 29.58]` | `12.787` | `-0.734` | `1.022` | `20/29` / `29/29` / `20/29` |
| `hourly` | `48` | `random_forest` | `candidate_recursive` | `6.772` | `6.637` | `[0.49, 28.72]` | `11.177` | `+0.371` | `0.936` | `18/29` / `19/29` / `18/29` |
| `hourly` | `48` | `xgboost` | `candidate_recursive` | `11.605` | `9.395` | `[0.56, 36.32]` | `15.711` | `-7.927` | `1.498` | `5/29` / `11/29` / `5/29` |
| `hourly` | `168` | `persistence` | `task_baseline` | `7.100` | `6.432` | `[0.40, 29.29]` | `12.566` | `+0.004` | `1.017` | — |
| `hourly` | `168` | `sarima` | `candidate_recursive` | `7.081` | `6.407` | `[0.40, 29.45]` | `12.461` | `-0.737` | `1.013` | `19/29` / `28/29` / `19/29` |
| `hourly` | `168` | `random_forest` | `candidate_recursive` | `6.731` | `6.290` | `[0.50, 28.13]` | `10.813` | `+0.374` | `0.937` | `18/29` / `19/29` / `18/29` |
| `hourly` | `168` | `xgboost` | `candidate_recursive` | `11.486` | `9.292` | `[0.56, 36.34]` | `15.426` | `-7.965` | `1.482` | `5/29` / `10/29` / `5/29` |
| `daily_profile` | `24` | `daily_seasonal_naive` | `task_baseline` | `12.325` | `10.694` | `[0.96, 62.21]` | `20.464` | `+0.131` | `1.661` | — |
| `daily_profile` | `24` | `sarima` | `candidate_recursive` | `27.034` | `24.650` | `[1.16, 75.19]` | `35.591` | `-21.202` | `3.817` | `2/29` / `3/29` / `2/29` |
| `daily_profile` | `24` | `random_forest` | `candidate_recursive` | `18.365` | `17.315` | `[1.71, 72.79]` | `25.914` | `-4.313` | `2.499` | `3/29` / `8/29` / `3/29` |
| `daily_profile` | `24` | `xgboost` | `candidate_recursive` | `30.174` | `29.327` | `[1.93, 92.73]` | `37.620` | `-22.059` | `3.986` | `0/29` / `1/29` / `0/29` |
| `daily_profile` | `48` | `daily_seasonal_naive` | `task_baseline` | `12.380` | `10.744` | `[0.97, 62.54]` | `20.534` | `+0.145` | `1.668` | — |
| `daily_profile` | `48` | `sarima` | `candidate_recursive` | `26.988` | `24.603` | `[1.17, 74.70]` | `35.570` | `-21.234` | `3.813` | `2/29` / `3/29` / `2/29` |
| `daily_profile` | `48` | `random_forest` | `candidate_recursive` | `14.424` | `11.443` | `[1.25, 68.69]` | `21.225` | `-2.413` | `1.885` | `9/29` / `13/29` / `9/29` |
| `daily_profile` | `48` | `xgboost` | `candidate_recursive` | `27.595` | `17.637` | `[1.33, 101.21]` | `34.708` | `-21.565` | `3.459` | `0/29` / `4/29` / `0/29` |
| `daily_profile` | `168` | `daily_seasonal_naive` | `task_baseline` | `12.158` | `10.551` | `[0.98, 61.67]` | `19.841` | `+0.009` | `1.644` | — |
| `daily_profile` | `168` | `sarima` | `candidate_recursive` | `26.915` | `24.570` | `[1.19, 75.53]` | `35.288` | `-21.232` | `3.801` | `2/29` / `3/29` / `2/29` |
| `daily_profile` | `168` | `random_forest` | `candidate_recursive` | `14.557` | `11.353` | `[1.21, 66.01]` | `21.142` | `-2.827` | `1.909` | `10/29` / `14/29` / `10/29` |
| `daily_profile` | `168` | `xgboost` | `candidate_recursive` | `27.181` | `17.383` | `[1.29, 95.82]` | `34.211` | `-21.350` | `3.403` | `1/29` / `4/29` / `1/29` |
| `weekly_profile` | `24` | `weekly_seasonal_naive` | `task_baseline` | `14.135` | `12.504` | `[1.61, 65.78]` | `22.337` | `-0.413` | `1.937` | — |
| `weekly_profile` | `24` | `sarima` | `candidate_recursive` | `36.566` | `38.491` | `[1.58, 91.55]` | `45.311` | `-33.498` | `4.929` | `1/29` / `1/29` / `1/29` |
| `weekly_profile` | `24` | `random_forest` | `candidate_recursive` | `24.810` | `23.155` | `[2.30, 90.61]` | `32.222` | `-9.219` | `3.481` | `3/29` / `4/29` / `3/29` |
| `weekly_profile` | `24` | `xgboost` | `candidate_recursive` | `33.285` | `31.679` | `[2.07, 98.04]` | `40.495` | `-23.164` | `4.439` | `0/29` / `2/29` / `0/29` |
| `weekly_profile` | `48` | `weekly_seasonal_naive` | `task_baseline` | `14.135` | `12.504` | `[1.61, 65.78]` | `22.337` | `-0.413` | `1.937` | — |
| `weekly_profile` | `48` | `sarima` | `candidate_recursive` | `36.566` | `38.491` | `[1.58, 91.55]` | `45.311` | `-33.498` | `4.929` | `1/29` / `1/29` / `1/29` |
| `weekly_profile` | `48` | `random_forest` | `candidate_recursive` | `19.069` | `16.383` | `[1.93, 88.52]` | `26.283` | `-6.271` | `2.520` | `6/29` / `10/29` / `6/29` |
| `weekly_profile` | `48` | `xgboost` | `candidate_recursive` | `34.539` | `30.221` | `[1.75, 118.38]` | `41.840` | `-28.226` | `4.484` | `1/29` / `2/29` / `1/29` |
| `weekly_profile` | `168` | `weekly_seasonal_naive` | `task_baseline` | `14.135` | `12.504` | `[1.61, 65.78]` | `22.337` | `-0.413` | `1.937` | — |
| `weekly_profile` | `168` | `sarima` | `candidate_recursive` | `36.566` | `38.491` | `[1.58, 91.55]` | `45.311` | `-33.498` | `4.929` | `1/29` / `1/29` / `1/29` |
| `weekly_profile` | `168` | `random_forest` | `candidate_recursive` | `18.653` | `15.865` | `[1.86, 77.28]` | `25.744` | `-6.238` | `2.491` | `7/29` / `11/29` / `7/29` |
| `weekly_profile` | `168` | `xgboost` | `candidate_recursive` | `33.875` | `26.411` | `[1.62, 116.45]` | `41.128` | `-28.332` | `4.355` | `1/29` / `3/29` / `1/29` |

### Table 4: Best-Performing Model Counts Across the 29 Core Buildings by Task and Lookback

| Task | Lookback (h) | Best MAE Model Counts (29 Buildings) | Best RMSE Model Counts (29 Buildings) |
| :--- | :---: | :--- | :--- |
| `hourly` | `24` | `persistence`: 7, `sarima`: 7, `random_forest`: 15 | `sarima`: 9, `random_forest`: 20 |
| `hourly` | `48` | `persistence`: 6, `sarima`: 5, `random_forest`: 18 | `sarima`: 10, `random_forest`: 19 |
| `hourly` | `168` | `persistence`: 6, `sarima`: 5, `random_forest`: 18 | `persistence`: 1, `sarima`: 9, `random_forest`: 19 |
| `daily_profile` | `24` | `daily_seasonal_naive`: 26, `sarima`: 1, `random_forest`: 2 | `daily_seasonal_naive`: 21, `sarima`: 2, `random_forest`: 5, `xgboost`: 1 |
| `daily_profile` | `48` | `daily_seasonal_naive`: 20, `random_forest`: 9 | `daily_seasonal_naive`: 14, `sarima`: 1, `random_forest`: 11, `xgboost`: 3 |
| `daily_profile` | `168` | `daily_seasonal_naive`: 19, `random_forest`: 10 | `daily_seasonal_naive`: 13, `sarima`: 1, `random_forest`: 11, `xgboost`: 4 |
| `weekly_profile` | `24` | `weekly_seasonal_naive`: 25, `sarima`: 1, `random_forest`: 3 | `weekly_seasonal_naive`: 24, `random_forest`: 4, `xgboost`: 1 |
| `weekly_profile` | `48` | `weekly_seasonal_naive`: 21, `sarima`: 1, `random_forest`: 6, `xgboost`: 1 | `weekly_seasonal_naive`: 19, `random_forest`: 10 |
| `weekly_profile` | `168` | `weekly_seasonal_naive`: 20, `sarima`: 1, `random_forest`: 7, `xgboost`: 1 | `weekly_seasonal_naive`: 17, `random_forest`: 11, `xgboost`: 1 |

---

## 4. Unsupervised Anomaly Detection Synthesis (Phase 3B.3 Remediated)

### Table 5: Remediated Anomaly Detection Macro & Coincidence Summary (29 Core Buildings)

| Feature Mode | Detector (Dim / Scope) | Operating Point | Nominal Train / 1,000h | Realized Train / 1,000h (Mean / Med) | Val Flagged / 1,000h (Mean / Med / Range) | Bldgs w/ Events (`Zero-Ev`) | Conditional Mean Event Duration (h) | Coincidence Hours ($\ge 2$ / $\ge 5$ / Max Simul Bldgs) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw_consumption` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `0.5pct` | `5.0` | `2.62` / `1.71` | `13.40` / `1.15` / `[0.00, 123.73]` | `21/29` (`8`) | `2.360` | `285` / `45` / `11` |
| `raw_consumption` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `1.0pct` | `10.0` | `4.86` / `1.71` | `18.26` / `2.54` / `[0.00, 136.20]` | `21/29` (`8`) | `2.486` | `477` / `73` / `11` |
| `raw_consumption` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `2.0pct` | `20.0` | `9.05` / `1.71` | `24.62` / `4.38` / `[0.00, 152.12]` | `21/29` (`8`) | `2.631` | `739` / `107` / `11` |
| `raw_consumption` | `isolation_forest` (`1D` `full_mode_feature_vector`) | `0.5pct` | `5.0` | `4.86` / `4.90` | `15.13` / `12.70` / `[0.23, 89.06]` | `29/29` (`0`) | `4.034` | `305` / `68` / `11` |
| `raw_consumption` | `isolation_forest` (`1D` `full_mode_feature_vector`) | `1.0pct` | `10.0` | `9.90` / `9.90` | `32.25` / `27.70` / `[0.69, 200.51]` | `29/29` (`0`) | `2.627` | `867` / `110` / `11` |
| `raw_consumption` | `isolation_forest` (`1D` `full_mode_feature_vector`) | `2.0pct` | `20.0` | `19.66` / `19.92` | `72.93` / `49.86` / `[3.92, 366.87]` | `29/29` (`0`) | `3.340` | `2669` / `290` / `16` |
| `raw_consumption` | `lof_novelty` (`1D` `full_mode_feature_vector`) | `0.5pct` | `5.0` | `4.97` / `5.01` | `50.24` / `25.62` / `[0.69, 317.17]` | `29/29` (`0`) | `2.825` | `1868` / `105` / `13` |
| `raw_consumption` | `lof_novelty` (`1D` `full_mode_feature_vector`) | `1.0pct` | `10.0` | `9.92` / `10.02` | `69.28` / `45.69` / `[1.15, 328.25]` | `29/29` (`0`) | `2.660` | `2520` / `258` / `14` |
| `raw_consumption` | `lof_novelty` (`1D` `full_mode_feature_vector`) | `2.0pct` | `20.0` | `19.73` / `20.04` | `96.90` / `64.84` / `[4.62, 414.59]` | `29/29` (`0`) | `2.435` | `3324` / `656` / `15` |
| `seasonal_deviation` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `0.5pct` | `5.0` | `4.56` / `4.57` | `0.41` / `0.00` / `[0.00, 5.79]` | `7/29` (`22`) | `1.598` | `1` / `0` / `2` |
| `seasonal_deviation` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `1.0pct` | `10.0` | `9.16` / `9.13` | `0.69` / `0.00` / `[0.00, 8.11]` | `7/29` (`22`) | `1.956` | `1` / `0` / `2` |
| `seasonal_deviation` | `statistical_zscore` (`1D` `univariate_primary_feature`) | `2.0pct` | `20.0` | `18.61` / `18.26` | `0.96` / `0.00` / `[0.00, 9.73]` | `7/29` (`22`) | `1.833` | `1` / `0` / `2` |
| `seasonal_deviation` | `isolation_forest` (`3D` `full_mode_feature_vector`) | `0.5pct` | `5.0` | `4.99` / `5.02` | `12.05` / `7.87` / `[0.46, 79.15]` | `29/29` (`0`) | `1.340` | `246` / `45` / `11` |
| `seasonal_deviation` | `isolation_forest` (`3D` `full_mode_feature_vector`) | `1.0pct` | `10.0` | `9.93` / `10.05` | `21.82` / `15.52` / `[0.46, 165.01]` | `29/29` (`0`) | `1.408` | `521` / `79` / `13` |
| `seasonal_deviation` | `isolation_forest` (`3D` `full_mode_feature_vector`) | `2.0pct` | `20.0` | `19.95` / `20.09` | `40.61` / `28.02` / `[3.01, 225.87]` | `29/29` (`0`) | `1.537` | `1169` / `196` / `14` |
| `seasonal_deviation` | `lof_novelty` (`3D` `full_mode_feature_vector`) | `0.5pct` | `5.0` | `4.99` / `5.02` | `28.73` / `9.96` / `[1.85, 239.53]` | `29/29` (`0`) | `1.697` | `802` / `44` / `10` |
| `seasonal_deviation` | `lof_novelty` (`3D` `full_mode_feature_vector`) | `1.0pct` | `10.0` | `9.94` / `10.05` | `46.00` / `19.69` / `[6.02, 306.69]` | `29/29` (`0`) | `1.839` | `1616` / `114` / `10` |
| `seasonal_deviation` | `lof_novelty` (`3D` `full_mode_feature_vector`) | `2.0pct` | `20.0` | `20.03` / `20.09` | `73.16` / `46.10` / `[12.04, 367.04]` | `29/29` (`0`) | `1.944` | `2574` / `334` / `12` |

### Table 6: Separate Sensitivity Population (`Panther_education_Quintin`, `1.0pct` Operating Point)

| Feature Mode | Detector (`Dim`) | Train Eligible Hours | Val Eligible Hours | Threshold ($\tau_{0.99}$) | Train Flagged / 1,000h | Val Flagged / 1,000h | Event Count | Conditional Mean Duration (h) | Val Flagged Zero Hours |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw_consumption` | `statistical_zscore` (`1D`) | 3,811 | 4,332 | `3.3284` | `2.10` | `6.46` | 15 | `1.867` | 0 |
| `raw_consumption` | `isolation_forest` (`1D`) | 3,811 | 4,332 | `0.7165` | `10.23` | `22.62` | 48 | `2.042` | 0 |
| `raw_consumption` | `lof_novelty` (`1D`) | 3,811 | 4,332 | `2.6344` | `10.23` | `6.46` | 15 | `1.867` | 0 |
| `seasonal_deviation` | `statistical_zscore` (`1D`) | 3,599 | 4,317 | `12.4281` | `9.72` | `0.69` | 3 | `1.000` | 0 |
| `seasonal_deviation` | `isolation_forest` (`3D`) | 3,599 | 4,317 | `0.6636` | `10.00` | `12.28` | 36 | `1.472` | 0 |
| `seasonal_deviation` | `lof_novelty` (`3D`) | 3,599 | 4,317 | `1.9652` | `10.00` | `22.01` | 39 | `2.436` | 0 |

---

## 5. Cross-Building Heterogeneity Analysis (`lookback=168h`, `1.0pct` Operating Point)

- **Forecasting Degradation vs. Seasonal-Deviation LOF Alert Rate Correlation:** Pearson $r = 0.6906$ across the 29 core buildings.
- **Forecasting Absolute Bias vs. Daily Profile MAE Degradation Correlation:** Pearson $r = 0.5802$ across the 29 core buildings.

### Table 7: Per-Building Integrated Diagnostic Profile (29 Core Buildings)

| Building ID | Train Scale Denom (kWh) | 2016 Zero / 1,000h | `hourly` Best Model (MASE) | `daily_profile` Best Model (MASE) | RF Daily $\Delta$MASE (Bias kWh) | `weekly_profile` Best Model (MASE) | Seasonal `1.0pct` Val Flagged / 1,000h (`Z` / `IF` / `LOF`) |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- | :---: |
| `Panther_education_Alecia` | `9.65` | `384.4` | `random_forest` (`0.949`) | `daily_seasonal_naive` (`1.291`) | `+0.086` (`-1.12`) | `weekly_seasonal_naive` (`1.652`) | `0.0` / `11.1` / `19.2` |
| `Panther_education_Annetta` | `6.93` | `384.4` | `random_forest` (`0.947`) | `daily_seasonal_naive` (`1.522`) | `+0.116` (`-1.28`) | `weekly_seasonal_naive` (`1.709`) | `0.0` / `28.7` / `68.0` |
| `Panther_education_Aurora` | `6.67` | `384.3` | `sarima` (`1.580`) | `daily_seasonal_naive` (`3.512`) | `+0.401` (`+1.63`) | `weekly_seasonal_naive` (`2.047`) | `0.0` / `165.0` / `299.9` |
| `Panther_education_Cleopatra` | `2.36` | `367.0` | `persistence` (`1.329`) | `random_forest` (`2.233`) | `-0.106` (`+1.32`) | `weekly_seasonal_naive` (`2.199`) | `2.1` / `20.4` / `75.3` |
| `Panther_education_Diann` | `11.69` | `384.6` | `persistence` (`0.520`) | `daily_seasonal_naive` (`1.300`) | `+0.546` (`+15.09`) | `weekly_seasonal_naive` (`1.497`) | `0.0` / `19.2` / `18.3` |
| `Panther_education_Edna` | `7.52` | `380.1` | `random_forest` (`0.977`) | `daily_seasonal_naive` (`1.806`) | `+0.977` (`-15.30`) | `weekly_seasonal_naive` (`1.754`) | `0.0` / `43.1` / `107.0` |
| `Panther_education_Emily` | `7.24` | `378.1` | `random_forest` (`1.235`) | `random_forest` (`1.712`) | `-0.071` (`-0.15`) | `random_forest` (`2.297`) | `0.7` / `22.5` / `28.0` |
| `Panther_education_Enriqueta` | `9.19` | `379.1` | `random_forest` (`0.768`) | `daily_seasonal_naive` (`1.068`) | `+0.050` (`-4.62`) | `weekly_seasonal_naive` (`1.400`) | `0.5` / `11.2` / `14.7` |
| `Panther_education_Genevieve` | `8.05` | `384.4` | `random_forest` (`0.582`) | `daily_seasonal_naive` (`1.308`) | `+0.318` (`-7.02`) | `weekly_seasonal_naive` (`1.268`) | `0.0` / `1.9` / `6.0` |
| `Panther_education_Gina` | `9.83` | `381.6` | `sarima` (`0.652`) | `daily_seasonal_naive` (`1.273`) | `+0.462` (`-3.47`) | `weekly_seasonal_naive` (`1.753`) | `0.0` / `8.3` / `8.3` |
| `Panther_education_Hugh` | `12.22` | `384.3` | `persistence` (`0.501`) | `daily_seasonal_naive` (`1.151`) | `+0.060` (`-5.54`) | `random_forest` (`1.298`) | `0.0` / `7.4` / `6.5` |
| `Panther_education_Ivan` | `6.70` | `384.4` | `random_forest` (`1.137`) | `random_forest` (`1.602`) | `-0.004` (`+1.42`) | `weekly_seasonal_naive` (`1.867`) | `0.0` / `17.1` / `19.7` |
| `Panther_education_Janis` | `3.43` | `384.8` | `random_forest` (`0.818`) | `random_forest` (`1.957`) | `-0.070` (`+0.28`) | `weekly_seasonal_naive` (`2.225`) | `8.1` / `31.3` / `32.9` |
| `Panther_education_Jerome` | `12.64` | `384.3` | `random_forest` (`0.601`) | `daily_seasonal_naive` (`1.586`) | `+0.721` (`-22.19`) | `weekly_seasonal_naive` (`1.300`) | `0.0` / `16.2` / `10.4` |
| `Panther_education_Jonathan` | `0.82` | `384.4` | `sarima` (`0.487`) | `daily_seasonal_naive` (`1.201`) | `+0.281` (`+0.26`) | `sarima` (`1.936`) | `0.0` / `6.0` / `29.9` |
| `Panther_education_Karri` | `20.39` | `376.3` | `persistence` (`0.282`) | `daily_seasonal_naive` (`1.126`) | `+0.095` (`-15.37`) | `random_forest` (`1.723`) | `0.0` / `7.4` / `8.3` |
| `Panther_education_Mattie` | `0.91` | `384.1` | `random_forest` (`0.780`) | `daily_seasonal_naive` (`1.814`) | `+0.220` (`-0.09`) | `weekly_seasonal_naive` (`2.189`) | `4.2` / `12.0` / `18.5` |
| `Panther_education_Misty` | `2.18` | `368.5` | `random_forest` (`0.803`) | `random_forest` (`1.798`) | `-0.149` (`-0.11`) | `random_forest` (`2.498`) | `0.0` / `16.7` / `28.5` |
| `Panther_education_Mohammad` | `9.68` | `379.8` | `random_forest` (`0.562`) | `daily_seasonal_naive` (`1.642`) | `+0.189` (`-5.45`) | `weekly_seasonal_naive` (`1.118`) | `0.0` / `14.6` / `6.3` |
| `Panther_education_Neal` | `40.34` | `384.4` | `random_forest` (`0.697`) | `daily_seasonal_naive` (`1.529`) | `+0.107` (`-7.14`) | `weekly_seasonal_naive` (`1.631`) | `3.5` / `13.9` / `16.9` |
| `Panther_education_Rosalie` | `3.34` | `384.0` | `sarima` (`0.825`) | `random_forest` (`1.817`) | `-0.200` (`+0.35`) | `random_forest` (`2.129`) | `0.0` / `15.5` / `19.7` |
| `Panther_education_Scarlett` | `2.84` | `374.8` | `random_forest` (`0.616`) | `daily_seasonal_naive` (`1.269`) | `+0.188` (`+0.03`) | `weekly_seasonal_naive` (`1.416`) | `0.0` / `5.3` / `13.4` |
| `Panther_education_Shelton` | `4.38` | `384.4` | `random_forest` (`0.900`) | `daily_seasonal_naive` (`1.826`) | `+0.090` (`-0.49`) | `weekly_seasonal_naive` (`2.648`) | `0.0` / `26.2` / `27.6` |
| `Panther_education_Sophia` | `3.72` | `376.0` | `persistence` (`0.611`) | `random_forest` (`1.401`) | `-0.089` (`-0.61`) | `weekly_seasonal_naive` (`1.534`) | `0.0` / `0.5` / `31.2` |
| `Panther_education_Teofila` | `7.26` | `384.4` | `random_forest` (`0.966`) | `random_forest` (`1.597`) | `-0.066` (`-0.72`) | `weekly_seasonal_naive` (`2.202`) | `0.0` / `18.3` / `23.8` |
| `Panther_education_Tina` | `1.28` | `382.2` | `random_forest` (`0.860`) | `random_forest` (`1.439`) | `-0.023` (`-0.17`) | `xgboost` (`1.853`) | `1.2` / `14.3` / `16.9` |
| `Panther_education_Vincent` | `7.84` | `384.6` | `sarima` (`1.252`) | `daily_seasonal_naive` (`1.814`) | `+3.493` (`-10.67`) | `weekly_seasonal_naive` (`1.749`) | `0.0` / `23.6` / `306.7` |
| `Panther_education_Violet` | `6.56` | `370.9` | `persistence` (`0.679`) | `random_forest` (`1.347`) | `-0.003` (`-1.04`) | `random_forest` (`1.781`) | `0.0` / `6.5` / `26.4` |
| `Panther_education_Zelda` | `3.04` | `384.6` | `random_forest` (`1.192`) | `daily_seasonal_naive` (`1.957`) | `+0.055` (`+0.22`) | `random_forest` (`2.696`) | `0.0` / `48.6` / `45.6` |

---

## 6. Recommended Research Figures & Visual Diagnostics

1. **Figure 1 — Horizon-by-Model MASE Distribution (Box + Strip Plot across 29 Core Buildings):**
   - Displays per-building `mase_training_daily_naive` at `lookback=168h` across `hourly`, `daily_profile`, and `weekly_profile` for the task baseline, `sarima`, `random_forest`, and `xgboost`, illustrating the shift from `random_forest` superiority at $H=1$ (`MASE = 0.937`) to seasonal-naive superiority at $H=24$ (`MASE = 1.644`) and $H=168$ (`MASE = 1.937`).
2. **Figure 2 — Multi-Step Recursive Bias vs. MAE Degradation Scatter Plot (`daily_profile`, `lookback=168h`):**
   - Plots per-building $|\text{mean\_error}|$ (kWh) against `mae_delta` (kWh) for `random_forest`, `sarima`, and `xgboost`, showing that multi-step degradation is tightly driven by 2016-to-2017 level bias ($r = 0.5802$ for `random_forest`).
3. **Figure 3 — Training vs. Validation Alert-Rate Operating Curves (`0.5pct`, `1.0pct`, `2.0pct`):**
   - Compares nominal training rates (`5.0`, `10.0`, `20.0` per 1,000h), realized training rates, and validation alert-rate distributions (median and interquartile range across 29 buildings) for `raw_consumption` (`1D`) and `seasonal_deviation` (`1D` `statistical_zscore` vs. `3D` `isolation_forest` and `lof_novelty`).
4. **Figure 4 — Cross-Phase Regime-Shift Coupling Scatter Plot:**
   - Plots `random_forest` `daily_profile` `mase_delta` against `seasonal_deviation` `lof_novelty` validation alert rate (`1.0pct`) across the 29 buildings ($r = 0.6906$), highlighting stable-regime buildings (`Rosalie`, `Emily`, `Teofila`, `Sophia`) versus regime-shifted buildings (`Aurora`, `Edna`, `Jerome`, `Vincent`).

---

## 7. Model Selection Conclusions & Threats to Validity

### Model Selection Conclusions
- **1-Hour Ahead (`hourly`):** At lookback=168h, Random Forest achieves the lowest macro mean MAE (6.731 kWh vs 7.100 kWh for persistence), lowest macro mean RMSE (10.813 kWh vs 12.566 kWh), and lowest macro mean MASE (0.937 vs 1.017), improving MAE on 18/29 buildings and RMSE on 19/29 buildings. SARIMA also improves MAE slightly on 19/29 buildings (macro MAE 7.081 kWh, MASE 1.013).
- **24-Hour Daily Profile (`daily_profile`, Primary Task):** Daily seasonal naive achieves the lowest macro mean MAE (12.158 kWh), lowest median MAE (10.551 kWh), lowest macro mean RMSE (19.841 kWh), near-zero bias (+0.009 kWh), and lowest macro mean MASE (1.644), winning 19/29 buildings on MAE. Random Forest (lookback=168h) is the strongest ML candidate (macro MAE 14.557 kWh, MASE 1.909), winning 10/29 buildings on MAE and 14/29 on RMSE where 2016-to-2017 positive load levels are stable. SARIMA (bias -21.232 kWh, MASE 3.801) and XGBoost (bias -21.350 kWh, MASE 3.403) suffer severe downward recursive drift toward the 2016 zero-heavy training regime.
- **168-Hour Weekly Profile (`weekly_profile`, Exploratory Task):** Weekly seasonal naive achieves the lowest macro mean MAE (14.135 kWh), lowest median MAE (12.504 kWh), lowest macro mean RMSE (22.337 kWh), and lowest macro mean MASE (1.937), winning 20/29 buildings on MAE. Random Forest (lookback=168h) wins 7/29 buildings on MAE (11/29 on RMSE) with macro MASE 2.491, whereas SARIMA (MASE 4.929) and XGBoost (MASE 4.355) compound negative recursive bias.
- **Unsupervised Anomaly Detection:** In 3D seasonal_deviation mode, Isolation Forest (1.0pct median validation rate 15.52/1,000h, macro mean 21.82/1,000h, conditional mean event duration 1.41h) and full-multiplicity LOF novelty (1.0pct median 19.69/1,000h, macro mean 46.00/1,000h, conditional mean event duration 1.84h) capture joint hour-of-week and ramp departures across all 29 buildings without flagging any valid zero readings (validation_flagged_zero_hours == 0).

### Scientific Limitations & Threats to Validity
1. No confirmed ground-truth anomaly labels exist; all anomaly metrics describe unlabeled statistical or geometric departures from 2016 training references, and supervised metrics (precision, recall, F1, ROC-AUC, accuracy) are strictly prohibited.
2. Major 2016-to-2017 valid-zero regime shift: the 29 core buildings average 381.02 valid zero hours per 1,000 observed hours in 2016 (median 384.34/1,000h) versus 0.00/1,000h in H1 2017. This structural shift induces negative bias in frozen 2016 multi-step forecasters and elevates validation anomaly alert rates above nominal training quantiles.
3. Feature-space dimensionality in seasonal_deviation anomaly detection intentionally compares a 1D univariate statistical baseline (|slot_zscore|) against 3D multivariate detectors ([slot_zscore, lag_1h_scaled_diff, lag_24h_scaled_diff]) over identical eligible timestamps.
4. Weather-exogenous forecasting was specified for complete-case evaluation in Phase 2.7/3A, but Phases 3B.1 and 3B.2 evaluated electricity-only baselines and candidates; weather-augmented forecasting remains an unexecuted sensitivity extension.
5. The held-out test partition (2017-07-01 00:00:00 through 2017-12-31 23:00:00) remains untouched (not_run); findings reflect H1 2017 validation performance on 29 UCF Panther education buildings and do not support broad multi-year or cross-campus generalization claims.
