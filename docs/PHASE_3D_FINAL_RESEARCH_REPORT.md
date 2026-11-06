# Campus Energy Consumption Intelligence & Unsupervised Anomaly Detection: Final Research Report (Phase 3D)

**Report Version:** `3D-final-v1` | **Synthesis Baseline:** `3C-remediated-v1` | **Evaluation Scope:** `validation_only` (`2016` Train $\to$ `H1 2017` Validation) | **Held-Out Test Partition (`H2 2017`):** `not_run` | **Ground-Truth Anomaly Labels:** `none`

---

## 1. Abstract & Executive Summary

This report presents the complete, cryptographically verified experimental methodology and empirical findings of the **Campus Energy Consumption Intelligence (CEI)** research program across Phases 1 through 3D. Following a rigorous data-quality audit that rejected an initial 300-row synthetic Kaggle dataset due to severe temporal sparsity (`52–60` observed days per building), `17` conflicting duplicate timestamp groups, and unverified synthetic labels, the study migrated in **Phase 2.6** to the immutable hourly electricity and weather records of the **University of Central Florida (UCF) Panther** site from the **Building Data Genome Project 2 (BDG2)** repository (Miller et al., 2020, *Scientific Data* 7, 368).

Across the **29 core Panther educational buildings** (`8,784` training hours in 2016; `4,344` validation hours in H1 2017) and the strictly isolated sensitivity building (`Panther_education_Quintin`, `56.61%` missing in 2016), we evaluate:
1. **Multi-Horizon Electricity Forecasting (Phases 3A, 3B.1, 3B.2):** Persistence and seasonal-naive baselines (`period=24h`, `period=168h`) versus frozen-origin recursive **SARIMA**, **Random Forest**, and **XGBoost** across three horizons (`hourly` $H=1$, secondary; `daily_profile` $H=24$, primary; `weekly_profile` $H=168$, exploratory) and three historical lookback windows (`24h`, `48h`, `168h`) over identical common-lookback origin and target masks.
2. **Training-Calibrated Unsupervised Anomaly Detection (Phase 3B.3 Remediated):** Per-building **Statistical Z-Score**, **Isolation Forest**, and full-empirical-multiplicity **Local Outlier Factor (LOF) novelty detection** across `raw_consumption` (`1D`) and `seasonal_deviation` (`1D` univariate reference vs. `3D` joint slot-deviation and ramp vector) at frozen 2016 training upper-tail score quantiles (`99.5th`, `99th`, and `98th` percentiles; nominal `0.5%`, `1.0%`, and `2.0%` training alert rates).

### Principal Empirical Findings
- **Horizon-Dependent Forecasting Superiority:** No single model dominates all horizons. At **$H=1$ (`hourly`)**, `random_forest` (`lookback=168h`) achieves the lowest macro mean MAE (**`6.731 kWh`** vs. `7.100 kWh` for `persistence`), lowest macro mean RMSE (**`10.813 kWh`** vs. `12.566 kWh`), and lowest macro mean MASE (**`0.937`** vs. `1.017`), winning **`18/29`** buildings on MAE (`19/29` on RMSE). At **$H=24$ (`daily_profile`, primary task)** and **$H=168$ (`weekly_profile`, exploratory task)**, `daily_seasonal_naive` (Macro MAE **`12.158 kWh`**, Median **`10.551 kWh`**, Macro RMSE **`19.841 kWh`**, Bias **`+0.009 kWh`**, Macro MASE **`1.644`**) and `weekly_seasonal_naive` (Macro MAE **`14.135 kWh`**, Macro MASE **`1.937`**) outperform all frozen-origin recursive candidates in aggregate macro metrics, winning **`19/29`** and **`20/29`** buildings on MAE, respectively. `random_forest` (`lookback=168h`) is the sole competitive ML candidate on multi-step profiles (`daily_profile` Macro MAE **`14.557 kWh`**, Macro MASE **`1.909`**, winning **`10/29`** buildings on MAE and **`14/29`** on RMSE).
- **Mechanistic Separation of Model Configuration vs. Recursive Drift:** Multi-step degradation in Phase 3B.2 reflects distinct architectural and configuration mechanisms rather than regime shift alone: (a) `sarima` is configured as a non-seasonal AR(1) `(1,0,0)x(0,0,0,0)` with constant intercept (`trend='c'`), causing multi-step forecasts $\hat{y}_{t+h|t} = \mu_{2016} + \phi_1^h(y_t - \mu_{2016})$ to decay geometrically toward the zero-diluted 2016 unconditional mean (`daily_profile` Bias `-21.232 kWh`, `weekly_profile` `-33.498 kWh`); (b) `xgboost` (`n_estimators=40`, `learning_rate=0.05`) retains $(1 - 0.05)^{40} = 0.1285$ of its zero-diluted global `base_score = mean(y_train)`, producing a **`-7.965 kWh`** negative bias (`MASE = 1.482`) **even at $H=1$ without recursion** that compounds across 24-step (`-21.350 kWh`) and 168-step (`-28.332 kWh`) rollouts; and (c) strict `lag < lookback` filtering in `feature_lags` excludes `lag_24` at `lookback=24h` (`[1, 2, 3, 6, 12]`) and includes `lag_24` at `lookback=48h` (`[1, 2, 3, 6, 12, 24]`), cutting `random_forest` `daily_profile` Macro MAE from `18.365` to `14.424 kWh`.
- **Campus-Wide Valid-Zero Shift vs. Building-Specific Active-Load Shifts:** All 29 core buildings shift from a campus-wide mean of **`381.02` valid zero hours per 1,000 observed hours** (`38.10%`, median `384.34/1,000h`, range `[366.98, 384.79]`) in 2016 to **`0.00/1,000h`** in H1 2017. While this campus-wide shift depresses 2016 unconditional means and lowers training score quantiles, 2016 zero rates are nearly uniform across the 29 buildings (`20/29` within `384.0–384.8/1,000h`) and do **not** significantly explain cross-building differences in forecasting degradation (Pearson $r = 0.2330$, $p = 0.2239$; Spearman $\rho = 0.3286$, $p = 0.0818$).
- **Unsupervised Anomaly Detection & Outlier-Leverage Diagnostics:** Full-empirical-multiplicity `lof_novelty` and `isolation_forest` in 3D `seasonal_deviation` mode flag **zero valid `0.0 kWh` readings** (`validation_flagged_zero_hours == 0` across all `522` core and `18` `Quintin` rows) and maintain strict threshold monotonicity (`0.5pct > 1.0pct > 2.0pct` in `29/29` buildings). Furthermore, an exploratory cross-phase correlation audit proves that the apparent $N=29$ linear correlation between `random_forest` `daily_profile` `mase_delta` and `seasonal_deviation` `lof_novelty` alert rate (Pearson $r = 0.6906$) is a **non-robust single-outlier artifact** driven by `Panther_education_Vincent` (hat leverage $h = 0.8507$, Cook's $D = 1.2558$): across all 29 buildings, Spearman rank correlation is negative and null ($\rho = -0.1466$, $p = 0.4480$), and excluding `Panther_education_Vincent` ($N=28$) reduces Pearson $r$ to `0.2518` ($p = 0.1962$).

---

## 2. Dataset Provenance, Quality Audit & Canonical Migration (Phases 1–2.6)

### 2.1 Rejection of Initial Synthetic Dataset (Phases 1 & 2)
In Phases 1 and 2, the supplied 300-row Kaggle dataset (`energy_consumption_data.csv`) was profiled and conservatively preprocessed without imputation or deletion. Inspection revealed:
- Only `300` total records across `5` building identifiers (`B001–B005`), yielding `52–60` distinct observed dates per building across a 364-day span (`295–307` missing days per building).
- `17` duplicate `(building_id, date)` groups (`34` rows) with conflicting electricity readings differing by hundreds of kWh.
- Undocumented synthetic binary flags that did not represent verified physical faults.

Because irregular 84%-missing daily series cannot support valid 24-hour or 168-hour seasonal lags, the project migrated in **Phase 2.6** to the BDG2 UCF Panther dataset while preserving all Phase 1–2 scripts and tests intact.

### 2.2 BDG2 UCF Panther Canonical Ingestion (Phase 2.6)
- **Source Repository:** Building Data Genome Project 2 (`buds-lab/building-data-genome-project-2`, Miller et al., 2020, [doi:10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x); license recorded conservatively as `CC BY-SA 4.0`).
- **Core Population (`electricity_core.csv`):** `29` UCF Panther educational buildings (`primaryspaceusage == 'Education'`), spanning `17,544` hourly timestamps (`2016-01-01 00:00:00` through `2017-12-31 23:00:00`, timezone-naive source clock) with `508,776` canonical building-hour records (`meter_reading` in kWh).
- **Sensitivity Population (`electricity_sensitivity_quintin.csv`):** `Panther_education_Quintin` (`17,544` timestamps, `4,973` missing hours in 2016 = `56.61%` missing) is stored and evaluated in strict isolation so its prolonged 2016 meter outage cannot distort core macro summaries.
- **Weather (`weather.csv`):** `17,544` site-level hourly records for `site_id=Panther` (`airTemperature`, `dewTemperature`, `windSpeed`, `precipDepth1HR`, `seaLvlPressure`), kept separate and aligned strictly by exact timestamp.

---

## 3. Experimental Methodology & Leakage Prevention Architecture (Phases 2.7 & 3A)

### 3.1 Chronological Partitioning & Non-Overlapping Forecast Blocks
| Partition | Inclusive Timestamp Range | Total Hours | Experimental Role & Status |
| :--- | :---: | :---: | :--- |
| **Train** | `2016-01-01 00:00:00` – `2016-12-31 23:00:00` | `8,784` | Fit model parameters, seasonal slot references, scalers, MASE denominators, and quantile thresholds |
| **Validation** | `2017-01-01 00:00:00` – `2017-06-30 23:00:00` | `4,344` | Evaluate frozen 2016 forecasting models and frozen anomaly operating points |
| **Test** | `2017-07-01 00:00:00` – `2017-12-31 23:00:00` | `4,416` | **Held-out (`not_run`)** — untouched across all phases |

- **Block Origin Definition:** For horizon $H \in \{24, 168\}$, non-overlapping target blocks begin at $\text{block\_start} = \text{partition\_start} + k H\text{ hours}$ ($k \ge 0$) while the full $H$-hour block lies within the partition, with forecast origin $o = \text{block\_start} - 1\text{ hour}$. For $H=1$ (`hourly`), every target timestamp $t$ in the partition has origin $o = t - 1\text{ hour}$.
- **Common Lookback & Shared Eligibility Mask:** For each task with seasonal baseline period $P \in \{1, 24, 168\}$ and candidate lookback $L \in \{24, 48, 168\}$, the common required history is $M = \max(L, P)\text{ hours}$. An origin $o$ is eligible if and only if all $M$ historical observations $\{o - (M-1)\text{h}, \dots, o\}$ and all $H$ target observations $\{o+1\text{h}, \dots, o+H\text{h}\}$ are non-missing. Every baseline and candidate model is scored on the identical set of eligible origins and targets.
- **MASE Scaling:** For each building $b$, $\text{MASE}_{b} = \text{MAE}_{b,\text{val}} / d_b$, where $d_b = \frac{1}{|\mathcal{P}_{b,\text{train}}|} \sum_{(t, t-24) \in \mathcal{P}_{b,\text{train}}} |y_{b,t} - y_{b,t-24}|$ is computed strictly over non-missing period-24 training pairs in 2016.

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

## 4. Multi-Horizon Forecasting Evaluation & Mechanistic Analysis (Phases 3B.1 & 3B.2)

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

### 4.1 Mechanistic Separation of Configuration Effects vs. Recursive Error Propagation
1. **SARIMA Non-Seasonal AR(1) Specification (`(1,0,0)x(0,0,0,0)` with `trend='c'`):**
   - In `src/cei/config/phase3b2_candidate_experiment.json`, SARIMA is configured with `order = [1, 0, 0]` and `seasonal_order = [0, 0, 0, 0]` (`trend='c'`) across `hourly`, `daily_profile`, and `weekly_profile`. Because no seasonal autoregressive or moving-average polynomials ($s=24, 168$) are present, multi-step recursive predictions $\hat{y}_{t+h|t} = \mu_{2016} + \phi_1^h(y_t - \mu_{2016})$ decay geometrically toward the unconditional 2016 training mean $\mu_{2016}$ (depressed by the 38.10% 2016 valid-zero regime), producing large negative biases at $H=24$ (`-21.232 kWh`, MASE `3.801`) and $H=168$ (`-33.498 kWh`, MASE `4.929`).
   - Furthermore, `_fit_sarima` fits on the full 2016 series without using `lookback_hours` for feature construction; `lookback_hours` only affects the common-origin eligibility mask $\max(\text{lookback}, \text{period})$. Consequently, for `weekly_profile` (`period = 168`, so $\max(\text{lookback}, 168) = 168$ for `lookback = 24, 48, 168`), SARIMA's evaluated origins and metrics are **identical across all three lookbacks** (`Macro MAE = 36.566 kWh`, `MASE = 4.929`).
2. **XGBoost $H=1$ Shrinkage Under-Convergence (`n_estimators=40, learning_rate=0.05`) vs. Recursive Compounding:**
   - Even at `hourly` ($H=1$, where **no recursive rollout occurs**), XGBoost exhibits a large negative Macro Bias (`-7.671 kWh` at `lb=24h`, `-7.927 kWh` at `lb=48h`, `-7.965 kWh` at `lb=168h`; Macro MASE `1.609`, `1.498`, `1.482`). Because `XGBRegressor` initializes at `base_score = mean(y_train)` (diluted by 38.10% 2016 zeros) and executes `n_estimators = 40` trees at `learning_rate = 0.05`, $(1 - 0.05)^{40} = 0.1285$ of the initial downward offset remains unclosed even on strictly positive 2017 lag vectors. During 24-step and 168-step recursive rollouts, this 1-step underprediction compounds as underestimated predictions are fed back as subsequent lag features (`daily_profile` Bias `-21.350 kWh`, `weekly_profile` `-28.332 kWh` at `lb=168h`).
   - By contrast, `random_forest` (`n_estimators=20, max_depth=10`) averages un-shrunk bootstrap regression trees that isolate positive lag vectors from zero lag vectors without global `base_score` shrinkage (`hourly` Bias `+0.374 kWh`, `daily_profile` Bias `-2.827 kWh` at `lb=168h`).
3. **Strict `lag < lookback` Feature Selection (`feature_lags`):**
   - `feature_lags(config, lookback)` filters `feature_lags_hours = [1, 2, 3, 6, 12, 24, 48, 168]` using strict inequality (`lag < lookback`):
     - `lookback = 24h` uses only `[1, 2, 3, 6, 12]` (**excluding `lag_24`**), explaining why `random_forest` underperforms `persistence` at $H=1$ (`MAE 7.492` vs. `7.191 kWh`) and has higher $H=24$ error (`MAE 18.365 kWh`).
     - `lookback = 48h` uses `[1, 2, 3, 6, 12, 24]` (adding diurnal `lag_24`), driving a sharp improvement in `random_forest` (`hourly` MAE `6.772 kWh`, `daily_profile` MAE `14.424 kWh`, `weekly_profile` MAE `19.069 kWh`).
     - `lookback = 168h` uses `[1, 2, 3, 6, 12, 24, 48]` (**excluding `lag_168`** because `168 < 168` is false).

---

## 5. Unsupervised Anomaly Detection & Remediation Findings (Phase 3B.3)

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

## 6. Regime-Shift Analysis & Cross-Building Heterogeneity (Phase 3C Remediated)

### 6.1 Campus-Wide Zero Shift vs. Building-Specific Positive Load Shifts
- **Campus-Wide Effect:** All 29 core buildings drop from `381.02` valid zero hours per 1,000h in 2016 (`median 384.34/1,000h`, range `[366.98, 384.79]`) to `0.00/1,000h` in H1 2017. This campus-wide shift depresses 2016 unconditional training means ($\mu_{2016}$) and shifts 2016 anomaly score quantiles relative to active H1 2017 operations.
- **Cross-Building Variation:** Because 2016 valid-zero rates are nearly uniform across the 29 buildings (`20/29` within `384.0–384.8/1,000h`), 2016 zero frequency does **not** significantly explain cross-building differences in daily Random Forest MASE degradation (**Pearson $r = 0.2330$, $p = 0.2239$; Spearman $\rho = 0.3286$, $p = 0.0818$**). Instead, building-to-building differences in forecasting accuracy and anomaly alert rates reflect building-specific changes in positive active-load distributions and diurnal ramp amplitude.

### 6.2 Exploratory Cross-Phase Correlation & Outlier-Leverage Audit
- **Daily Random Forest MASE Degradation vs. Seasonal-Deviation LOF Alert Rate (`exploratory_outlier_sensitive_diagnostic`):**
  - **$N=29$ Linear Correlation:** Pearson $r = 0.6906$ ($p = 3.3802e-05$).
  - **$N=29$ Non-Parametric Rank Correlation:** Spearman $\rho = -0.1466$ ($p = 0.4480$); Kendall $\tau = -0.1284$ ($p = 0.3292$).
  - **Single-Outlier Leverage (`Panther_education_Vincent`):** `Panther_education_Vincent` (`daily_rf_mase_delta = +3.4928`, `seasonal_lof = 306.69/1,000h`) has OLS hat-matrix leverage **$h = 0.8507$** and **Cook's $D = 1.2558$**.
  - **Excluding `Panther_education_Vincent` ($N=28$):** Pearson $r = 0.2518$ ($p = 0.1962$, non-significant); Spearman $\rho = -0.2740$ ($p = 0.1582$).
  - **Diagnostic Conclusion:** Pearson's apparent positive association is dominated by one influential building (`Panther_education_Vincent`) and is **not** a robust general relationship across the 29 buildings (where the 10 RF-winning buildings have a median seasonal LOF rate of `27.22/1,000h` vs. `18.30/1,000h` for the 19 baseline-winning buildings). No causal relationship is implied or supported.
- **Daily Random Forest Absolute Bias vs. MAE Degradation ($N=29$):** Pearson $r = 0.5802$ ($p = 0.0010$); Spearman $\rho = 0.7537$ ($p = 2.3581e-06$) — descriptive error-decomposition diagnostic reflecting $|\text{mean\_error}| \le \text{MAE}$.

### Table 7: Per-Building Integrated Diagnostic Profile (29 Core Buildings, `lookback=168h`, `1.0pct` Operating Point)

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

## 7. Unsupported Claims, Scientific Limitations & Threats to Validity

### 7.1 Claims Explicitly Not Supported by the Completed Experiments
1. **No Confirmed Fault Detection or Supervised Anomaly Accuracy:** No ground-truth anomaly labels exist in BDG2 Panther (`ground_truth_anomaly_labels == 'none'`). Flagged hours and contiguous events represent unlabeled statistical or geometric departures from 2016 training distributions, and supervised metrics (precision, recall, F1, ROC-AUC, accuracy) cannot be claimed.
2. **No Universal Forecasting Winner Across Horizons:** Random Forest (`lookback=168h`) improves upon persistence at $H=1$ (`MASE 0.937` vs. `1.017`), whereas daily seasonal naive (`MASE 1.644`) and weekly seasonal naive (`MASE 1.937`) outperform all frozen-origin recursive candidates in aggregate macro metrics at $H=24$ and $H=168$.
3. **No Robust Cross-Building Linear or Causal Coupling Between Forecasting Degradation and LOF Alert Rates:** The $N=29$ Pearson correlation ($r = 0.6906$) between `daily_rf_mase_delta` and `seasonal_deviation` `lof_novelty` validation alert rate is dominated by `Panther_education_Vincent` ($h = 0.8507$, Cook's $D = 1.2558$; $N=28$ Pearson $r = 0.2518, p = 0.1962$; $N=29$ Spearman $\rho = -0.1466, p = 0.4480$).
4. **No Attribution of All Forecasting Error to the 2016 Zero Regime Alone:** Multi-step errors in Phase 3B.2 are co-determined by model-configuration choices (`SARIMA(1,0,0)x(0,0,0,0)` non-seasonal AR(1) mean-reversion, `XGBoost(n_estimators=40, lr=0.05)` $H=1$ under-convergence, and strict `lag < lookback` lag filtering) and building-specific changes in positive active-load distributions (`2016` zero rate vs. `daily_rf_mase_delta`: Pearson $r = 0.2330, p = 0.2239$).
5. **No Weather-Exogenous Forecasting Uplift Claim:** Although complete-case weather-aware forecasting rules were specified in Phase 2.7/3A, Phases 3B.1 and 3B.2 evaluated electricity-only baselines and candidates; weather-augmented forecasting remains an unexecuted future extension.
6. **No Held-Out Test Partition (`H2 2017`) or Multi-Year Generalization Claim:** The held-out test partition (`2017-07-01 00:00:00` through `2017-12-31 23:00:00`) remains untouched (`not_run`). Findings are descriptive of H1 2017 validation on the 29 UCF Panther educational buildings.

---

## 8. Reproducibility & Audit Trail Summary

- **Verified Runtime Environment:** Python `3.11.15` on `macOS-27.0.1-arm64-arm-64bit` (`numpy==2.4.6`, `pandas==3.1.0`, `scipy==1.17.1`, `scikit-learn==1.9.1`, `statsmodels==0.15.0`, `xgboost==3.2.0`, `pytest==8.4.2`).
- **Protected Artifact Ledger:** `28` cryptographically verified files (`6` datasets, `6` configurations, `8` accepted JSON reports, `5` historical/superseded JSON reports, `3` specification and synthesis Markdown documents).
- **In-Memory Deterministic Parity:** Verified 100% agreement between source modules and accepted reports on disk for Phase 3A (`783` coverage rows), Phase 3B.1 (`261` baseline rows), Phase 3B.2 (`783` candidate rows), Phase 3B.3 (`522` core + `18` sensitivity anomaly rows), and Phase 3C (`3C-remediated-v1`).
- **Verification Command:** `.venv/bin/pytest -v && .venv/bin/python scripts/run_phase3d_reproducibility.py --verify-only`.
