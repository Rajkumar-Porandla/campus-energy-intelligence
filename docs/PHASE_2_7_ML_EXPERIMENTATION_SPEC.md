# Phase 2.7 — ML Readiness & Experimental Design

**Status: Revised specification submitted for final approval.** The split,
horizons, weather-aware comparison, and alert-rate sensitivity levels have
been approved. This document adds operational definitions requested in review;
it does not authorize or run experiments, train models, or create predictions.

## Scope and analysis population

- Dataset: immutable raw-reading-derived BDG2 UCF Panther canonical series.
- Primary population: the configured 29 educational buildings.
- Quintin remains a separate sensitivity population and is never included in
  core results.
- Target: the source hourly `meter_reading`, documented by BDG2 in kWh and
  associated with its supplied timestamp. Retain the source value as given;
  do not convert units or infer a different interval convention.
- Missing electricity readings remain missing. Zero is a valid observation.
  Do not interpolate, impute, or aggregate source readings.
- Timestamps are treated as the source's ordered, timezone-naive hourly index.
  Resolve the timezone/DST interpretation before making local-clock claims.

## Chronological evaluation

Use the same calendar boundaries for every core building; do not randomly
shuffle observations or buildings:

| Partition | Inclusive source timestamps | Purpose |
|---|---|---|
| Train | 2016-01-01 00:00 through 2016-12-31 23:00 | Fit model parameters and train-only transformations |
| Validation | 2017-01-01 00:00 through 2017-06-30 23:00 | Select model settings; assess frozen anomaly operating points |
| Test | 2017-07-01 00:00 through 2017-12-31 23:00 | One final, held-out comparison after choices are frozen |

Model selection and all preprocessing choices are frozen before test scoring.
For final test evaluation, refit using train plus validation only. During
validation/test, observations at or before an origin may be used if they
would have arrived by that origin; target-period observations after the origin
must never be inputs. Keep each building's timeline intact and report its
results separately before any across-building summary.

The split provides one training year and only six held-out months. It does not
support strong claims about annual generalization or multi-year robustness.

## Forecasting targets, horizons, and baselines

Forecast the reported hourly energy value for each building independently.
The proposed primary task is the next 24 hourly values from a forecast origin
(a 24-step profile). Report 1-hour-ahead as a secondary task and a 168-hour
profile as exploratory only.

For horizon `H` (24 or 168 hours), define each block's first target timestamp
as `block_start = partition_start + k * H hours`, for integer `k >= 0` while
the full block remains in the partition. Its forecast origin is the preceding
hour, `block_start - 1 hour`; its target timestamps are the inclusive hourly
range from `block_start` through `block_start + (H - 1) hours`. Blocks do not
overlap. For example, the first validation 24-hour block has origin
2016-12-31 23:00 and targets 2017-01-01 00:00 through 2017-01-01 23:00.
The first test block has origin 2017-06-30 23:00 and targets beginning
2017-07-01 00:00. At later block origins, observations from earlier blocks may
be used only if they would have been observed by that origin. The 1-hour task
uses every eligible hourly target timestamp in the partition: origin `t - 1
hour`, target `t`. All models for a task use the same origins and target mask.

Candidate seasonal-naive baselines, defined using observations available at
the origin:

- **Persistence:** repeat the last observed hourly value; reference for the
  1-hour task.
- **Daily seasonal naive (period 24):** repeat the corresponding values from
  the preceding 24-hour cycle; primary baseline for the 24-hour profile.
- **Weekly seasonal naive (period 168):** repeat the corresponding values
  from the preceding 168-hour cycle; baseline for the exploratory week.

Compare candidate forecasting algorithms (SARIMA, Random Forest, and XGBoost)
against the relevant baseline per building and task. No algorithm is presumed
to perform better. Any transformation, scaler, feature selection, or
hyperparameter search is fitted/selected on training/validation data only.

### Forecast missingness

- Do not create target values for missing readings.
- Before scoring, register a common historical lookback for each task equal to
  the longest input history required by any included model or baseline. An
  origin is eligible only when every electricity value in that common
  lookback and every target in that task's scoring window is observed. Report
  excluded origins and target coverage per building and task. This common
  lookback concerns per-origin prediction inputs; all methods also use the
  same permitted training partition for model fitting.
- Do not give a model a model-specific eligibility mask. If a candidate
  requires a longer lookback than the registered common lookback, increase
  the lookback and recompute the shared mask for every comparator before
  validation/test, or omit that candidate from that task's comparison.
- Score every candidate and baseline on the identical eligible origins and
  target timestamps. Do not let model-specific missingness change the
  comparison set.
- Never replace an actual zero with missing or vice versa.

### Metrics

For each building, define the MASE denominator on the training partition as
the mean absolute period-24 seasonal-naive error over pairs `(y[t], y[t-24])`
where both observed values are nonmissing. Compute MASE as the evaluation
MAE divided by this denominator. If there are no valid training pairs or the
denominator is zero, report MASE as undefined; do not substitute another
scale. State the valid training-pair count and evaluation observation count.

Report per-building MAE, RMSE, mean error (bias), and MASE. Summarize with the
unweighted macro mean and median across buildings, plus the per-building
range. Do not pool kWh across buildings or use MAPE as a primary metric
because valid zero readings make it undefined. Treat the held-out results as
descriptive for this site and period, not as evidence of broad generalization.

## Weather-aware comparison

Keep weather separate and join only on the exact source `timestamp` and
`site_id=Panther` key. The migrated weather fields are historical observed
measurements, not future weather forecasts. For a forecast at origin `o`,
only observed weather measurements timestamped at or before `o` may be used;
never use observed weather at target timestamps after `o` as if it were known
in advance. The dataset provides no archived forecast vintages. If genuine
future weather forecasts become available later, they may be used only with
their issue/availability timestamps proving they were issued by `o`, and must
be evaluated as a separately identified experiment.

Proposed primary weather candidates are air temperature, dew temperature,
wind speed, 1-hour precipitation depth, and sea-level pressure. Wind direction
may be a separately specified sensitivity feature because it is directional.
Exclude cloud coverage (42.99% missing) and 6-hour precipitation depth
(96.32% missing) from the primary comparison. Do not interpret missing
weather as zero.

Do not impute weather in the primary comparison. Before validation, register a
common historical weather-input window and feature recipe. For each task,
first form a paired eligibility mask requiring the common electricity
history, complete electricity target window, and complete selected weather
inputs in that window, all timestamped no later than the origin. Evaluate the
weather-aware model and its electricity-only counterpart on exactly the same
origins and target timestamps from this mask. Report the resulting coverage
loss against the electricity-only mask. If a later approved sensitivity
analysis imputes weather, fit the imputer on training data only and report it
separately.

The weather-aligned timestamps are complete, but timestamps do not carry a
timezone offset. Resolve timezone/DST semantics before relying on local
calendar or operational-time interpretations.

## Unsupervised anomaly experiments (no ground-truth labels)

Use per-building series and chronologically separate fit/reference periods
from validation and test. Candidate methods are statistical anomaly scores,
Isolation Forest, and Local Outlier Factor in novelty mode. Evaluate raw
consumption and, if approved, deviations relative to a seasonal reference
estimated from training data only. Do not fit thresholds or transformations
using test observations.

There are no verified anomaly labels. Therefore do not report accuracy,
precision, recall, F1, ROC-AUC, or claim that detected points are true
anomalies. Report scores and, for an explicitly selected threshold, flagged
hours per 1,000 observed hours, flagged fraction, event counts/durations
(consecutive flags grouped within a building), building distribution, and
cross-building coincidence. Treat coincidences as review context, not proof
of a weather or operational cause.

For the approved sensitivity analysis, orient scores so larger values mean
more anomalous and set thresholds from training-only upper-tail score
quantiles: the 99.5th, 99th, and 98th percentiles (nominal 0.5%, 1%, and 2%
training alert-rate operating points). Freeze them before validation/test;
do not tune them against unlabeled validation or test scores. Report all three
operating points. The percentages describe training-score threshold
operating points only; they are not validated detection accuracy, expected
future alert rates, or evidence that flagged readings are true anomalies.

## Building-level comparisons

- Report coverage, missingness, zero counts, and forecasting/anomaly results
  separately for every core building; include Quintin only as a separate
  sensitivity result.
- Compare like with like: all core buildings are Panther education buildings.
  Do not generalize to other building categories or campuses.
- Present absolute kWh errors per building and macro summaries. Do not sum
  building readings into a campus total or rank consumption without
  acknowledging differences in building size and use intensity.
- Do not normalize by floor area unless the area field and its units are
  independently validated; any normalized analysis is a separate sensitivity
  result.

## Leakage and reproducibility checklist

Before any later experiment, record dataset checksums, configuration, split
boundaries, eligible origins, target coverage, features, model settings, and
random seeds. Confirm that:

1. No random split, future target, future weather observation, or test-fitted
   transform enters training or model selection.
2. Rolling/seasonal features use only data available at the forecast origin.
3. Missing-value handling preserves the distinction between zero and missing
   and is learned on training data only if a future sensitivity test uses it.
4. All compared forecasts use identical buildings, origins, horizons, and
   scoring masks; coverage differences are reported rather than hidden.
5. Test results are computed only after model and threshold choices are
   frozen, and no model is described as validated without ground-truth evidence.

## Final approval gate

The split boundaries, forecast horizons, weather-aware complete-case
comparison, and 0.5%, 1%, and 2% training-derived anomaly operating points
have been approved. This revision makes their origins, target masks, input
availability, and metric definitions explicit. Final approval of this revised
specification is required before Phase 3. No experiment or model
implementation is authorized by this document.
