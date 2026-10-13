**[Live Dashboard →](https://predictive-maintenance-dashboard-i.streamlit.app)**

# Enterprise Predictive Maintenance & IoT Sensor Optimization

End-to-end pipeline for turbofan engine remaining useful life (RUL) prediction on the NASA CMAPSS simulation: a Databricks/PySpark bronze → silver → gold medallion architecture with Delta Lake and Unity Catalog, MLflow experiment tracking and model registry, and a Streamlit dashboard. Headline result on the FD001 test set: **LSTM last-cycle RMSE 13.05 ± 0.33 vs. 17.51 for a Ridge baseline**, with the difference tested per engine.

---

## What This Solves

Turbofan engines degrade gradually, and a maintenance planner needs two things: an estimate of how many cycles an engine has left, and an alert early enough to schedule removal before failure. In aviation, a missed failure is primarily a **safety** risk, not just a downtime cost. Time-based maintenance is safe but removes engines that still have useful life. This project predicts remaining useful life (RUL) from sensor history and turns it into an alert with an explicit, validated miss rate.

This pipeline was built on the NASA CMAPSS turbofan degradation simulation. It shows the method on simulated data and is not a certified prognostic system.

---

## Stack

**Data & pipeline:** PySpark · Databricks · Delta Lake · Unity Catalog
**Modeling:** scikit-learn (Ridge, logistic regression) · PyTorch (LSTM) · MLflow tracking + Unity Catalog model registry
**Quality gates:** null/bound checks plus engine-identity checks that block Silver writes on failure
**Serving:** Gold Delta table → Streamlit dashboard. Gold was also loaded to Snowflake during a trial period; that trial has expired, and the write code (credentials read from a Databricks secret scope) is kept in notebook 05.

---

## Pipeline

```
Raw NASA CMAPSS .txt files (Unity Catalog volume)
  ↓  01 Bronze: schema-on-read, lineage columns; RUL truth file keyed by engine
Bronze Delta tables
  ↓  03 Silver: engine = (split, unit_id); RUL from failure (train) or RUL_FD001.txt (test);
  ↓     cap at 125; 30-cycle rolling features; train-only scaling; quality gates
Silver Delta tables
  ↓  04 Modeling: Ridge vs. LSTM, engine-grouped CV, frozen alert threshold, MLflow
  ↓  05 Gold: served model's predictions for FD001 test engines
Gold Delta table → Streamlit dashboard
```

---

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `bronze_ingestion` | PySpark ingestion from a Unity Catalog volume into Bronze Delta tables with lineage columns; attaches `unit_id` to the RUL truth file |
| 02 | `eda` | RUL distribution, zero-variance detection, sensor–RUL correlation, multicollinearity |
| 03 | `silver_layer` | Feature engineering keyed on (`split`, `unit_id`), test labels from the truth file, quality gates that block bad writes |
| 04 | `modeling` | Ridge vs. LSTM with engine-grouped CV, last-cycle and PHM08 metrics, per-engine significance tests, alert threshold tuning, grouped SHAP |
| 05 | `gold_layer` | Gold predictions for test engines with the frozen threshold, MLflow model registration, Snowflake write code |

---

## EDA-Driven Decisions

* 8 of 26 columns are constant in FD001's single operating condition and were dropped.
* `sensor_14` was dropped: r = 0.963 with `sensor_9`.
* Strong linear signal (|r| up to 0.775 with capped RUL) made Ridge a credible baseline rather than a straw man.
* Alert labels are imbalanced: 15.0% of training rows and 2.5% of test rows are within 30 cycles of failure, so accuracy is not a useful metric for the alert.
* Rolling features: 30-cycle mean and std per sensor, computed within each engine. Min-max scaling is fit on training engines only.

---

## Scope (what was trained and evaluated)

| | Engines | Rows | Labels |
|---|---|---|---|
| Train | 100 FD001 run-to-failure engines | 20,631 | RUL = cycles until that engine's failure, capped at 125 |
| Test | 100 FD001 engines, truncated before failure | 13,096 | RUL from `RUL_FD001.txt` + cycles remaining in the record |

* **FD003 is ingested (bronze/silver) but not modeled or evaluated.**
* Model choices were made on training engines only, with cross-validation **grouped by engine**. That covers Ridge alpha, alert thresholds, and LSTM early stopping. The test set was used once, for the numbers below.
* The Gold table, MLflow registry entry, and dashboard contain **test-engine predictions only** (13,096 rows). Training rows would be in-sample fits, so they are excluded.

---

## Model Selection & Results

### 1. RUL accuracy — standard CMAPSS protocol (last cycle of each test engine, n = 100)

| Metric | Ridge (α = 100) | LSTM (64→32), 3 seeds |
|---|---|---|
| RMSE vs. truth capped at 125 | 17.51 | **13.05 ± 0.33** |
| RMSE vs. raw truth | 18.83 | **14.73 ± 0.48** |
| MAE (capped) | 14.01 | **9.95 ± 0.49** |
| PHM08 score (raw truth, lower is better) | 664 | **308 ± 7** |
| Train RMSE (all rows) | 18.98 | 13.50 ± 0.93 |

For context, published FD001 results include a deep CNN at RMSE 12.61 / score 273.7 and a BiLSTM at 13.65 / 295 (Li, Ding & Sun, 2018, *Reliability Engineering & System Safety*; BiLSTM by Wang et al.; both as tabulated in Zhang, Song & Li, 2022, *IEEE Trans. Instrumentation & Measurement*). Papers differ in their RUL cap and in whether they report a best run or an average, so treat this as a sanity check, not a ranking.

### 2. RUL accuracy — every test cycle with ≥ 30 cycles of history (10,196 rows, same rows for both models)

| Metric | Ridge | LSTM |
|---|---|---|
| RMSE | 19.82 | **14.88 ± 1.09** |
| MAE | 15.35 | **11.83 ± 0.83** |

**Is the difference real?** Rows from the same engine are strongly autocorrelated, so the unit of analysis is the **engine**, not the row:
* The LSTM has lower per-engine RMSE on 67–85 of 100 test engines, depending on seed. Wilcoxon signed-rank p < 0.0001 for every seed.
* At the last cycle, absolute errors favor the LSTM with Wilcoxon p ≤ 0.002 for every seed.
* Engine-level bootstrap (10,000 resamples of engines): the pooled RMSE gap (Ridge − LSTM) is 3.9 to 6.1 cycles. Every seed's 95% CI excludes zero (lowest lower bound across seeds: 2.5).

### 3. Maintenance alert (label: true RUL ≤ 30 cycles)

The alert fires when predicted RUL ≤ τ. τ is chosen **before touching test**: the lowest threshold that reaches 90% recall on out-of-fold training predictions (Ridge, logistic) or on 20 held-out training engines (LSTM). It is then frozen and applied to test.

| Test metric (332 positive rows from 25 engines) | Ridge, τ = 50.3 | LSTM, τ ≈ 31–33 | Logistic classifier |
|---|---|---|---|
| Row recall | 0.75 (248/332) | 0.76 ± 0.07 | 0.81 (270/332) |
| Row precision | 0.73 | 0.84 ± 0.06 | 0.84 |
| Row F1 | 0.74 | 0.80 ± 0.03 | 0.82 |
| Engines alerted at last cycle (of 25 truly ≤ 30) | 21 | 21–24 | 25 |
| Healthy engines falsely alerted (of 75) | 1 | 1–2 | 1 |

Reference point: thresholding Ridge at the naive τ = 30 catches only about a third of positive rows. A regression trained on capped RUL is conservative near the boundary, so the threshold must be tuned rather than set equal to the label horizon.

**Validation-to-test gap.** Thresholds tuned for 90% recall reached 75–81% row recall on test. Most of that gap is a change in the positive mix, not a model failure. Test engines are cut off mid-life, so 63% of test positives sit at RUL 21–30, right at the alert boundary, versus 32% in training. Reweighting training recall by bucket to the test mix predicts about 82%. If a recall floor must hold, the threshold needs margin, for example tuning for 95%. Engine-level counts rest on only 25 engines, so their 95% intervals are wide: 21/25 has a Wilson interval of 0.65 to 0.94.

### Decision

* **RUL estimate: LSTM** (seed 1, chosen by lowest validation RMSE), the champion candidate. It is more accurate on every regression metric, the gap holds per engine, and train ≈ validation ≈ test error, so there is no sign of overfitting.
* **Ridge stays as the interpretable baseline.** It's the model to check the LSTM against and to explain in terms of sensors.
* **Alert: a dedicated classifier is competitive** with a thresholded regressor (row F1 0.82 vs. 0.80; average precision 0.91 vs. 0.83 for Ridge). The next step is to tune both on the same validation engines and pick by recall at a fixed false-alarm rate.

### What drives the Ridge predictions (SHAP)

Mean |SHAP| **grouped by sensor**: sensor 8 > 4 > 13 > 11 > 12. Grouping matters. `sensor_8_roll_mean` and `sensor_13_roll_mean` correlate at r = 0.987, so splitting credit between them individually is arbitrary. By feature family, current normalized readings carry 48% of attribution, rolling means 41%, and rolling std 12%.

---

## Key Tradeoffs

**RUL cap at 125.** Early in life an engine shows no measurable degradation, so a linearly declining label asks the model to predict something the sensors can't see. A piecewise-linear target (constant, then declining) is the standard treatment for CMAPSS, and caps of 125 or 130 are the most common in the literature. The cost: the model can't distinguish an engine with 200 cycles left from one with 125, which is acceptable because maintenance decisions happen well inside that window.

**FD001 only.** FD002 and FD004 run under six operating conditions, so raw sensor levels shift with the regime. The approach would be to cluster on the three operational settings, then normalize each sensor within its regime before building windows.

**LSTM vs. Ridge.** The LSTM wins on accuracy. Ridge costs less to serve and is easier to explain to a reviewer. Ridge is registered in Unity Catalog (`ridge_rul_predictor_cmapss` v3) and is what Gold serves today. The LSTM runs are logged in MLflow, and moving Gold to the LSTM is the next step: it needs 30 cycles of history, so each engine's first 29 cycles need a fallback.

**Alert threshold.** The alert horizon (30 cycles) and the decision threshold τ are separate choices. τ is tuned to a recall target on training engines and logged as an MLflow parameter so Gold uses exactly the evaluated value.

---

## Corrections (October 2026)

An earlier version of this README reported Ridge test RMSE 24.08, LSTM 40.80, and alert recall 0.152. Those numbers came from two bugs, now fixed:
1. **Engine identity.** Train and test files reuse unit IDs 1–100 for different engines. Silver grouped by `unit_id` alone, which merged each train engine with a test engine. That corrupted RUL labels for 55% of test rows and 15 training engines, and mixed their rolling-window features. Test RUL also ignored `RUL_FD001.txt`. Silver now keys on (`split`, `unit_id`), joins the truth file, and runs a quality check that every train engine ends at RUL 0.
2. **Model comparison alignment.** The Ridge vs. LSTM comparison paired predictions by row position, but LSTM windows skip each engine's first cycles, so rows didn't line up. Predictions are now joined on (`unit_id`, `time_cycles`), and significance is tested per engine instead of with a row-level t-test on 10,096 autocorrelated rows.

---

## Data

NASA CMAPSS Turbofan Engine Degradation Simulation (Saxena et al., 2008), FD001 and FD003; FD001 used for modeling. [Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (CC0).

---

## Author

**Gretchen Kolthoff** · [GitHub](https://github.com/GretchenK20)
