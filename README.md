# Enterprise Predictive Maintenance & IoT Sensor Optimization

End-to-end MLOps pipeline for turbofan engine remaining useful life (RUL) prediction — built on NASA CMAPSS sensor data using a Databricks/PySpark bronze→silver→gold medallion architecture, with Snowflake as the serving layer and MLflow for experiment tracking and model governance.

---

## What This Solves

Unplanned industrial equipment failure costs manufacturers an average of $260K/hour in lost production. Current maintenance strategies force a bad choice: reactive repair (expensive, dangerous) or time-based preventive maintenance (wasteful). This project builds a third option — a data-driven system that forecasts *when* an engine will fail so maintenance can be scheduled in the optimal window, not too early and not too late.

The cost asymmetry shaped every modeling decision here. A false negative (missing an imminent failure) costs $5–10K/hour in downtime plus emergency repair at 3–9× planned maintenance rates. A false positive (unnecessary dispatch) costs maybe $500 in labor. That's why **recall is the primary metric on the alert layer**, not accuracy or precision.

---

## Stack

**Data & Pipeline:** PySpark · Databricks · Delta Lake · Unity Catalog · Snowflake  
**Modeling:** scikit-learn (RidgeCV) · PyTorch (LSTM) · MLflow  
**Governance:** Great Expectations pattern · MLflow Model Registry  
**Language:** Python · SQL

---

## Pipeline Architecture
Raw NASA CMAPSS .txt files (Unity Catalog Volume)
↓
Bronze Delta schema-on-read, lineage metadata, trailing null handling
↓ [data quality gate]
Silver Delta feature engineering, RUL cap, rolling windows, normalization
↓ [MLflow experiment tracking]
Gold Delta RUL predictions, alert tiers (CRITICAL/WARNING/MONITOR/HEALTHY), cost framing
↓
Snowflake PREDICTIVE_MAINTENANCE.GOLD.GOLD_PREDICTIONS — 75,043 rows
↓
MLflow Registry ridge_rul_predictor_cmapss v1 — production


---

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `bronze_ingestion` | PySpark schema-on-read ingestion from Unity Catalog volume → Bronze Delta tables with lineage metadata |
| 02 | `eda` | Hypothesis-driven EDA before any modeling decisions — RUL distribution, zero-variance detection, sensor-RUL correlation, multicollinearity heatmap |
| 03 | `silver_layer` | Feature engineering pipeline with quality-gated writes — catches data issues before they reach the model |
| 04 | `modeling` | RidgeCV vs. LSTM benchmark with MLflow tracking, SHAP feature importance, paired t-test for significance |
| 05 | `gold_layer` | Prediction generation, MLflow model registry, Snowflake write |

---

## EDA-Driven Decisions

I started with the business question, not the model. Every transformation in the Silver layer was justified by what the data showed — not applied by default.

**What EDA found:**
- 8 of 26 features were confirmed zero-variance constants in FD001's single operating condition — dropped
- RUL distribution is right-skewed with a flat healthy plateau from 0–150 cycles. Engines at RUL=300 are sensorially indistinguishable from engines at RUL=200 — no degradation signal. **RUL capped at 125 cycles** to focus the model on the actionable window
- Strong linear correlations confirmed up to |r| = 0.775 (sensor_11) — established Ridge as a legitimate baseline, not a straw-man
- Severe multicollinearity: sensor_9 ↔ sensor_14 at r = 0.963. sensor_14 dropped. This is also why LSTM was worth testing — its gating mechanism handles correlated features natively
- Alert class imbalance: ~10% positive rate — accuracy is a useless metric here

**Feature engineering (Silver layer):**
- 30-cycle rolling mean and std per sensor per engine — gives Ridge temporal context without sequence modeling
- Min-max normalization fit on training set only — prevented data leakage from test set (caught by quality checks on first write attempt)
- Binary alert label: RUL ≤ 30 cycles = imminent failure

---

## Model Selection & Results

I benchmarked two model classes that the EDA suggested were both viable:

| Metric | Ridge Regression | LSTM (PyTorch) |
|---|---|---|
| Test RMSE | **24.08 cycles** | 40.80 cycles |
| Test MAE | **17.24 cycles** | 28.85 cycles |
| Alert Recall | **0.152** (84/553 failures) | 0.058 (32/553 failures) |
| Alert Precision | 1.000 | 1.000 |
| Alert F1 | **0.260** | 0.085 |
| Paired t-test p-value | **0.0000** | — |

**Winner: Ridge Regression (RidgeCV, α = 100.0 selected via 5-fold CV)**

The LSTM overfit hard — train RMSE of 15.10 vs. test RMSE of 40.80 is a 25-cycle generalization gap. Ridge's gap is 3 cycles. The performance difference is statistically significant at p < 0.0001 (t = −41.49, paired t-test on 10,096 aligned test observations).

**Why Ridge won despite temporal data:** The 30-cycle rolling window features engineered in Silver gave Ridge enough temporal context to capture degradation linearly. Given the strong linear signal in the data (|r| up to 0.775), LSTM's complexity wasn't justified by the result.

**SHAP (Ridge top features):** sensor_9_roll_mean (4.0), sensor_3_roll_mean (3.1), sensor_4_roll_mean (2.5), sensor_7_roll_mean (2.1). The rolling mean features dominate — the Silver layer temporal engineering was the critical modeling decision, not the model itself.

**Deployment:** Ridge for production. LSTM retained as an experimental track — it may outperform on FD003's dual-fault mode where temporal complexity is more warranted.

---

## Key Tradeoffs

**RUL cap at 125:** Loses some information from healthy engines but eliminates label noise that would mislead the model. Standard for CMAPSS — justified empirically by the flat distribution above 125 cycles.

**FD001 + FD003 only:** FD002 and FD004 add 6 operating conditions which would require more complex normalization and domain adaptation. Scoped to single-condition subsets to keep the feature engineering honest.

**Ridge over LSTM:** LSTM could theoretically capture temporal dependencies better but overfitted on this dataset size (20K training rows). A larger dataset or stronger regularization might close the gap. I chose the model the data supported, not the one that sounds more impressive.

**Alert threshold at 30 cycles:** Arbitrary but operationally grounded — roughly one maintenance window. The model outputs continuous RUL scores, so this threshold can be adjusted based on the specific maintenance operation's lead time.

---

## Data

NASA CMAPSS Turbofan Engine Degradation Simulation — [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (CC0: Public Domain)  
Subsets: FD001 (single fault) + FD003 (dual fault) · 75,043 total rows in Gold

---

## Author

**Gretchen Kolthoff** · 
[GitHub](https://github.com/GretchenK20)
