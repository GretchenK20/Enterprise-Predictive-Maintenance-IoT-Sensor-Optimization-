import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("streamlit_app/data/gold_predictions.csv")


df = load_data()
TAU = float(df["alert_threshold"].iloc[0])
COLORS = {"CRITICAL": "#E84855", "WARNING": "#F4A261", "MONITOR": "#2A9D8F", "HEALTHY": "#2E4057"}
TIERS = ["CRITICAL", "WARNING", "MONITOR", "HEALTHY"]

st.title("⚙️ Turbofan Predictive Maintenance Dashboard")
st.markdown(
    "**NASA CMAPSS FD001 · 100 held-out test engines · "
    "Served model: Ridge v2 baseline (MLflow `ridge_rul_predictor_cmapss` v3)**"
)
st.caption(
    "Test engines only (no in-sample training rows). True RUL from `RUL_FD001.txt`. "
    f"Alert fires when predicted RUL ≤ τ = {TAU:.1f} cycles; τ was tuned for 90% recall "
    "on training engines, never on test."
)
st.markdown("---")

# ---------------- Fleet snapshot: each engine at its LAST observed cycle ----------------
last = df[df["is_last_cycle"] == 1]
st.header("Fleet Snapshot (latest cycle of each engine)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines", last["unit_id"].nunique())
c2.metric("🔴 Alerting now", int(last["alert"].sum()))
c3.metric("Truly within 30 cycles", int((last["rul_true_raw"] <= 30).sum()))
tp = int(((last["alert"] == 1) & (last["rul_true_raw"] <= 30)).sum())
c4.metric("Caught / missed", f"{tp} / {int((last['rul_true_raw'] <= 30).sum()) - tp}")

counts = last["alert_urgency"].value_counts().reindex(TIERS, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 1.5))
left, total = 0, counts.sum()
for tier, count in counts.items():
    pct = count / total
    ax.barh(0, pct, left=left, color=COLORS[tier], height=0.6)
    if pct > 0.04:
        ax.text(left + pct / 2, 0, f"{tier}\n{count}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    left += pct
ax.set_xlim(0, 1)
ax.axis("off")
st.pyplot(fig)
plt.close()
st.caption(f"Urgency tiers by predicted RUL: CRITICAL ≤ 10 · WARNING ≤ τ ({TAU:.1f}) · "
           f"MONITOR ≤ τ + 30 · HEALTHY above.")

# ---------------- Per-engine trajectory ----------------
st.markdown("---")
st.header("Per-Engine RUL Trajectory")
engine_id = st.selectbox("Select test engine (unit_id)", sorted(df["unit_id"].unique()))
eng = df[df["unit_id"] == engine_id].sort_values("time_cycles")
latest = eng.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (latest cycle)", f"{latest['rul_predicted']:.1f} cycles")
c2.metric("True RUL (RUL_FD001.txt)", f"{latest['rul_true_raw']:.0f} cycles")
c3.metric("Status", latest["alert_urgency"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(eng["time_cycles"], eng["rul_actual"], color="#2E4057", lw=2, label="True RUL (capped at 125)")
ax.plot(eng["time_cycles"], eng["rul_predicted"], color="#E84855", lw=2, ls="--", label="Predicted RUL")
ax.axhline(TAU, color="#F4A261", lw=1.5, ls=":", label=f"Alert threshold τ = {TAU:.1f}")
ax.axhline(30, color="#888", lw=1, ls="-.", label="Failure horizon (30 cycles)")
ax.set_xlabel("Time (cycles)")
ax.set_ylabel("RUL (cycles)")
ax.set_title(f"Test engine {engine_id} — observed until cycle {int(latest['time_cycles'])}")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close()

# ---------------- Model results ----------------
st.markdown("---")
st.header("Model Evaluation (FD001 test set)")
st.markdown("""
| Metric | Ridge (served baseline) | LSTM (champion candidate, mean ± sd, 3 seeds) |
|---|---|---|
| Last-cycle RMSE, standard protocol (100 engines) | 17.51 | **13.05 ± 0.33** |
| PHM08 score (lower is better) | 664 | **308 ± 7** |
| RMSE on aligned test rows (10,196) | 19.82 | **14.88 ± 1.09** |
| Alert recall / precision (rows) | 0.75 / 0.73 | 0.76 / 0.84 |
| Engines alerted of 25 in window / false alarms of 75 | 21 / 1 | 21–24 / 1–2 |

**Significance** is tested with the engine as the unit, not the row: the LSTM has lower per-engine
RMSE on 67–85 of 100 engines (Wilcoxon p < 0.0001 for every seed), and an engine-level bootstrap of
the RMSE gap (3.9–6.1 cycles) excludes zero. Gold currently serves Ridge; migrating it to the LSTM is
the next step.
""")

st.markdown("---")
st.caption("NASA CMAPSS (CC0) · Databricks (Delta, Unity Catalog) · MLflow · Gretchen Kolthoff")
