import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("streamlit_app/data/gold_predictions.csv")

df = load_data()

st.title("⚙️ Enterprise Predictive Maintenance Dashboard")
st.markdown("**NASA CMAPSS Turbofan Engine Fleet · Ridge Regression RUL Model · MLflow Registered**")
st.markdown("---")

st.sidebar.header("Filters")
subset_filter = st.sidebar.selectbox("Subset", ["All", "FD001", "FD003"])
split_filter  = st.sidebar.selectbox("Split", ["All", "train", "test"])

df_f = df.copy()
if subset_filter != "All": df_f = df_f[df_f["subset"] == subset_filter]
if split_filter  != "All": df_f = df_f[df_f["split"]  == split_filter]

COLORS = {"CRITICAL":"#E84855","WARNING":"#F4A261","MONITOR":"#2A9D8F","HEALTHY":"#2E4057"}

st.header("Fleet Status Overview")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Engines",      df_f["unit_id"].nunique())
c2.metric("🔴 CRITICAL obs",   int((df_f["alert_urgency"]=="CRITICAL").sum()))
c3.metric("🟡 WARNING obs",    int((df_f["alert_urgency"]=="WARNING").sum()))
c4.metric("Engines in Alert",  df_f[df_f["alert_30"]==1]["unit_id"].nunique())
c5.metric("Cost at Risk",      f"${df_f['cost_at_risk_usd'].sum()/1e6:.1f}M")

urgency_counts = df_f["alert_urgency"].value_counts().reindex(["CRITICAL","WARNING","MONITOR","HEALTHY"], fill_value=0)
fig, ax = plt.subplots(figsize=(10,1.5))
left, total = 0, urgency_counts.sum()
for tier, count in urgency_counts.items():
    pct = count/total
    ax.barh(0, pct, left=left, color=COLORS[tier], height=0.6)
    if pct > 0.02:
        ax.text(left+pct/2, 0, f"{tier}\n{count:,}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    left += pct
ax.set_xlim(0,1); ax.axis("off")
st.pyplot(fig); plt.close()

st.markdown("---")
st.header("Per-Engine RUL Analysis")

engine_id = st.selectbox("Select Engine (unit_id)", sorted(df_f["unit_id"].unique()))
subset_id = st.selectbox("Subset", df_f[df_f["unit_id"]==engine_id]["subset"].unique())
eng = df_f[(df_f["unit_id"]==engine_id)&(df_f["subset"]==subset_id)].sort_values("time_cycles")

if not eng.empty:
    latest = eng.iloc[-1]
    c1,c2,c3 = st.columns(3)
    c1.metric("Predicted RUL",  f"{latest['rul_predicted']:.1f} cycles")
    c2.metric("Alert Status",   latest["alert_urgency"])
    c3.metric("Cost at Risk",   f"${latest['cost_at_risk_usd']:,.0f}")

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(eng["time_cycles"], eng["rul_actual"],    color="#2E4057", lw=2, label="Actual RUL")
    ax.plot(eng["time_cycles"], eng["rul_predicted"], color="#E84855", lw=2, ls="--", label="Predicted RUL")
    ax.axhline(30, color="#F4A261", lw=1.5, ls=":", label="Alert threshold (30 cycles)")
    ax.fill_between(eng["time_cycles"], eng["rul_predicted"]-10, eng["rul_predicted"]+10, alpha=0.15, color="#E84855")
    ax.set_xlabel("Time (cycles)"); ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine {engine_id} ({subset_id}) — RUL Trajectory")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

st.markdown("---")
st.header("Model Performance Summary")
st.markdown("""
**Model:** RidgeCV (α=100, 5-fold CV) vs. LSTM (PyTorch 64→32, 60 epochs)  
**Test set:** CMAPSS FD001, n=10,096 · **Paired t-test:** t=−41.49, p<0.0001
""")

metrics    = ["Test RMSE\n(cycles ↓)","Test MAE\n(cycles ↓)","Alert Recall ↑","Alert Precision ↑","Alert F1 ↑"]
ridge_vals = [24.08, 17.24, 0.152, 1.000, 0.260]
lstm_vals  = [40.80, 28.85, 0.058, 1.000, 0.085]
x, w = np.arange(len(metrics)), 0.35

fig, ax = plt.subplots(figsize=(11,5))
b1 = ax.bar(x-w/2, ridge_vals, w, label="Ridge", color="#2E4057", alpha=0.85)
b2 = ax.bar(x+w/2, lstm_vals,  w, label="LSTM",  color="#E84855", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_title("Ridge vs. LSTM — Test Set Performance", fontweight="bold"); ax.legend()
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
st.pyplot(fig); plt.close()

st.markdown("---")
st.caption("NASA CMAPSS (CC0) · RidgeCV α=100 · Databricks→Snowflake→MLflow · Gretchen Kolthoff")