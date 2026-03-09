"""
ModelForge Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append(".")

from src.tracking.tracker import ExperimentTracker

st.set_page_config(
    page_title="ModelForge",
    page_icon="⚗️",
    layout="wide",
)

st.title("⚗️ ModelForge — ML Experiment Dashboard")
st.markdown("---")

tracker     = ExperimentTracker()
experiments = tracker.load_all()

if not experiments:
    st.warning("No experiments found. Run `python modelforge.py run --dataset ...` first.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
models = list(set(e["model"] for e in experiments))
selected_models = st.sidebar.multiselect("Model Types", models, default=models)

filtered = [e for e in experiments if e["model"] in selected_models]

# ── KPI Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Experiments", len(filtered))
col2.metric("Model Types",       len(set(e["model"] for e in filtered)))

best = max(filtered, key=lambda e: e["metrics"].get("accuracy",
                                    e["metrics"].get("f1", 0)), default=None)
if best:
    col3.metric("Best Accuracy", best["metrics"].get("accuracy", "N/A"))
    col4.metric("Best Model",    best["model"])

st.markdown("---")

# ── Experiment Table ─────────────────────────────────────────────────────────
st.subheader("📋 Experiment History")
rows = []
for e in filtered:
    row = {"ID": e["id"], "Timestamp": e["timestamp"],
           "Model": e["model"], "Training Time (s)": e["training_time"]}
    row.update(e["metrics"])
    rows.append(row)

df_exp = pd.DataFrame(rows)
st.dataframe(df_exp, use_container_width=True)

# ── Metric Comparison Chart ───────────────────────────────────────────────────
st.subheader("📊 Metric Comparison")
metric_cols = [c for c in df_exp.columns
               if c not in ("ID", "Timestamp", "Model", "Training Time (s)")]
if metric_cols:
    selected_metric = st.selectbox("Select Metric", metric_cols)
    fig = px.bar(
        df_exp, x="ID", y=selected_metric, color="Model",
        title=f"{selected_metric} by Experiment",
        labels={"ID": "Experiment ID"},
        text_auto=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Training Time Chart ───────────────────────────────────────────────────────
if "Training Time (s)" in df_exp.columns:
    st.subheader("⏱ Training Time per Experiment")
    fig2 = px.line(df_exp, x="ID", y="Training Time (s)", markers=True,
                   title="Training Time Trend")
    st.plotly_chart(fig2, use_container_width=True)
