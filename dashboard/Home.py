import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.data_loader import (
    load_driver_metrics, load_circuit_baselines,
    load_pit_losses, load_model_metadata,
)
from dashboard.utils.constants import DRIVER_NAMES
from dashboard.utils.charts import dark_layout

st.set_page_config(
    page_title="F1 AI Strategy System",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏎 F1 AI Strategy System")
st.caption("Season 2023–2025 · XGBoost v7 · 24 Circuits · 20 Drivers · 76,163 laps")

baselines  = load_circuit_baselines()
pit_losses = load_pit_losses()
metadata   = load_model_metadata()
low_conf   = metadata.get("low_confidence_circuits", [])
drivers_df = load_driver_metrics()

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Circuits",         "24")
c2.metric("Drivers Tracked",  "20")
c3.metric("Laps in Dataset",  "76,163")
c4.metric("Model RMSE",       "1.88 s")
c5.metric("Low-Conf Circuits", str(len(low_conf)))

st.divider()

# ── Architecture note ────────────────────────────────────────────────────────
with st.expander("How the system works", expanded=False):
    st.markdown("""
**Two-stage strategy ranking:**
1. **Physics score (70%)** — XGBoost v7 predicts every lap time for every candidate strategy,
   summed to a total race time. Strategies with shorter predicted times score higher.
2. **Historical prior (30%)** — Each strategy is scored against 2023–2025 data:
   stop frequency (35%), pit window alignment (40%), compound sequence (25%).

**Temporal split**: model trained on 2023–2024, validated on 2025 (no data leakage).
**Features**: 23 engineered features — tire age, compound degradation, fuel load, dirty-air flag, circuit baseline pace, and more.
    """)

# ── Charts ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Circuit Baseline Lap Times")
    sorted_circuits = sorted(baselines.items(), key=lambda x: x[1])
    names  = [c for c, _ in sorted_circuits]
    values = [v for _, v in sorted_circuits]
    colors = ["#F39C12" if n in low_conf else "#E8002D" for n in names]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Baseline: %{x:.3f}s<extra></extra>",
    ))
    dark_layout(fig, height=520)
    fig.update_layout(
        xaxis_title="Seconds",
        margin=dict(l=10, r=20, t=10, b=10),
    )
    # Legend annotation
    fig.add_annotation(
        x=max(values) * 0.98, y=2,
        text="🟠 Low confidence",
        showarrow=False, font=dict(color="#F39C12", size=11),
        xanchor="right",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Driver Performance Overview")
    df = drivers_df.copy()
    df["full_name"] = df["driver"].map(DRIVER_NAMES).fillna(df["driver"])
    df["podium_pct"] = (df["podium_rate"] * 100).round(1)
    df["win_pct"]    = (df["win_rate"]    * 100).round(1)

    fig2 = px.scatter(
        df,
        x="pace_score",
        y="consistency_score",
        size="total_races",
        color="driver_performance_score",
        color_continuous_scale="Plasma",
        text="driver",
        hover_name="full_name",
        hover_data={
            "driver": False,
            "driver_performance_score": ":.3f",
            "podium_pct": True,
            "win_pct": True,
            "total_races": True,
            "pace_score": ":.3f",
            "consistency_score": ":.3f",
        },
        labels={
            "pace_score": "Pace Score",
            "consistency_score": "Consistency Score",
            "driver_performance_score": "Overall Score",
        },
    )
    fig2.update_traces(textposition="top center", textfont_size=10)
    fig2.add_hline(y=0.5, line_dash="dot", line_color="#555")
    fig2.add_vline(x=0.5, line_dash="dot", line_color="#555")
    dark_layout(fig2, height=520)
    fig2.update_coloraxes(colorbar=dict(tickfont=dict(color="white"),
                                        title=dict(text="Score", font=dict(color="white"))))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Pit loss bar ──────────────────────────────────────────────────────────────
st.subheader("Pit Stop Time Loss by Circuit")
pit_data = {k: v for k, v in pit_losses.items() if k != "__global_fallback__"}
sorted_pit = sorted(pit_data.items(), key=lambda x: x[1], reverse=True)
pit_names  = [c for c, _ in sorted_pit]
pit_vals   = [v for _, v in sorted_pit]

fig3 = go.Figure(go.Bar(
    x=pit_names, y=pit_vals,
    marker_color="#3671C6",
    hovertemplate="%{x}<br>Pit loss: %{y:.2f}s<extra></extra>",
))
dark_layout(fig3, height=300)
fig3.update_layout(
    yaxis_title="Seconds lost",
    xaxis_tickangle=-40,
    margin=dict(l=10, r=10, t=10, b=80),
)
fig3.add_hline(
    y=pit_losses.get("__global_fallback__", 23.1),
    line_dash="dash", line_color="#F39C12",
    annotation_text="Global median",
    annotation_font_color="#F39C12",
)
st.plotly_chart(fig3, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("F1 AI Strategy System · v7 Model · 2023–2025")
