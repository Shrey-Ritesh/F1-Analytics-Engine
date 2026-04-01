import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.utils.data_loader import (
    load_circuit_profiles, load_circuit_baselines,
    load_pit_losses, load_historical_strategies,
)
from dashboard.utils.constants import STOP_COLORS
from dashboard.utils.charts import make_stint_bar, dark_layout

st.set_page_config(page_title="Circuit Intelligence · F1", page_icon="🏎", layout="wide")

st.title("🗺 Circuit Intelligence")
st.caption("Per-circuit strategy analytics derived from 2023–2025 F1 data")

profiles   = load_circuit_profiles()
baselines  = load_circuit_baselines()
pit_losses = load_pit_losses()
hist_df    = load_historical_strategies()

circuit = st.selectbox("Select Circuit", sorted(profiles.keys()))
profile = profiles.get(circuit, {})

st.divider()

# ── Header metrics ─────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Baseline Lap",   f"{baselines.get(circuit, 0):.3f} s")
m2.metric("Pit Loss",       f"{pit_losses.get(circuit, pit_losses.get('__global_fallback__', 23.1)):.2f} s")
m3.metric("Sample Size",    f"{profile.get('sample_size', '—')} races")
ot = profile.get("overtaking_difficulty")
ot_label = "Critical" if ot and ot > 0.85 else "Moderate" if ot and ot > 0.65 else "Low"
m4.metric("Overtaking",     f"{ot:.2f}" if ot else "—", delta=ot_label, delta_color="off")
dist = profile.get("stop_distribution", {})
if dist:
    dominant_n = max(dist, key=lambda k: float(dist[k]))
    m5.metric("Dominant Strategy", f"{dominant_n}-stop ({float(dist[dominant_n])*100:.0f}%)")

st.divider()

row1_left, row1_right = st.columns(2)

# ── Stop distribution bar ─────────────────────────────────────────────────
with row1_left:
    if dist:
        x_vals = [f"{k}-stop" for k in sorted(dist.keys())]
        y_vals = [float(dist[k]) * 100 for k in sorted(dist.keys())]
        bar_colors = [STOP_COLORS.get(int(k), "#888") for k in sorted(dist.keys())]
        fig = go.Figure(go.Bar(
            x=x_vals, y=y_vals,
            marker_color=bar_colors,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        dark_layout(fig, "Stop Strategy Distribution (2023–2025)", height=320)
        fig.update_layout(yaxis_title="% of historical races",
                          xaxis_title="Strategy")
        st.plotly_chart(fig, use_container_width=True)

# ── Top compound sequences ────────────────────────────────────────────────
with row1_right:
    top_c = profile.get("top_compounds", [])
    if top_c:
        seqs  = [str(t[0]) for t in top_c[:6]]
        freqs = [float(t[1]) * 100 for t in top_c[:6]]
        fig2 = go.Figure(go.Bar(
            x=seqs, y=freqs,
            marker_color="#E8002D",
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        ))
        dark_layout(fig2, "Top Compound Sequences", height=320)
        fig2.update_layout(
            yaxis_title="Frequency (%)",
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Pit window analysis ───────────────────────────────────────────────────
windows = profile.get("pit_windows", {})
if windows:
    st.subheader("Pit Window Analysis — Mean ± 1σ by Stop Count")
    fig3 = go.Figure()
    colors_w = ["#2ECC71", "#F39C12", "#E74C3C"]
    for idx, (n_stops_key, stop_data) in enumerate(sorted(windows.items())):
        stop_labels = [f"Stop {int(k) + 1}" for k in sorted(stop_data.keys())]
        means = [stop_data[k][0] for k in sorted(stop_data.keys())]
        stds  = [stop_data[k][1] for k in sorted(stop_data.keys())]
        fig3.add_trace(go.Scatter(
            x=stop_labels, y=means,
            error_y=dict(type="data", array=stds, visible=True),
            mode="markers+lines",
            marker=dict(size=10, color=colors_w[idx % 3]),
            line=dict(color=colors_w[idx % 3], dash="dot"),
            name=f"{n_stops_key}-stop",
        ))
    dark_layout(fig3, height=360)
    fig3.update_layout(yaxis_title="Lap Number", xaxis_title="Stop")
    st.plotly_chart(fig3, use_container_width=True)

# ── Year-by-year trend ────────────────────────────────────────────────────
circ_df = hist_df[(hist_df["circuit"] == circuit) & (hist_df["data_quality"] == "clean")]
if not circ_df.empty:
    st.subheader("Strategy Trend by Season")
    year_stop = circ_df.groupby(["race_year", "n_stops"]).size().reset_index(name="count")
    fig4 = go.Figure()
    for year in sorted(year_stop["race_year"].unique()):
        yd = year_stop[year_stop["race_year"] == year]
        fig4.add_trace(go.Bar(
            x=yd["n_stops"].astype(str) + "-stop",
            y=yd["count"],
            name=str(year),
        ))
    dark_layout(fig4, height=300)
    fig4.update_layout(barmode="group", yaxis_title="Driver count",
                       xaxis_title="Stop strategy")
    st.plotly_chart(fig4, use_container_width=True)

# ── Winning strategy reference ────────────────────────────────────────────
winning = profile.get("winning_strategy", {})
if winning and winning.get("compounds"):
    st.subheader("Reference Winning Strategy")
    pit_laps_w = winning.get("pit_laps", [])
    # pit_laps may be stored as floats
    pit_laps_w = [int(p) for p in pit_laps_w] if pit_laps_w else []
    compounds_w = winning.get("compounds", [])
    n_laps_w = baselines.get(circuit, 57)
    if pit_laps_w and compounds_w and len(compounds_w) == len(pit_laps_w) + 1:
        fig_w = make_stint_bar(pit_laps_w, compounds_w, int(n_laps_w), title="Winning Strategy")
        st.plotly_chart(fig_w, use_container_width=True)
    col_w = st.columns(3)
    col_w[0].metric("Stops",    winning.get("n_stops", "—"))
    col_w[1].metric("Pit Laps", str([int(p) for p in pit_laps_w]))
    col_w[2].metric("Compounds", " → ".join(compounds_w))

st.sidebar.markdown("---")
st.sidebar.caption("F1 AI Strategy System · v7 Model · 2023–2025")
