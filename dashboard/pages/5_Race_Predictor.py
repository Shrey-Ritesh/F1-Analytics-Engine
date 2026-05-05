import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import (
    load_circuit_baselines,
    load_driver_prediction_stats,
    load_race_outcome_model,
    load_qualifying_model,
    load_driver_form_module,
)
from dashboard.utils.constants import CIRCUIT_LAPS, DRIVER_NAMES
from dashboard.utils.charts import dark_layout

st.set_page_config(page_title="Race Predictor · F1", page_icon="🏁", layout="wide")

st.title("🏁 Race Outcome Predictor")
st.caption("Phase 2 · XGBoost regression on grid position, driver metrics, and circuit baseline pace")

# ── Load resources ──────────────────────────────────────────────────────────
baselines = load_circuit_baselines()
driver_stats_map = load_driver_prediction_stats()

with st.spinner("Loading race outcome model (auto-trains on first run)…"):
    predict_position = load_race_outcome_model()

with st.spinner("Loading qualifying model…"):
    predict_grid = load_qualifying_model()

get_driver_form, rank_drivers_by_form = load_driver_form_module()

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Race Scenario")

    driver_abbrev = st.selectbox(
        "Driver",
        sorted(DRIVER_NAMES.keys()),
        format_func=lambda x: f"{x} — {DRIVER_NAMES[x]}",
        index=list(sorted(DRIVER_NAMES.keys())).index("VER"),
    )
    driver_full = DRIVER_NAMES[driver_abbrev]
    # Dataset uses abbreviations as driver identifiers (e.g. "VER", not "Max Verstappen")

    circuit = st.selectbox("Circuit", sorted(CIRCUIT_LAPS.keys()), index=4)

    st.divider()
    st.subheader("Grid Position")
    use_quali_pred = st.checkbox("Predict qualifying position automatically", value=False)

    if use_quali_pred:
        driver_metrics = driver_stats_map.get(driver_abbrev, {})
        quali_result = predict_grid(circuit, driver_metrics, baselines)
        auto_grid = quali_result["grid_rounded"]
        st.info(f"Predicted qualifying: **P{auto_grid}**")
        grid_position = auto_grid
    else:
        grid_position = st.slider("Grid Position", 1, 20, 5)

    track_temp = st.slider("Track Temperature (°C)", 15, 60, 35)

    st.divider()
    run_btn = st.button("Predict Race Outcome", type="primary", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────────────
# Stats keyed by driver abbreviation (matches dataset column values)
driver_metrics = driver_stats_map.get(driver_abbrev, {})
result = predict_position(grid_position, track_temp, circuit, driver_metrics, baselines)

predicted_pos = result["position_rounded"]
category = result["category"]

CATEGORY_COLOR = {
    "Win":           "#FFD700",
    "Podium":        "#C0C0C0",
    "Points":        "#2ECC71",
    "Out of Points": "#E74C3C",
}
cat_color = CATEGORY_COLOR.get(category, "#888")

# ── Top KPI row ──────────────────────────────────────────────────────────────
col_pos, col_cat, col_grid, col_temp = st.columns(4)
with col_pos:
    st.metric("Predicted Finish", f"P{predicted_pos}")
with col_cat:
    st.markdown(
        f"<div style='padding:12px;background:{cat_color}22;border:1px solid {cat_color};"
        f"border-radius:8px;text-align:center;font-weight:bold;color:{cat_color};'>"
        f"{category}</div>",
        unsafe_allow_html=True,
    )
with col_grid:
    st.metric("Starting Grid", f"P{grid_position}")
with col_temp:
    st.metric("Track Temp", f"{track_temp}°C")

st.divider()

# ── Driver form & scenario analysis ─────────────────────────────────────────
col_form, col_grid_sweep = st.columns([1, 1])

with col_form:
    st.subheader(f"Recent Form — {driver_full}")
    form = get_driver_form(driver_abbrev)  # dataset uses abbreviations

    if "error" in form:
        st.warning(f"No form data: {form['error']}")
    else:
        f1, f2, f3 = st.columns(3)
        f1.metric("Mean Pos (last 5)", f"P{form['mean_position']:.1f}")
        f2.metric("Trend", form["form_label"])
        f3.metric("Form Score", f"{form['form_score']:.2f}")

        positions = form["recent_positions"]
        circuits_short = [c.replace(" Grand Prix", "").replace(" GP", "") for c in form["recent_circuits"]]

        fig_form = go.Figure()
        fig_form.add_trace(go.Scatter(
            x=circuits_short,
            y=positions,
            mode="lines+markers",
            marker=dict(size=10, color=cat_color),
            line=dict(color=cat_color, width=2),
            name="Position",
        ))
        fig_form.update_yaxes(autorange="reversed", title="Position")
        dark_layout(fig_form, title=f"Last {len(positions)} races", height=260)
        fig_form.update_layout(xaxis_title="")
        st.plotly_chart(fig_form, use_container_width=True)

with col_grid_sweep:
    st.subheader("Position vs Grid Start")
    st.caption("Predicted finish across all 20 grid slots at this circuit")

    grid_positions = list(range(1, 21))
    preds = [
        predict_position(g, track_temp, circuit, driver_metrics, baselines)["position_rounded"]
        for g in grid_positions
    ]

    sweep_colors = []
    for p in preds:
        if p == 1:
            sweep_colors.append("#FFD700")
        elif p <= 3:
            sweep_colors.append("#C0C0C0")
        elif p <= 10:
            sweep_colors.append("#2ECC71")
        else:
            sweep_colors.append("#E74C3C")

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Bar(
        x=grid_positions,
        y=preds,
        marker_color=sweep_colors,
        name="Predicted finish",
    ))
    fig_sweep.add_trace(go.Scatter(
        x=[grid_position],
        y=[predicted_pos],
        mode="markers",
        marker=dict(size=14, color="white", symbol="star"),
        name="Selected grid",
    ))
    dark_layout(fig_sweep, height=260)
    fig_sweep.update_layout(
        xaxis_title="Grid Position",
        yaxis_title="Predicted Finish",
        yaxis=dict(autorange="reversed"),
        showlegend=True,
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

st.divider()

# ── Driver comparison at this circuit ────────────────────────────────────────
st.subheader(f"All Drivers — Predicted Finish at {circuit}")
st.caption("Assuming each driver starts from their predicted qualifying position")

rows = []
for abbr, full_name in DRIVER_NAMES.items():
    dm = driver_stats_map.get(abbr, {})  # keyed by abbreviation
    quali = predict_grid(circuit, dm, baselines)
    outcome = predict_position(quali["grid_rounded"], track_temp, circuit, dm, baselines)
    rows.append({
        "Driver": f"{abbr} — {full_name}",
        "Predicted Grid": f"P{quali['grid_rounded']}",
        "Predicted Finish": outcome["position_rounded"],
        "Category": outcome["category"],
    })

comp_df = pd.DataFrame(rows).sort_values("Predicted Finish")

fig_comp = go.Figure(go.Bar(
    x=comp_df["Driver"],
    y=comp_df["Predicted Finish"],
    marker_color=[CATEGORY_COLOR.get(c, "#888") for c in comp_df["Category"]],
    text=comp_df["Predicted Finish"].apply(lambda p: f"P{p}"),
    textposition="outside",
))
fig_comp.update_yaxes(autorange="reversed")
dark_layout(fig_comp, height=380)
fig_comp.update_layout(xaxis_tickangle=-45, yaxis_title="Predicted Finish", xaxis_title="")
st.plotly_chart(fig_comp, use_container_width=True)

# ── Driver form leaderboard ─────────────────────────────────────────────────
with st.expander("Driver Form Leaderboard (last 5 races)"):
    form_df = rank_drivers_by_form(n_races=5)
    if not form_df.empty:
        form_df.index = range(1, len(form_df) + 1)
        st.dataframe(
            form_df.rename(columns={
                "driver": "Driver",
                "mean_position": "Avg Position",
                "form_label": "Trend",
                "form_score": "Form Score",
                "win_count": "Wins",
                "podium_count": "Podiums",
                "points_count": "Points Finishes",
                "n_races_used": "Races",
            }),
            use_container_width=True,
        )
