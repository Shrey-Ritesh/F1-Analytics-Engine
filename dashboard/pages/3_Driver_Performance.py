import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import load_driver_metrics, load_historical_strategies
from dashboard.utils.constants import DRIVER_NAMES
from dashboard.utils.charts import make_driver_radar, dark_layout

st.set_page_config(page_title="Driver Performance · F1", page_icon="🏎", layout="wide")

st.title("👤 Driver Performance")
st.caption("ML-derived scores from 2023–2025 lap time and race data")

df     = load_driver_metrics().copy()
hist   = load_historical_strategies()

df["full_name"]  = df["driver"].map(DRIVER_NAMES).fillna(df["driver"])
df["rank"]       = df["driver_performance_score"].rank(ascending=False).astype(int)
df               = df.sort_values("driver_performance_score", ascending=False).reset_index(drop=True)
df["podium_pct"] = (df["podium_rate"] * 100).round(1)
df["win_pct"]    = (df["win_rate"]    * 100).round(1)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 Rankings", "🕸 Radar Comparison", "📈 Pace vs Consistency", "📅 Driver History"]
)

# ── Tab 1: Rankings table ─────────────────────────────────────────────────
with tab1:
    display = df[[
        "rank", "driver", "full_name", "driver_performance_score",
        "pace_score", "consistency_score", "podium_pct", "win_pct", "total_races",
    ]].rename(columns={
        "rank":                     "Rank",
        "driver":                   "Code",
        "full_name":                "Driver",
        "driver_performance_score": "Overall Score",
        "pace_score":               "Pace",
        "consistency_score":        "Consistency",
        "podium_pct":               "Podium %",
        "win_pct":                  "Win %",
        "total_races":              "Races",
    })
    st.dataframe(
        display.style
            .background_gradient(subset=["Overall Score", "Pace", "Consistency"], cmap="RdYlGn")
            .format({
                "Overall Score": "{:.3f}",
                "Pace":          "{:.3f}",
                "Consistency":   "{:.3f}",
                "Podium %":      "{:.1f}",
                "Win %":         "{:.1f}",
            }),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Radar chart ────────────────────────────────────────────────────
with tab2:
    default_sel = [d for d in ["NOR", "VER", "HAM", "LEC", "RUS"] if d in df["driver"].values][:5]
    selected = st.multiselect(
        "Compare drivers (max 5)",
        options=df["driver"].tolist(),
        default=default_sel,
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
    )
    if len(selected) > 5:
        st.warning("Select at most 5 drivers.")
        selected = selected[:5]
    if selected:
        fig = make_driver_radar(df, selected)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Pace vs Consistency scatter ────────────────────────────────────
with tab3:
    fig3 = px.scatter(
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
            "pace_score": ":.3f",
            "consistency_score": ":.3f",
        },
        labels={
            "pace_score": "Pace Score",
            "consistency_score": "Consistency Score",
            "driver_performance_score": "Overall Score",
        },
    )
    fig3.update_traces(textposition="top center", textfont_size=10)
    # Quadrant lines + labels
    fig3.add_hline(y=df["consistency_score"].median(), line_dash="dot", line_color="#555")
    fig3.add_vline(x=df["pace_score"].median(),       line_dash="dot", line_color="#555")
    for x_a, y_a, text in [
        (0.95, 0.95, "Fast + Consistent"),
        (0.05, 0.95, "Slow + Consistent"),
        (0.95, 0.05, "Fast + Erratic"),
        (0.05, 0.05, "Slow + Erratic"),
    ]:
        fig3.add_annotation(x=x_a, y=y_a, text=text, showarrow=False,
                            font=dict(color="#555", size=11), xref="paper", yref="paper")
    dark_layout(fig3, "Pace vs Consistency (bubble = races driven)", height=520)
    fig3.update_coloraxes(colorbar=dict(
        tickfont=dict(color="white"),
        title=dict(text="Score", font=dict(color="white")),
    ))
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Driver history ─────────────────────────────────────────────────
with tab4:
    driver_sel = st.selectbox(
        "Select driver",
        df["driver"].tolist(),
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
    )
    drv_df = hist[(hist["driver"] == driver_sel) & (hist["data_quality"] == "clean")]

    if drv_df.empty:
        st.info("No historical strategy data for this driver.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            year_stop = drv_df.groupby(["race_year", "n_stops"]).size().reset_index(name="count")
            fig_h1 = go.Figure()
            for yr in sorted(year_stop["race_year"].unique()):
                yd = year_stop[year_stop["race_year"] == yr]
                fig_h1.add_trace(go.Bar(
                    x=yd["n_stops"].astype(str) + "-stop",
                    y=yd["count"],
                    name=str(yr),
                ))
            dark_layout(fig_h1, f"{driver_sel} — Stop Count by Season", height=320)
            fig_h1.update_layout(barmode="group", yaxis_title="Races", xaxis_title="Strategy")
            st.plotly_chart(fig_h1, use_container_width=True)

        with col_b:
            stop_pos = drv_df.groupby("n_stops")["final_position"].mean().reset_index()
            fig_h2 = go.Figure(go.Bar(
                x=stop_pos["n_stops"].astype(str) + "-stop",
                y=stop_pos["final_position"],
                marker_color=["#2ECC71", "#F39C12", "#E74C3C"][:len(stop_pos)],
                hovertemplate="%{x}<br>Avg finish: P%{y:.1f}<extra></extra>",
            ))
            dark_layout(fig_h2, f"{driver_sel} — Avg Finish Position by Strategy", height=320)
            fig_h2.update_layout(yaxis_title="Avg Finish Position", yaxis_autorange="reversed")
            st.plotly_chart(fig_h2, use_container_width=True)

        # Timeline
        st.subheader(f"{driver_sel} — Results by Circuit & Strategy")
        circ_order = sorted(drv_df["circuit"].unique())
        drv_df_sorted = drv_df.copy()
        drv_df_sorted["n_stops_label"] = drv_df_sorted["n_stops"].astype(str) + "-stop"
        color_map_stops = {"1-stop": "#2ECC71", "2-stop": "#F39C12", "3-stop": "#E74C3C"}
        fig_tl = px.scatter(
            drv_df_sorted.dropna(subset=["final_position"]),
            x="circuit", y="final_position",
            color="n_stops_label",
            symbol="race_year",
            color_discrete_map=color_map_stops,
            hover_data={"circuit": True, "final_position": True, "race_year": True, "n_stops_label": True},
            labels={"circuit": "Circuit", "final_position": "Finish Position", "n_stops_label": "Strategy"},
            category_orders={"circuit": circ_order},
        )
        dark_layout(fig_tl, height=380)
        fig_tl.update_layout(
            yaxis_autorange="reversed",
            yaxis_title="Finish Position",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_tl, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("F1 AI Strategy System · v7 Model · 2023–2025")
