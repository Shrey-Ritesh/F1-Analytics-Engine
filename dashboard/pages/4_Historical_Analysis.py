import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import load_historical_strategies, load_circuit_profiles
from dashboard.utils.constants import STOP_COLORS
from dashboard.utils.charts import dark_layout

st.set_page_config(page_title="Historical Analysis · F1", page_icon="🏎", layout="wide")

st.title("📊 Historical Strategy Analysis")
st.caption("Cross-circuit, cross-season pattern mining from 2023–2025 F1 data")

hist     = load_historical_strategies()
profiles = load_circuit_profiles()

# ── Sidebar filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    year_filter = st.multiselect("Season", [2023, 2024, 2025], default=[2023, 2024, 2025])
    quality_opt = st.radio("Data Quality", ["Clean only", "All"], index=0)
    circuit_opt = st.selectbox("Circuit (optional)", ["All"] + sorted(hist["circuit"].unique()))
    st.markdown("---")
    st.caption("F1 AI Strategy System · v7 Model · 2023–2025")

df = hist[hist["race_year"].isin(year_filter)].copy()
if quality_opt == "Clean only":
    df = df[df["data_quality"] == "clean"]
if circuit_opt != "All":
    df = df[df["circuit"] == circuit_opt]

total  = len(df)
clean  = (df["data_quality"] == "clean").sum() if "data_quality" in df.columns else total
pct_clean = (clean / total * 100) if total > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Entries in selection", f"{total:,}")
m2.metric("Clean data",           f"{clean:,} ({pct_clean:.0f}%)")
if not df.empty:
    most_common = df.groupby("n_stops").size().idxmax()
    m3.metric("Most common strategy", f"{most_common}-stop")

st.divider()

tab1, tab2, tab3 = st.tabs(["🗺 Stop Patterns", "🔄 Compound Analysis", "🏆 Position vs Strategy"])

# ── Tab 1: Stop patterns ──────────────────────────────────────────────────
with tab1:
    if df.empty:
        st.info("No data for current filters.")
    else:
        # Heatmap: circuits × stop counts
        st.subheader("Stop Count Distribution by Circuit")
        pivot = (
            df.groupby(["circuit", "n_stops"])
            .size()
            .reset_index(name="count")
        )
        all_circuits = sorted(pivot["circuit"].unique())
        all_stops    = sorted(pivot["n_stops"].unique())

        matrix = pd.DataFrame(0, index=all_circuits, columns=all_stops)
        for _, row in pivot.iterrows():
            if row["n_stops"] in matrix.columns:
                total_c = pivot[pivot["circuit"] == row["circuit"]]["count"].sum()
                matrix.loc[row["circuit"], row["n_stops"]] = (
                    row["count"] / total_c * 100 if total_c > 0 else 0
                )

        fig_hm = go.Figure(go.Heatmap(
            z=matrix.values,
            x=[f"{c}-stop" for c in matrix.columns],
            y=matrix.index.tolist(),
            colorscale="Blues",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            text=[[f"{v:.0f}%" for v in row] for row in matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        dark_layout(fig_hm, height=max(400, len(all_circuits) * 22))
        fig_hm.update_layout(
            xaxis_title="Strategy",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Stacked bar: all circuits
        if circuit_opt == "All":
            st.subheader("Stop Distribution per Circuit (stacked)")
            fig_sb = go.Figure()
            for n in sorted(all_stops):
                vals = []
                for circ in all_circuits:
                    total_c = pivot[pivot["circuit"] == circ]["count"].sum()
                    row_val = pivot[(pivot["circuit"] == circ) & (pivot["n_stops"] == n)]
                    vals.append(float(row_val["count"].values[0]) / total_c * 100
                                if not row_val.empty and total_c > 0 else 0)
                fig_sb.add_trace(go.Bar(
                    name=f"{n}-stop",
                    x=all_circuits, y=vals,
                    marker_color=STOP_COLORS.get(n, "#888"),
                ))
            dark_layout(fig_sb, height=400)
            fig_sb.update_layout(
                barmode="stack",
                xaxis_tickangle=-40,
                yaxis_title="% of races",
            )
            st.plotly_chart(fig_sb, use_container_width=True)

# ── Tab 2: Compound analysis ──────────────────────────────────────────────
with tab2:
    if df.empty:
        st.info("No data for current filters.")
    else:
        st.subheader("Top Compound Sequences")
        df["compound_seq"] = df["compounds"].apply(
            lambda c: "->".join(c) if isinstance(c, list) and c else "Unknown"
        )
        seq_counts = (
            df[df["compound_seq"] != "Unknown"]
            .groupby("compound_seq").size()
            .sort_values(ascending=False)
            .head(15)
        )
        fig_seq = go.Figure(go.Bar(
            x=seq_counts.index.tolist(),
            y=seq_counts.values,
            marker_color="#E8002D",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        ))
        dark_layout(fig_seq, height=360)
        fig_seq.update_layout(
            yaxis_title="Count",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_seq, use_container_width=True)

        # Start compound by year
        st.subheader("Start Compound Choice by Season")
        df["start_compound"] = df["compounds"].apply(
            lambda c: c[0] if isinstance(c, list) and c else None
        )
        sc_df = df.dropna(subset=["start_compound"])
        sc_counts = sc_df.groupby(["race_year", "start_compound"]).size().reset_index(name="count")
        from dashboard.utils.constants import COMPOUND_COLORS
        fig_sc = go.Figure()
        for cpd in ["SOFT", "MEDIUM", "HARD"]:
            sub = sc_counts[sc_counts["start_compound"] == cpd]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Bar(
                name=cpd,
                x=sub["race_year"].astype(str),
                y=sub["count"],
                marker_color=COMPOUND_COLORS.get(cpd, "#888"),
            ))
        dark_layout(fig_sc, height=320)
        fig_sc.update_layout(barmode="group", xaxis_title="Season", yaxis_title="Count")
        st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 3: Position vs strategy ───────────────────────────────────────────
with tab3:
    pos_df = df.dropna(subset=["final_position"]).copy()
    pos_df = pos_df[pos_df["n_stops"].isin([1, 2, 3])]

    if pos_df.empty:
        st.info("No data for current filters.")
    else:
        col_box, col_scatter = st.columns(2)

        with col_box:
            fig_box = go.Figure()
            for n in sorted(pos_df["n_stops"].unique()):
                sub = pos_df[pos_df["n_stops"] == n]["final_position"]
                fig_box.add_trace(go.Box(
                    y=sub,
                    name=f"{n}-stop",
                    marker_color=STOP_COLORS.get(n, "#888"),
                    boxmean=True,
                ))
            dark_layout(fig_box, "Finish Position Distribution by Strategy", height=420)
            fig_box.update_layout(yaxis_autorange="reversed", yaxis_title="Finish Position")
            st.plotly_chart(fig_box, use_container_width=True)

        with col_scatter:
            jitter_df = pos_df.copy()
            jitter_df["n_stops_jit"] = jitter_df["n_stops"] + np.random.uniform(-0.2, 0.2, len(jitter_df))
            fig_sc2 = px.scatter(
                jitter_df,
                x="n_stops_jit",
                y="final_position",
                color=jitter_df["n_stops"].astype(str).map(lambda x: f"{x}-stop"),
                color_discrete_map={"1-stop": "#2ECC71", "2-stop": "#F39C12", "3-stop": "#E74C3C"},
                opacity=0.5,
                hover_data={"circuit": True, "driver": True, "race_year": True,
                            "n_stops_jit": False},
                labels={"n_stops_jit": "Stop Strategy", "final_position": "Finish Position",
                        "color": "Strategy"},
            )
            # Trend line per stop count (simple mean)
            for n in sorted(pos_df["n_stops"].unique()):
                mean_p = pos_df[pos_df["n_stops"] == n]["final_position"].mean()
                fig_sc2.add_hline(
                    y=mean_p,
                    line_dash="dash",
                    line_color=STOP_COLORS.get(n, "#888"),
                    annotation_text=f"{n}-stop mean: P{mean_p:.1f}",
                    annotation_font_color=STOP_COLORS.get(n, "#888"),
                )
            dark_layout(fig_sc2, "Final Position vs Strategy (jittered)", height=420)
            fig_sc2.update_layout(
                yaxis_autorange="reversed",
                yaxis_title="Finish Position",
                xaxis_title="Stop Strategy",
                xaxis=dict(tickvals=[1, 2, 3], ticktext=["1-stop", "2-stop", "3-stop"]),
            )
            st.plotly_chart(fig_sc2, use_container_width=True)
