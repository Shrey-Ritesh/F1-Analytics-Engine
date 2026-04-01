import sys, os, contextlib, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from dashboard.utils.data_loader import (
    load_optimizer, load_circuit_profiles, load_model_metadata,
    build_driver_id_map, build_team_encoding_map,
)
from dashboard.utils.constants import CIRCUIT_LAPS, DRIVER_NAMES, STOP_COLORS
from dashboard.utils.charts import make_stint_bar, format_race_time, format_pit_laps, dark_layout

st.set_page_config(page_title="Strategy Optimizer · F1", page_icon="🏎", layout="wide")

st.title("🔧 Pit Stop Strategy Optimizer")
st.caption("Physics simulation (70%) + historical prior (30%)")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Race Scenario")
    circuit = st.selectbox("Circuit", sorted(CIRCUIT_LAPS.keys()), index=4)  # Bahrain default
    default_laps = CIRCUIT_LAPS[circuit]

    driver_abbrev = st.selectbox(
        "Driver",
        sorted(DRIVER_NAMES.keys()),
        format_func=lambda x: f"{x} — {DRIVER_NAMES[x]}",
        index=list(sorted(DRIVER_NAMES.keys())).index("VER"),
    )

    team_map  = build_team_encoding_map()
    team_name = st.selectbox("Team", sorted(team_map.keys()), index=list(sorted(team_map.keys())).index("Red Bull Racing"))

    grid_pos      = st.slider("Grid Position", 1, 20, 1)
    start_compound = st.radio("Start Compound", ["SOFT", "MEDIUM", "HARD"], horizontal=True)
    total_laps    = st.number_input("Total Laps", min_value=44, max_value=78, value=default_laps)
    track_temp    = st.slider("Track Temperature (°C)", 20, 55, 35)
    n_stops_opts  = st.multiselect("Stop counts to evaluate", [1, 2, 3], default=[1, 2, 3])
    run_btn       = st.button("▶  Run Optimizer", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("F1 AI Strategy System · v7 Model · 2023–2025")

if not run_btn:
    st.info("Configure a race scenario in the sidebar and click **Run Optimizer**.")
    st.stop()

if not n_stops_opts:
    st.error("Select at least one stop count.")
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run(circuit, driver_id, team_encoded, grid_pos, total_laps,
         start_compound, n_stops_tuple, track_temp):
    optimize_fn = load_optimizer()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return optimize_fn(
            driver_id=driver_id,
            team_encoded=team_encoded,
            circuit=circuit,
            grid_position=grid_pos,
            total_laps=int(total_laps),
            start_compound=start_compound,
            n_stops_range=n_stops_tuple,
            track_temperature=float(track_temp),
        )

driver_id_map = build_driver_id_map()
driver_id     = driver_id_map.get(driver_abbrev, 1)
team_encoded  = team_map[team_name]

with st.spinner("Simulating strategies…"):
    result = _run(
        circuit, driver_id, team_encoded, grid_pos, int(total_laps),
        start_compound, tuple(sorted(n_stops_opts)), track_temp,
    )

physics  = result["physics_ranked"]
combined = result["combined_ranked"]
profiles = load_circuit_profiles()
metadata = load_model_metadata()
low_conf = metadata.get("low_confidence_circuits", [])

if circuit in low_conf:
    st.warning(f"⚠ **Low confidence circuit**: absolute times may vary ±4 s. Strategy *deltas* remain reliable.")

# ── Summary metrics ────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategies evaluated", f"{result['total_evaluated']:,}")
m2.metric("Pruned (unrealistic)",  f"{result['total_pruned']:,}")
m3.metric("Best race time",        format_race_time(physics[0]["total_race_time"]))
m4.metric("Best combined score",   f"{combined[0]['combined_score']:.4f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Top Strategies", "📊 Strategy Space", "🏁 Best Per Stop Count", "📖 Circuit Profile"]
)

# ── Tab 1: Top strategies table ───────────────────────────────────────────
with tab1:
    import pandas as pd

    rows = []
    for s in combined[:15]:
        rows.append({
            "Rank":       s["combined_rank"],
            "Stops":      s["n_stops"],
            "Pit Laps":   format_pit_laps(s["pit_laps"]),
            "Compounds":  " → ".join(s["compounds"]),
            "Race Time":  format_race_time(s["total_race_time"]),
            "Score":      round(s["combined_score"], 4),
            "Prior":      round(s["prior_score"],    3),
            "Time Score": round(s["time_score"],     3),
        })
    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table.style.background_gradient(subset=["Score"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

# ── Tab 2: Strategy space scatter ─────────────────────────────────────────
with tab2:
    import pandas as pd

    all_strats = combined
    df_scatter = pd.DataFrame([{
        "race_time":     s["total_race_time"],
        "combined_score":s["combined_score"],
        "n_stops":       str(s["n_stops"]) + "-stop",
        "pit_laps":      format_pit_laps(s["pit_laps"]),
        "compounds":     " → ".join(s["compounds"]),
        "prior":         round(s["prior_score"], 3),
    } for s in all_strats])

    color_map = {"1-stop": "#2ECC71", "2-stop": "#F39C12", "3-stop": "#E74C3C"}
    fig = px.scatter(
        df_scatter,
        x="race_time", y="combined_score",
        color="n_stops", color_discrete_map=color_map,
        hover_data={"pit_laps": True, "compounds": True, "prior": True,
                    "race_time": ":.1f", "combined_score": ":.4f"},
        labels={"race_time": "Total Race Time (s)", "combined_score": "Combined Score", "n_stops": "Strategy"},
        opacity=0.6,
    )
    # Mark best physics and best combined
    fig.add_scatter(
        x=[physics[0]["total_race_time"]], y=[physics[0]["combined_score"]],
        mode="markers+text", marker=dict(size=14, color="white", symbol="star"),
        text=["Best Physics"], textposition="top right",
        name="Best Physics", showlegend=True,
    )
    fig.add_scatter(
        x=[combined[0]["total_race_time"]], y=[combined[0]["combined_score"]],
        mode="markers+text", marker=dict(size=14, color="#FFD700", symbol="star"),
        text=["Best Combined"], textposition="top right",
        name="Best Combined", showlegend=True,
    )
    dark_layout(fig, "Physics Time vs Combined Score — Full Strategy Space", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Best per stop count ────────────────────────────────────────────
with tab3:
    stop_counts = sorted(set(s["n_stops"] for s in combined))
    for n in stop_counts:
        best = next(s for s in combined if s["n_stops"] == n)
        st.subheader(f"{n}-stop best strategy")

        mcols = st.columns(4)
        mcols[0].metric("Race Time",      format_race_time(best["total_race_time"]))
        mcols[1].metric("Combined Score", f"{best['combined_score']:.4f}")
        mcols[2].metric("Pit Laps",       format_pit_laps(best["pit_laps"]))
        mcols[3].metric("Prior Score",    f"{best['prior_score']:.3f}")

        fig_stint = make_stint_bar(best["pit_laps"], best["compounds"], int(total_laps))
        st.plotly_chart(fig_stint, use_container_width=True)
        st.divider()

# ── Tab 4: Circuit profile ────────────────────────────────────────────────
with tab4:
    profile = profiles.get(circuit, {})
    if not profile:
        st.info("No historical profile available for this circuit.")
    else:
        c_left, c_right = st.columns(2)

        with c_left:
            dist = profile.get("stop_distribution", {})
            if dist:
                labels_pie = [f"{k}-stop" for k in dist]
                vals_pie   = [float(v) for v in dist.values()]
                colors_pie = [STOP_COLORS.get(int(k), "#888") for k in dist]
                fig_pie = go.Figure(go.Pie(
                    labels=labels_pie, values=vals_pie,
                    marker_colors=colors_pie,
                    hole=0.4,
                    hovertemplate="%{label}: %{percent}<extra></extra>",
                ))
                dark_layout(fig_pie, "Historical Stop Distribution", height=320)
                st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            ot = profile.get("overtaking_difficulty")
            if ot is not None:
                label = "track position critical" if ot > 0.85 else "track position matters" if ot > 0.65 else "overtaking feasible"
                st.metric("Overtaking Difficulty", f"{ot:.2f}", delta=label, delta_color="off")

            sample = profile.get("sample_size")
            if sample:
                st.metric("Historical Sample", f"{sample} races")

            winning = profile.get("winning_strategy", {})
            if winning:
                w_compounds = " → ".join(winning.get("compounds", []))
                st.info(
                    f"**Reference winner**: {winning.get('n_stops')}-stop  |  "
                    f"Laps {winning.get('pit_laps')}  |  {w_compounds}"
                )

        # Pit windows vs optimizer recommendation
        windows = profile.get("pit_windows", {})
        dominant = max(dist, key=lambda k: float(dist[k])) if dist else None
        rec_best = next((s for s in combined if str(s["n_stops"]) == str(dominant)), None)

        if dominant and str(dominant) in windows:
            w_data = windows[str(dominant)]
            st.subheader(f"Pit Windows — {dominant}-stop (historical mean ± 1σ)")

            stop_labels, means, stds = [], [], []
            for stop_key, (mean, std) in w_data.items():
                stop_labels.append(f"Stop {int(stop_key) + 1}")
                means.append(mean)
                stds.append(std)

            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=stop_labels, y=means,
                error_y=dict(type="data", array=stds, visible=True, color="#aaa"),
                mode="markers", marker=dict(size=12, color="#3671C6"),
                name="Historical window",
            ))
            if rec_best:
                rec_labels = [f"Stop {i+1}" for i in range(len(rec_best["pit_laps"]))]
                fig_w.add_trace(go.Scatter(
                    x=rec_labels, y=rec_best["pit_laps"],
                    mode="markers", marker=dict(size=12, color="#FFD700", symbol="star"),
                    name="Optimizer recommendation",
                ))
            dark_layout(fig_w, height=320)
            fig_w.update_layout(yaxis_title="Lap Number")
            st.plotly_chart(fig_w, use_container_width=True)
