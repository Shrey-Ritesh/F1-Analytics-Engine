import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from dashboard.utils.data_loader import load_circuit_dna, load_circuit_archetypes
from dashboard.utils.charts import dark_layout

st.set_page_config(page_title="Circuit DNA · F1", page_icon="🧬", layout="wide")

st.title("🧬 Circuit DNA & Archetypes")
st.caption("Phase 2 · 18-feature fingerprint per circuit, KMeans k=4 clustering")

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading circuit fingerprints (auto-generates on first run)…"):
    dna_df = load_circuit_dna()
    archetypes = load_circuit_archetypes()

FEATURE_NAMES = archetypes.get("feature_names", [])
CLUSTERING_FEATURES = [
    "overtaking_difficulty", "one_stop_pct", "two_stop_pct", "three_stop_pct",
    "dominant_stop_count", "strategy_entropy", "first_pit_mean",
    "avg_stint_length", "top_compound_freq",
]

ARCHETYPE_COLORS = {
    "street_circuit":   "#F39C12",
    "high_degradation": "#E74C3C",
    "high_overtaking":  "#2ECC71",
    "balanced":         "#3498DB",
}

# ── Archetype overview ────────────────────────────────────────────────────────
st.subheader("Archetype Overview")

arch_map = archetypes.get("archetypes", {})
arch_cols = st.columns(4)
archetype_labels = ["street_circuit", "high_degradation", "high_overtaking", "balanced"]
arch_display = {
    "street_circuit":   ("Street Circuit", "1-stop dominant · track position precious"),
    "high_degradation": ("High Degradation", "Tire wear forces multi-stop strategies"),
    "high_overtaking":  ("High Overtaking", "2-stop dominant · aggressive strategy"),
    "balanced":         ("Balanced", "Mixed profile · no single dominant approach"),
}
for idx, label in enumerate(archetype_labels):
    color = ARCHETYPE_COLORS[label]
    circuits_in = arch_map.get(label, [])
    display_name, subtitle = arch_display[label]
    with arch_cols[idx]:
        st.markdown(
            f"<div style='padding:14px;border-left:4px solid {color};"
            f"background:{color}18;border-radius:4px;margin-bottom:8px'>"
            f"<div style='font-weight:bold;color:{color};font-size:1em'>{display_name}</div>"
            f"<div style='color:#aaa;font-size:0.8em;margin:4px 0'>{subtitle}</div>"
            f"<div style='font-size:0.85em'><b>{len(circuits_in)} circuits:</b> "
            + ", ".join(c.replace(" Grand Prix", "").replace(" GP", "") for c in circuits_in)
            + "</div></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Circuit selector ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    circuit_list = sorted(dna_df["circuit"].tolist())
    selected_circuit = st.selectbox("Select Circuit", circuit_list, index=circuit_list.index("Bahrain Grand Prix"))

    row = dna_df[dna_df["circuit"] == selected_circuit].iloc[0]
    archetype = row["archetype_label"]
    arch_color = ARCHETYPE_COLORS.get(archetype, "#888")

    st.markdown(
        f"<div style='margin:12px 0;padding:10px;border:1px solid {arch_color};"
        f"border-radius:8px;background:{arch_color}15;text-align:center'>"
        f"<div style='color:{arch_color};font-size:1.1em;font-weight:bold'>"
        f"{arch_display[archetype][0]}</div>"
        f"<div style='color:#aaa;font-size:0.82em'>{arch_display[archetype][1]}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Key Stats**")
    stat_cols = [
        ("baseline_lap_time", "Baseline Lap (s)"),
        ("pit_loss_time", "Pit Loss (s)"),
        ("overtaking_difficulty", "Overtaking Diff."),
        ("one_stop_pct", "1-Stop %"),
        ("two_stop_pct", "2-Stop %"),
        ("three_stop_pct", "3-Stop %"),
        ("avg_stint_length", "Avg Stint (laps)"),
        ("strategy_entropy", "Strategy Entropy"),
    ]
    for col_name, label in stat_cols:
        val = row.get(col_name, 0)
        if col_name in ("one_stop_pct", "two_stop_pct", "three_stop_pct"):
            st.metric(label, f"{val * 100:.1f}%")
        elif col_name == "baseline_lap_time":
            st.metric(label, f"{val:.3f}s")
        elif col_name == "pit_loss_time":
            st.metric(label, f"{val:.1f}s")
        elif col_name == "avg_stint_length":
            st.metric(label, f"{val:.1f}")
        else:
            st.metric(label, f"{val:.3f}")

with col_right:
    st.subheader(f"Feature Radar — {selected_circuit}")

    # Normalize CLUSTERING_FEATURES to 0-1 across all circuits for fair comparison
    feat_data = dna_df[CLUSTERING_FEATURES].values.astype(float)
    scaler = MinMaxScaler()
    feat_norm = scaler.fit_transform(feat_data)
    norm_df = pd.DataFrame(feat_norm, columns=CLUSTERING_FEATURES)
    norm_df["circuit"] = dna_df["circuit"].values

    sel_idx = dna_df[dna_df["circuit"] == selected_circuit].index[0]
    sel_row_norm = norm_df[norm_df["circuit"] == selected_circuit].iloc[0]

    theta = [f.replace("_", " ").title() for f in CLUSTERING_FEATURES]
    r_vals = [float(sel_row_norm[f]) for f in CLUSTERING_FEATURES]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_vals + [r_vals[0]],
        theta=theta + [theta[0]],
        fill="toself",
        fillcolor=f"{arch_color}30",
        line=dict(color=arch_color, width=2),
        name=selected_circuit.replace(" Grand Prix", ""),
    ))

    # Overlay archetype centroid
    archetype_circuits = arch_map.get(archetype, [])
    arch_idx = norm_df[norm_df["circuit"].isin(archetype_circuits)]
    centroid_r = [float(arch_idx[f].mean()) for f in CLUSTERING_FEATURES]

    fig_radar.add_trace(go.Scatterpolar(
        r=centroid_r + [centroid_r[0]],
        theta=theta + [theta[0]],
        mode="lines",
        line=dict(color="#ffffff55", width=1, dash="dot"),
        name=f"{arch_display[archetype][0]} avg",
    ))

    fig_radar.update_layout(
        **dark_layout(),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color="#555"),
            bgcolor="#111",
        ),
        height=420,
        showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── PCA cluster scatter ───────────────────────────────────────────────────────
st.subheader("Circuit Cluster Map (PCA 2D)")
st.caption("9 clustering features reduced to 2 principal components")

feat_matrix = dna_df[CLUSTERING_FEATURES].values.astype(float)
feat_scaled = MinMaxScaler().fit_transform(feat_matrix)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(feat_scaled)

scatter_df = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "circuit": dna_df["circuit"],
    "archetype": dna_df["archetype_label"],
    "short_name": dna_df["circuit"].str.replace(" Grand Prix", "").str.replace(" GP", ""),
})

fig_pca = px.scatter(
    scatter_df, x="PC1", y="PC2",
    color="archetype",
    color_discrete_map=ARCHETYPE_COLORS,
    text="short_name",
    hover_data={"circuit": True, "archetype": True, "PC1": False, "PC2": False, "short_name": False},
    height=480,
)
fig_pca.update_traces(textposition="top center", marker=dict(size=10))
fig_pca.update_layout(**dark_layout())
st.plotly_chart(fig_pca, use_container_width=True)

st.divider()

# ── Full feature comparison table ─────────────────────────────────────────────
with st.expander("Full Circuit Feature Table (all 18 features)"):
    display_cols = ["circuit", "archetype_label"] + FEATURE_NAMES
    available = [c for c in display_cols if c in dna_df.columns]
    table_df = dna_df[available].sort_values("archetype_label").reset_index(drop=True)
    st.dataframe(table_df, use_container_width=True)
