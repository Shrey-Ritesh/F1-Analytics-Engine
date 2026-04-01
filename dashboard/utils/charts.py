from __future__ import annotations
import plotly.graph_objects as go
from .constants import COMPOUND_COLORS


def make_stint_bar(pit_laps: list, compounds: list, total_laps: int,
                   title: str = "Strategy") -> go.Figure:
    """Horizontal stacked bar showing each stint as a colored segment."""
    boundaries = [0] + list(pit_laps) + [total_laps]
    stint_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

    fig = go.Figure()
    seen = set()
    for i, (length, compound) in enumerate(zip(stint_lengths, compounds)):
        color = COMPOUND_COLORS.get(compound, "#888888")
        show_legend = compound not in seen
        seen.add(compound)
        fig.add_trace(go.Bar(
            x=[length],
            y=[title],
            orientation="h",
            name=compound,
            marker_color=color,
            marker_line_color="#1a1a2e",
            marker_line_width=1.5,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>Stint {i + 1}</b><br>"
                f"Compound: {compound}<br>"
                f"Laps: {boundaries[i]}–{boundaries[i + 1]}<br>"
                f"Length: {length} laps<extra></extra>"
            ),
        ))

    # Pit stop markers
    for lap in pit_laps:
        fig.add_vline(x=lap, line_dash="dash", line_color="white", line_width=1.5)

    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=0, r=0, t=4, b=4),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(title="Lap", color="white", gridcolor="#333"),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", y=1.3, font_size=11),
    )
    return fig


def make_driver_radar(drivers_df, selected_abbrevs: list) -> go.Figure:
    """Spider/radar chart comparing selected drivers."""
    axes = ["pace_score", "consistency_score", "podium_rate", "win_rate",
            "driver_performance_score"]
    labels = ["Pace", "Consistency", "Podium %", "Win %", "Overall"]

    colors = ["#FF8000", "#E8002D", "#27F4D2", "#FFF200", "#FF87BC",
              "#64C4FF", "#229971", "#B6BABD"]

    fig = go.Figure()
    for i, abbrev in enumerate(selected_abbrevs):
        row = drivers_df[drivers_df["driver"] == abbrev]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [float(row.get(a, 0)) for a in axes]
        vals_closed = vals + [vals[0]]
        labels_closed = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=colors[i % len(colors)],
            opacity=0.25,
            line=dict(color=colors[i % len(colors)], width=2),
            name=abbrev,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color="#888"),
            bgcolor="#0e1117",
            angularaxis=dict(color="white"),
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def format_race_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{m}m {s:05.2f}s"


def format_pit_laps(pit_laps: list) -> str:
    return ", ".join(f"L{int(l)}" for l in pit_laps)


def dark_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#444"),
        yaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#444"),
    )
    return fig
