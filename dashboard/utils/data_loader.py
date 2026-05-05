import os
import sys
import ast
import json
import contextlib
import io

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Ensure project root is on the path so model.* imports resolve
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


@st.cache_data
def load_driver_metrics() -> pd.DataFrame:
    df = pd.read_csv(_path("data", "processed", "driver_performance_metrics_v2.csv"))
    for col in ["podium_rate", "win_rate"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    return df


@st.cache_data
def load_historical_strategies() -> pd.DataFrame:
    df = pd.read_csv(_path("data", "processed", "historical_strategies.csv"))

    def _parse(val):
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

    df["pit_laps"]  = df["pit_laps"].apply(_parse)
    df["compounds"] = df["compounds"].apply(_parse)
    df["race_year"] = df["race_year"].astype(int)
    return df


@st.cache_data
def load_circuit_profiles() -> dict:
    with open(_path("model", "strategy_optimizer", "circuit_strategy_profiles.json")) as f:
        return json.load(f)


@st.cache_data
def load_circuit_baselines() -> dict:
    with open(_path("model", "lap_time_model", "circuit_baselines.json")) as f:
        return json.load(f)


@st.cache_data
def load_pit_losses() -> dict:
    with open(_path("model", "lap_time_model", "pit_loss_estimates.json")) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    df = pd.read_csv(_path("model", "lap_time_model", "feature_importance_v7.csv"))
    df.columns = [c.strip() for c in df.columns]
    gain_col = [c for c in df.columns if "gain" in c.lower() or "importance" in c.lower()]
    if gain_col:
        df["pct"] = df[gain_col[0]] / df[gain_col[0]].sum() * 100
    return df


@st.cache_data
def load_model_metadata() -> dict:
    with open(_path("model", "lap_time_model", "model_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_category_mappings() -> dict:
    with open(_path("data", "training_data", "category_mappings.json")) as f:
        return json.load(f)


@st.cache_data
def build_driver_id_map() -> dict:
    df = load_historical_strategies()
    return df.groupby("driver")["driver_id"].max().astype(int).to_dict()


@st.cache_data
def build_team_encoding_map() -> dict:
    mappings = load_category_mappings()
    return {v: int(k) for k, v in mappings["team"].items()}


@st.cache_resource
def load_optimizer():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from model.strategy_optimizer.pit_stop_optimizer import optimize_strategy
    return optimize_strategy


# ── Phase 2 loaders ────────────────────────────────────────────────────────

@st.cache_resource
def load_race_outcome_model():
    """Load (or auto-train) the race outcome XGBoost model."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from model.race_outcome.race_outcome_model import load_model, predict_position
    model = load_model()
    return predict_position


@st.cache_resource
def load_qualifying_model():
    """Load (or auto-train) the qualifying position XGBoost model."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from model.qualifying_model.qualifying_model import load_model, predict_grid_position
    model = load_model()
    return predict_grid_position


@st.cache_resource
def load_driver_form_module():
    """Return driver form functions (no training needed)."""
    from model.driver_form.driver_form import get_driver_form, rank_drivers_by_form
    return get_driver_form, rank_drivers_by_form


@st.cache_data
def load_driver_prediction_stats() -> dict:
    """
    Per-driver median stats for use in race outcome and qualifying predictions.

    Merges driver_avg_lap_time, driver_consistency_score, driver_win_rate,
    driver_podium_rate from the feature dataset with driver_performance_score
    from driver_performance_metrics_v2.csv.

    Returns dict: driver_name → {driver_avg_lap_time, driver_performance_score,
                                  driver_consistency_score, driver_win_rate, driver_podium_rate}
    """
    data_path = _path("data", "f1_features_dataset.csv")
    df = pd.read_csv(data_path, usecols=[
        "driver", "driver_consistency_score", "driver_win_rate",
        "driver_podium_rate", "driver_avg_lap_time",
    ])
    stats = (
        df.groupby("driver")
        .agg(
            driver_avg_lap_time=("driver_avg_lap_time", "median"),
            driver_consistency_score=("driver_consistency_score", "median"),
            driver_win_rate=("driver_win_rate", "median"),
            driver_podium_rate=("driver_podium_rate", "median"),
        )
    )
    # Merge driver_performance_score from the metrics CSV
    metrics_path = _path("data", "processed", "driver_performance_metrics_v2.csv")
    if os.path.exists(metrics_path):
        perf_df = pd.read_csv(metrics_path, usecols=["driver", "driver_performance_score"])
        perf_map = perf_df.set_index("driver")["driver_performance_score"].to_dict()
        stats["driver_performance_score"] = stats.index.map(perf_map)
    else:
        stats["driver_performance_score"] = np.nan

    # Fill NaNs with dataset-wide medians
    for col in stats.columns:
        stats[col] = stats[col].fillna(stats[col].median())
    return stats.to_dict("index")


@st.cache_data
def load_circuit_dna() -> pd.DataFrame:
    """
    Load circuit DNA fingerprint CSV, generating it via circuit_dna.run()
    if the file does not yet exist.
    """
    dna_path = _path("data", "processed", "circuit_dna.csv")
    if not os.path.exists(dna_path):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            from model.circuit_dna.circuit_dna import run
            run(save=True)
    return pd.read_csv(dna_path)


@st.cache_data
def load_circuit_archetypes() -> dict:
    """Load circuit archetypes JSON, generating via circuit_dna.run() if absent."""
    arch_path = _path("data", "processed", "circuit_archetypes.json")
    if not os.path.exists(arch_path):
        load_circuit_dna()  # triggers generation
    with open(arch_path) as f:
        return json.load(f)
