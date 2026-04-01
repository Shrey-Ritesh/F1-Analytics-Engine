import os
import ast
import json
import contextlib
import io

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


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
