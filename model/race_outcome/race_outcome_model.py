"""
race_outcome_model.py — Phase 2 race outcome predictor
=======================================================
Predicts a driver's expected final race position given pre-race inputs.

Features (all available before the race starts)
--------
  grid_position, track_temperature, circuit_baseline_pace,
  driver_performance_score, driver_avg_lap_time, driver_consistency_score,
  driver_win_rate, driver_podium_rate

Target: final_race_position (1–20, XGBoost regression)

Training split: 2023–2024 → train, 2025 → test (temporal split, zero leakage)
"""

import json
import contextlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "f1_features_dataset.csv"
BASELINES_JSON = PROJECT_ROOT / "model" / "lap_time_model" / "circuit_baselines.json"
MODEL_PKL = Path(__file__).resolve().parent / "race_outcome_model.pkl"

FEATURE_COLS = [
    "grid_position",
    "track_temperature",
    "circuit_baseline_pace",
    "driver_avg_lap_time",
    "driver_consistency_score",
    "driver_win_rate",
    "driver_podium_rate",
]
TARGET = "final_race_position"


def _build_race_df(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Aggregate lap-level data to one row per (race_year, circuit, driver)."""
    agg = (
        df.groupby(["race_year", "circuit", "driver"], sort=False)
        .agg(
            final_race_position=("final_race_position", "first"),
            grid_position=("grid_position", "first"),
            track_temperature=("track_temperature", "mean"),
            driver_avg_lap_time=("driver_avg_lap_time", "first"),
            driver_consistency_score=("driver_consistency_score", "first"),
            driver_win_rate=("driver_win_rate", "first"),
            driver_podium_rate=("driver_podium_rate", "first"),
        )
        .reset_index()
    )
    agg["circuit_baseline_pace"] = agg["circuit"].map(baselines)
    agg = agg.dropna(subset=["final_race_position", "circuit_baseline_pace"])

    fill_cols = [
        "driver_avg_lap_time",
        "driver_consistency_score", "driver_win_rate", "driver_podium_rate",
    ]
    for col in fill_cols:
        agg[col] = agg[col].fillna(agg[col].median())

    return agg


def train() -> xgb.XGBRegressor:
    """Train XGBoost regressor, save to MODEL_PKL, return model."""
    df = pd.read_csv(DATA_CSV)
    with open(BASELINES_JSON) as f:
        baselines = json.load(f)

    race_df = _build_race_df(df, baselines)
    train_df = race_df[race_df["race_year"] < 2025]
    test_df = race_df[race_df["race_year"] == 2025]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = np.clip(model.predict(X_test), 1, 20)
    mae = float(np.mean(np.abs(preds - y_test.values)))
    within_3 = float(np.mean(np.abs(preds - y_test.values) <= 3) * 100)
    print(f"race_outcome_model: MAE={mae:.2f} positions | within ±3={within_3:.1f}%")

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    return model


def load_model() -> xgb.XGBRegressor:
    """Load saved model, auto-training if the pkl does not yet exist."""
    if not MODEL_PKL.exists():
        return train()
    return joblib.load(MODEL_PKL)


def predict_position(
    grid_position: int,
    track_temperature: float,
    circuit: str,
    driver_metrics: dict,
    baselines: dict,
) -> dict:
    """
    Predict final race position for a single driver.

    Parameters
    ----------
    grid_position     : int 1–20
    track_temperature : float, °C
    circuit           : str, e.g. "Bahrain Grand Prix"
    driver_metrics    : dict with keys from FEATURE_COLS[3:]
    baselines         : dict from circuit_baselines.json

    Returns
    -------
    {predicted_position, position_rounded, category}
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = load_model()

    fallback_baseline = float(np.mean(list(baselines.values())))
    circuit_baseline_pace = float(baselines.get(circuit, fallback_baseline))

    row = {
        "grid_position": grid_position,
        "track_temperature": track_temperature,
        "circuit_baseline_pace": circuit_baseline_pace,
        **{k: float(driver_metrics.get(k, 0.0)) for k in FEATURE_COLS[3:]},  # driver metrics
    }
    X = pd.DataFrame([row])[FEATURE_COLS]
    pred = float(np.clip(model.predict(X)[0], 1, 20))
    pos_rounded = int(round(pred))

    if pos_rounded == 1:
        category = "Win"
    elif pos_rounded <= 3:
        category = "Podium"
    elif pos_rounded <= 10:
        category = "Points"
    else:
        category = "Out of Points"

    return {
        "predicted_position": pred,
        "position_rounded": pos_rounded,
        "category": category,
    }


if __name__ == "__main__":
    print("Training race outcome model...")
    train()
