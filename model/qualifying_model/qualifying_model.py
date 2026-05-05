"""
qualifying_model.py — Phase 2 qualifying position predictor
============================================================
Predicts a driver's expected grid/qualifying position at a circuit before
the race weekend, based on historical driver speed and circuit characteristics.

Features
--------
  driver_avg_lap_time, driver_consistency_score, driver_win_rate,
  driver_podium_rate, circuit_baseline_pace, race_year

Target: grid_position (1–20, XGBoost regression)

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
MODEL_PKL = Path(__file__).resolve().parent / "qualifying_model.pkl"

FEATURE_COLS = [
    "driver_avg_lap_time",
    "driver_consistency_score",
    "driver_win_rate",
    "driver_podium_rate",
    "circuit_baseline_pace",
    "race_year",
]
TARGET = "grid_position"


def _build_quali_df(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Aggregate lap-level data to one row per (race_year, circuit, driver)."""
    agg = (
        df.groupby(["race_year", "circuit", "driver"], sort=False)
        .agg(
            grid_position=("grid_position", "first"),
            driver_avg_lap_time=("driver_avg_lap_time", "first"),
            driver_consistency_score=("driver_consistency_score", "first"),
            driver_win_rate=("driver_win_rate", "first"),
            driver_podium_rate=("driver_podium_rate", "first"),
        )
        .reset_index()
    )
    agg["circuit_baseline_pace"] = agg["circuit"].map(baselines)
    agg = agg.dropna(subset=["grid_position", "circuit_baseline_pace"])

    fill_cols = [
        "driver_avg_lap_time", "driver_consistency_score",
        "driver_win_rate", "driver_podium_rate",
    ]
    for col in fill_cols:
        agg[col] = agg[col].fillna(agg[col].median())

    return agg


def train() -> xgb.XGBRegressor:
    """Train XGBoost regressor, save to MODEL_PKL, return model."""
    df = pd.read_csv(DATA_CSV)
    with open(BASELINES_JSON) as f:
        baselines = json.load(f)

    quali_df = _build_quali_df(df, baselines)
    train_df = quali_df[quali_df["race_year"] < 2025]
    test_df = quali_df[quali_df["race_year"] == 2025]

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
    print(f"qualifying_model: MAE={mae:.2f} positions | within ±3={within_3:.1f}%")

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    return model


def load_model() -> xgb.XGBRegressor:
    """Load saved model, auto-training if the pkl does not yet exist."""
    if not MODEL_PKL.exists():
        return train()
    return joblib.load(MODEL_PKL)


def predict_grid_position(
    circuit: str,
    driver_metrics: dict,
    baselines: dict,
    race_year: int = 2025,
) -> dict:
    """
    Predict qualifying/grid position for a single driver.

    Parameters
    ----------
    circuit        : str, e.g. "Bahrain Grand Prix"
    driver_metrics : dict with driver feature keys (driver_avg_lap_time, etc.)
    baselines      : dict from circuit_baselines.json
    race_year      : int, year of the race (default 2025)

    Returns
    -------
    {predicted_grid, grid_rounded}
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = load_model()

    fallback_baseline = float(np.mean(list(baselines.values())))
    circuit_baseline_pace = float(baselines.get(circuit, fallback_baseline))

    row = {
        "driver_avg_lap_time": float(driver_metrics.get("driver_avg_lap_time", 90.0)),
        "driver_consistency_score": float(driver_metrics.get("driver_consistency_score", 0.5)),
        "driver_win_rate": float(driver_metrics.get("driver_win_rate", 0.0)),
        "driver_podium_rate": float(driver_metrics.get("driver_podium_rate", 0.0)),
        "circuit_baseline_pace": circuit_baseline_pace,
        "race_year": race_year,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]
    pred = float(np.clip(model.predict(X)[0], 1, 20))
    grid_rounded = int(round(pred))

    return {
        "predicted_grid": pred,
        "grid_rounded": grid_rounded,
    }


if __name__ == "__main__":
    print("Training qualifying model...")
    train()
