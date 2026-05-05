"""
driver_form.py — Phase 2 driver rolling form tracker
=====================================================
Computes rolling form metrics for each driver based on their most recent
race results. No ML — rolling statistics over race-level outcomes.

Note: Racing Bulls triple encoding (AlphaTauri/RB/Racing Bulls codes 1/9/10)
is not relevant here since we group by driver name, not team.

Public API
----------
  get_driver_form(driver_name, n_races=5)  →  dict
  rank_drivers_by_form(n_races=5)          →  DataFrame
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "f1_features_dataset.csv"

_RACE_RESULTS_CACHE: Optional[pd.DataFrame] = None


def _load_race_results() -> pd.DataFrame:
    """Return one row per (race_year, circuit, driver) — race-level results only."""
    df = pd.read_csv(
        DATA_CSV,
        usecols=["race_year", "circuit", "driver", "final_race_position",
                 "grid_position", "podium_finish", "race_winner", "round_number"],
    )
    race_df = (
        df.groupby(["race_year", "circuit", "driver"], sort=False)
        .agg(
            final_race_position=("final_race_position", "first"),
            grid_position=("grid_position", "first"),
            podium_finish=("podium_finish", "first"),
            race_winner=("race_winner", "first"),
            round_number=("round_number", "first"),
        )
        .reset_index()
    )
    race_df = race_df.sort_values(["race_year", "round_number"]).reset_index(drop=True)
    return race_df


def _get_race_results() -> pd.DataFrame:
    global _RACE_RESULTS_CACHE
    if _RACE_RESULTS_CACHE is None:
        _RACE_RESULTS_CACHE = _load_race_results()
    return _RACE_RESULTS_CACHE


def get_driver_form(driver_name: str, n_races: int = 5) -> dict:
    """
    Compute rolling form for a driver over their last n_races.

    Parameters
    ----------
    driver_name : str, full name as it appears in the dataset (e.g. "Max Verstappen")
    n_races     : int, number of most recent races to evaluate (default 5)

    Returns
    -------
    dict:
      driver, n_races_used, recent_positions (list[int]), recent_circuits (list[str]),
      mean_position (float), form_trend (float, negative = improving),
      form_score (float 0–1, higher = better current form),
      form_label (str: "Improving" | "Declining" | "Stable"),
      win_count (int), podium_count (int), points_count (int)
    """
    race_df = _get_race_results()
    driver_df = race_df[race_df["driver"] == driver_name].copy()

    if driver_df.empty:
        return {"driver": driver_name, "error": "Driver not found in dataset"}

    driver_df = driver_df.dropna(subset=["final_race_position"]).tail(n_races)

    if driver_df.empty:
        return {"driver": driver_name, "error": "No race results with valid positions"}

    positions = [int(p) for p in driver_df["final_race_position"].tolist()]
    circuits = driver_df["circuit"].tolist()
    wins = int(driver_df["race_winner"].sum())
    podiums = int(driver_df["podium_finish"].sum())
    points = int((driver_df["final_race_position"] <= 10).sum())

    mean_pos = float(np.mean(positions))

    # Linear trend: negative slope = getting better (lower position numbers)
    if len(positions) >= 2:
        x = np.arange(len(positions), dtype=float)
        slope = float(np.polyfit(x, positions, 1)[0])
    else:
        slope = 0.0

    # form_score: 1.0 = winning every race, 0.0 = last every race
    form_score = float(np.clip(1.0 - (mean_pos - 1.0) / 19.0, 0.0, 1.0))

    if slope < -0.5:
        form_label = "Improving"
    elif slope > 0.5:
        form_label = "Declining"
    else:
        form_label = "Stable"

    return {
        "driver": driver_name,
        "n_races_used": len(positions),
        "recent_positions": positions,
        "recent_circuits": circuits,
        "mean_position": round(mean_pos, 2),
        "form_trend": round(slope, 3),
        "form_score": round(form_score, 3),
        "form_label": form_label,
        "win_count": wins,
        "podium_count": podiums,
        "points_count": points,
    }


def rank_drivers_by_form(n_races: int = 5) -> pd.DataFrame:
    """
    Return a DataFrame of all drivers ranked by form_score descending.

    Parameters
    ----------
    n_races : int, rolling window size (default 5)

    Returns
    -------
    DataFrame with columns: driver, mean_position, form_trend, form_score,
                            form_label, win_count, podium_count, points_count, n_races_used
    """
    race_df = _get_race_results()
    drivers = sorted(race_df["driver"].unique())

    rows = []
    for driver in drivers:
        form = get_driver_form(driver, n_races)
        if "error" not in form:
            rows.append(form)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)[[
        "driver", "mean_position", "form_trend", "form_score",
        "form_label", "win_count", "podium_count", "points_count", "n_races_used",
    ]]
    return out.sort_values("form_score", ascending=False).reset_index(drop=True)
