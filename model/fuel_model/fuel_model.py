"""
fuel_model.py — Phase 2 fuel load and tire degradation corrections
=================================================================
Provides fuel-corrected lap times and improved tire degradation rates
for use by circuit_dna.py and other Phase 2 modules.

Key changes vs existing pipeline:
  - fuel_load_kg:             110 kg start, 1.85 kg/lap burn (was 1.5 kg/lap)
  - fuel_corrected_lap_time:  lap_time_seconds minus fuel weight penalty
  - tire_deg_rate_v2:         linear regression slope per stint on fuel-corrected
                              times, skipping first 2 warm-up laps
"""

import numpy as np
import pandas as pd


def compute_fuel_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `fuel_load_kg` column: 110 kg start, 1.85 kg burned per lap, clipped to 0.

    Parameters
    ----------
    df : DataFrame with at minimum a `lap` column (1-based lap number).

    Returns
    -------
    df with new column `fuel_load_kg` added.
    """
    df = df.copy()
    df["fuel_load_kg"] = (110.0 - df["lap"] * 1.85).clip(lower=0.0)
    return df


def fuel_correct_lap_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `fuel_corrected_lap_time` column: lap_time_seconds adjusted for fuel load.

    The standard F1 fuel weight time penalty is 0.03 s per kg of fuel.
    Requires `fuel_load_kg` column — calls `compute_fuel_load` automatically
    if it is absent.

    Parameters
    ----------
    df : DataFrame with `lap_time_seconds` (and `lap` if fuel_load_kg is missing).

    Returns
    -------
    df with new column `fuel_corrected_lap_time` added.
    """
    df = df.copy()
    if "fuel_load_kg" not in df.columns:
        df = compute_fuel_load(df)
    df["fuel_corrected_lap_time"] = df["lap_time_seconds"] - (df["fuel_load_kg"] * 0.03)
    return df


def compute_tire_degradation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `tire_deg_rate_v2` column: per-stint linear regression slope of
    fuel-corrected lap time vs. tire age, skipping the first 2 warm-up laps.

    Groups by (race_year, circuit, driver, stint_number).  For each group:
      - Exclude laps where tire_age_laps <= 2 (warm-up / cold tyre phase).
      - If fewer than 3 data points remain, assign NaN for the whole stint.
      - Otherwise fit degree-1 polynomial (np.polyfit) of
        fuel_corrected_lap_time ~ tire_age_laps and use the slope.

    Parameters
    ----------
    df : DataFrame containing fuel_corrected_lap_time, tire_age_laps,
         stint_number, race_year, circuit, driver.

    Returns
    -------
    df with new column `tire_deg_rate_v2` added.
    """
    df = df.copy()

    group_keys = ["race_year", "circuit", "driver", "stint_number"]

    # Pre-allocate output series with NaN
    deg_rate = pd.Series(np.nan, index=df.index, dtype=float)

    for keys, grp in df.groupby(group_keys, sort=False):
        # Skip warm-up laps
        mask = grp["tire_age_laps"] > 2
        valid = grp[mask]

        if len(valid) < 3:
            # Not enough points — leave as NaN
            continue

        x = valid["tire_age_laps"].to_numpy(dtype=float)
        y = valid["fuel_corrected_lap_time"].to_numpy(dtype=float)

        # np.polyfit returns [slope, intercept] for degree=1
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Assign slope to ALL laps in this stint (constant per stint)
        deg_rate.loc[grp.index] = slope

    df["tire_deg_rate_v2"] = deg_rate
    return df


def apply_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: applies compute_fuel_load → fuel_correct_lap_time →
    compute_tire_degradation_v2 in sequence and prints a summary.

    Parameters
    ----------
    df : Raw feature DataFrame.

    Returns
    -------
    df with columns fuel_load_kg, fuel_corrected_lap_time, tire_deg_rate_v2 added.
    """
    df = compute_fuel_load(df)
    df = fuel_correct_lap_time(df)
    df = compute_tire_degradation_v2(df)
    print(
        f"fuel_model: added fuel_load_kg, fuel_corrected_lap_time, "
        f"tire_deg_rate_v2 to {len(df)} rows"
    )
    return df
