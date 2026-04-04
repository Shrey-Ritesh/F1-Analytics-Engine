import pandas as pd
import numpy as np
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def expand_features(data_dir: str):
    """
    Reads the baseline master_dataset.csv and raw data (if necessary for missing columns
    like weather, sectors, etc) and computes all the derived strategy and ML features.
    Outputs to data/f1_features_dataset.csv
    """
    master_path = os.path.join(data_dir, 'master_dataset.csv')
    if not os.path.exists(master_path):
        logging.error(f"Base {master_path} not found. Run the initial pipeline first.")
        return None
        
    logging.info(f"Loading base dataset from {master_path}")
    df = pd.read_csv(master_path)
    
    # --- Feature Expansion ---
    logging.info("Computing extended features...")
    
    # Sort chronologically by race year, round number, then driver and lap
    if 'round_number' in df.columns:
        df = df.sort_values(by=['race_year', 'round_number', 'driver', 'lap'])
    else:
        df = df.sort_values(by=['race_year', 'circuit', 'driver', 'lap'])
    
    # 1. Basic Race Information 
    # (race_year, circuit, lap exist).
    # We will derive `laps_remaining` by computing the max laps for each circuit.
    max_laps = df.groupby(['race_year', 'circuit'])['lap'].transform('max')
    df['laps_remaining'] = max_laps - df['lap']
    
    # 2. Tire & Strategy Information
    # Calculate stint number and tire age. 
    # A new stint starts when pit_stop == 1.
    # Stint number increments cumulatively every time there is a pit stop for a driver in a race.
    df['stint_number'] = df.groupby(['race_year', 'circuit', 'driver'])['pit_stop'].cumsum() + 1
    
    # Tire age increments within a stint. 
    # It resets to 1 after a pit stop.
    df['tire_age_laps'] = df.groupby(['race_year', 'circuit', 'driver', 'stint_number']).cumcount() + 1
    
    # Stint length is the total number of laps in that specific stint
    df['stint_length'] = df.groupby(['race_year', 'circuit', 'driver', 'stint_number'])['lap'].transform('count')
    
    # 3. Race State Information
    # position already exists.
    # positions_gained from the start of the race.
    # We need grid_position (lap 1 position)
    grid_positions = df[df['lap'] == 1][['race_year', 'circuit', 'driver', 'position']].rename(columns={'position': 'grid_position'})
    df = pd.merge(df, grid_positions, on=['race_year', 'circuit', 'driver'], how='left')
    df['positions_gained'] = df['grid_position'] - df['position']
    
    # Gap to leader and gap to car ahead
    # To compute this properly, we need cumulative race times or actual telemetry gaps.
    # Since we only have lap_time_seconds, we can compute cumulative time.
    df['cumulative_race_time'] = df.groupby(['race_year', 'circuit', 'driver'])['lap_time_seconds'].cumsum()
    
    # Leader's cumulative time per lap
    leader_time = df.groupby(['race_year', 'circuit', 'lap'])['cumulative_race_time'].transform('min')
    df['gap_to_leader_seconds'] = df['cumulative_race_time'] - leader_time
    
    # Gap to car ahead
    # Sort by race, lap, and position
    df = df.sort_values(by=['race_year', 'circuit', 'lap', 'position'])
    df['time_of_car_ahead'] = df.groupby(['race_year', 'circuit', 'lap'])['cumulative_race_time'].shift(1)
    df['gap_to_car_ahead_seconds'] = df['cumulative_race_time'] - df['time_of_car_ahead']
    df['gap_to_car_ahead_seconds'] = df['gap_to_car_ahead_seconds'].fillna(0) # Leader has 0 gap
    
    # 4. Driver Information
    # driver_id is now explicitly loaded from master_dataset.csv (actual driver number)
    
    # Historical driver average lap time and consistency
    # MUST PREVENT LEAKAGE: Look strictly at prior races (round_number < current_round)
    historical_stats = []
    for drv in df['driver'].unique():
        driver_data = df[df['driver'] == drv]
        for rnd in driver_data['round_number'].unique():
            past_data = driver_data[driver_data['round_number'] < rnd]
            if len(past_data) > 0:
                avg_lap = past_data['lap_time_seconds'].median()
                std_lap = past_data['lap_time_seconds'].std()
            else:
                # Fallback for the first race of the season. 
                # We use -1.0 as a safe numeric representation for 'no history' so dropna() does not delete the entire first race.
                avg_lap = -1.0
                std_lap = -1.0
            historical_stats.append({
                'driver': drv,
                'round_number': rnd,
                'driver_avg_lap_time': avg_lap,
                'driver_consistency_score': std_lap
            })
    
    hist_stats_df = pd.DataFrame(historical_stats)
    df = pd.merge(df, hist_stats_df, on=['driver', 'round_number'], how='left')
    
    # Win rate / Podium rate
    # MUST PREVENT LEAKAGE: Calculate based only on races completed *before* the current race.
    final_positions = df.loc[df.groupby(['race_year', 'round_number', 'driver'])['lap'].idxmax()].copy()
    final_positions = final_positions.sort_values(by=['race_year', 'round_number'])
    final_positions['is_winner'] = (final_positions['position'] == 1).astype(int)
    final_positions['is_podium'] = (final_positions['position'] <= 3).astype(int)
    
    # Safely compute cumulative sums shifting strictly within each driver's group
    final_positions['cum_wins'] = final_positions.groupby('driver')['is_winner'].transform(lambda x: x.cumsum().shift(1).fillna(0))
    final_positions['cum_podiums'] = final_positions.groupby('driver')['is_podium'].transform(lambda x: x.cumsum().shift(1).fillna(0))
    final_positions['cum_races'] = final_positions.groupby('driver').cumcount()
    
    final_positions['driver_win_rate'] = np.where(final_positions['cum_races'] > 0, final_positions['cum_wins'] / final_positions['cum_races'], 0.0)
    final_positions['driver_podium_rate'] = np.where(final_positions['cum_races'] > 0, final_positions['cum_podiums'] / final_positions['cum_races'], 0.0)
    
    rates_map = final_positions[['race_year', 'round_number', 'driver', 'driver_win_rate', 'driver_podium_rate']]
    df = pd.merge(df, rates_map, on=['race_year', 'round_number', 'driver'], how='left')
    
    # Final Target variables mapping backward to all laps
    final_pos_map = final_positions[['race_year', 'circuit', 'driver', 'position', 'is_winner', 'is_podium']]
    final_pos_map = final_pos_map.rename(columns={'position': 'final_race_position', 'is_winner': 'race_winner', 'is_podium': 'podium_finish'})
    df = pd.merge(df, final_pos_map, on=['race_year', 'circuit', 'driver'], how='left')
    
    # 5. Derived Strategy Metrics
    # Tire degradation rate (slope of lap time increase over a stint)
    # A simple proxy: (current lap time - min lap time in stint) / tire age
    stint_min_time = df.groupby(['race_year', 'circuit', 'driver', 'stint_number'])['lap_time_seconds'].transform('min')
    df['tire_degradation_rate'] = (df['lap_time_seconds'] - stint_min_time) / df['tire_age_laps']
    
    # Fuel load estimate: assuming ~110kg fuel at start, ~0kg at end. ~1.5kg used per lap.
    df['fuel_load_estimate'] = 110.0 - (df['lap'] * 1.5)
    df['fuel_load_estimate'] = df['fuel_load_estimate'].clip(lower=0.0) # Cannot be negative
    
    df['expected_lap_time'] = df.groupby(['race_year', 'circuit', 'tire_compound'])['lap_time_seconds'].transform('median')
    
    # Measures how fast a driver is compared to the expected baseline lap time
    # Negative value -> faster than expectation, Positive value -> slower
    df['relative_pace_delta'] = df['lap_time_seconds'] - df['expected_lap_time']
    
    # --- NEW EXPERT FEATURES ---
    
    # a) tire_cold_flag: binary 1 if tire_age_laps <= 2 (cold tire, pre-thermal window)
    df['tire_cold_flag'] = (df['tire_age_laps'] <= 2).astype(int)
    
    # b) compound_base_deg_rate: mapping generic tire compound to a float degradation constant
    tire_deg_map = {'SOFT': 0.035, 'MEDIUM': 0.022, 'HARD': 0.014, 'INTERMEDIATE': 0.008, 'WET': 0.005}
    df['compound_base_deg_rate'] = df['tire_compound'].map(tire_deg_map).fillna(0.01)
    
    # c) adjusted_tire_stress: pressure applied based on how far in the specific tire life we are
    df['adjusted_tire_stress'] = df['tire_age_laps'] * df['compound_base_deg_rate']
    
    # d) fuel_time_effect: estimated lap time cost from fuel mass (~0.03s per kg in F1)
    df['fuel_time_effect'] = df['fuel_load_estimate'] * 0.03
    
    # e) stint_progress_pct: tracks how far through the specific tire stint we are (0.0 to 1.0)
    df['stint_progress_pct'] = df['tire_age_laps'] / df['stint_length']
    
    # f) dirty_air_flag: binary 1 if gap_to_car_ahead is < 1.0 (wakes degrade aero/tires)
    df['dirty_air_flag'] = (df['gap_to_car_ahead_seconds'] < 1.0).astype(int)
    
    # g) laps_remaining_pct: driver behavior proxy based on remaining race phase
    df['laps_remaining_pct'] = df['laps_remaining'] / (df['lap'] + df['laps_remaining'])
    
    # h) position_vs_grid: relative position tracking compared to starting grid slot
    df['position_vs_grid'] = df['position'] - df['grid_position']
    
    # Clean up and arrange columns
    requested_columns = [
        'race_year', 'round_number', 'circuit', 'lap', 'laps_remaining',
        'driver', 'driver_id', 'team', 'grid_position', 'driver_avg_lap_time', 'driver_consistency_score', 'driver_win_rate', 'driver_podium_rate',
        'position', 'positions_gained', 'gap_to_leader_seconds', 'gap_to_car_ahead_seconds',
        'tire_compound', 'tire_age_laps', 'stint_number', 'stint_length', 'pit_stop',
        'lap_time_seconds', 'track_temperature',
        'tire_degradation_rate', 'fuel_load_estimate', 'expected_lap_time', 'relative_pace_delta',
        'final_race_position', 'podium_finish', 'race_winner',
        'tire_cold_flag', 'compound_base_deg_rate', 'adjusted_tire_stress', 'fuel_time_effect',
        'stint_progress_pct', 'dirty_air_flag', 'laps_remaining_pct', 'position_vs_grid'
    ]
    
    # Add any columns we missed as NaN
    for col in requested_columns:
        if col not in df.columns:
            df[col] = np.nan
            
    # Keep only the requested columns in a tidy format
    final_df = df[requested_columns]
    
    # Ensure final output is sorted chronologically by calendar race order
    if 'round_number' in final_df.columns:
        final_df = final_df.sort_values(by=['race_year', 'round_number', 'lap', 'position'])

    # ML models generally cannot handle NaN values natively.
    # We drop any row that has a missing feature (including the very first race which has no historical leakage data).
    final_df = final_df.dropna()
        
    out_path = os.path.join(data_dir, 'f1_features_dataset.csv')
    final_df.to_csv(out_path, index=False)
    logging.info(f"Successfully generated expanded dataset at {out_path} with {len(final_df)} rows and {len(final_df.columns)} features.")
    
    # --- Create explicit ML training dataset with Encodings ---
    training_dir = os.path.join(data_dir, 'training_data')
    os.makedirs(training_dir, exist_ok=True)
    
    ml_df = final_df.copy()
    
    categorical_cols = ['team', 'circuit', 'tire_compound']
    mappings_dict = {}
    
    for col in categorical_cols:
        if col in ml_df.columns:
            # Convert to categorical to extract codes
            ml_df[col] = ml_df[col].astype('category')
            # Create a dictionary mapping the code to the original string
            mappings_dict[col] = dict(enumerate(ml_df[col].cat.categories))
            # Apply the codes
            ml_df[col] = ml_df[col].cat.codes
            
    # The 'driver' column is a string names list, but we already have 'driver_id' (numeric F1 number).
    # We can drop the string 'driver' from the ML dataset so it doesn't cause TypeErrors in XGBoost.
    if 'driver' in ml_df.columns:
        ml_df = ml_df.drop(columns=['driver'])
            
    # Save the mappings to a JSON file for future reference
    mappings_path = os.path.join(training_dir, 'category_mappings.json')
    with open(mappings_path, 'w') as f:
        json.dump(mappings_dict, f, indent=4, ensure_ascii=False)
        
    # Export the perfectly encoded dataset
    ml_out_path = os.path.join(training_dir, 'f1_training_dataset.csv')
    ml_df.to_csv(ml_out_path, index=False)
    
    logging.info(f"Successfully generated ML Encoded dataset to {ml_out_path}")
    logging.info(f"Saved Category string-to-int mappings to {mappings_path}")
    
    return final_df

if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    expand_features(target_dir)
