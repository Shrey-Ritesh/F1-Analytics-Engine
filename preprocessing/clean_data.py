import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_and_merge_data(year: int, data_dir: str):
    """
    Reads raw laps and results data for a given year, cleans them,
    and merges them into a single comprehensive dataset.
    """
    laps_path = os.path.join(data_dir, f'raw_laps_{year}.csv')
    results_path = os.path.join(data_dir, f'raw_results_{year}.csv')
    
    if not os.path.exists(laps_path) or not os.path.exists(results_path):
        logging.error(f"Raw data for {year} not found in {data_dir}. Run ingestion first.")
        return None
        
    logging.info(f"Loading raw data for {year}...")
    laps_df = pd.read_csv(laps_path)
    results_df = pd.read_csv(results_path)
    
    logging.info("Cleaning laps data...")
    # Clean Lap Times (pandas parses timedelta as string like "0 days 00:01:23.456000")
    # Using pd.to_timedelta and extracting total seconds.
    if 'LapTime' in laps_df.columns:
        laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime'], errors='coerce')
        laps_df['lap_time_seconds'] = laps_df['LapTime'].dt.total_seconds()
    else:
        laps_df['lap_time_seconds'] = np.nan
        
    # Pit stops: FastF1 laps marks in/out laps. We can proxy pit stops using PitOutTime or PitInTime.
    # If PitInTime is not null, it's a pit stop lap.
    if 'PitInTime' in laps_df.columns:
        laps_df['pit_stop'] = laps_df['PitInTime'].notna().astype(int)
    else:
        laps_df['pit_stop'] = 0
        
    # Standardize column names for laps to lowercase
    laps_df.columns = [c.lower() for c in laps_df.columns]
    
    logging.info("Cleaning results data...")
    # In results, 'DriverNumber' or 'Abbreviation' can be used to standardize driver names
    if 'Abbreviation' in results_df.columns:
        results_df['driver'] = results_df['Abbreviation']
    elif 'DriverNumber' in results_df.columns:
        results_df['driver'] = 'Driver_' + results_df['DriverNumber'].astype(str)
    else:
        results_df['driver'] = 'Unknown'
        
    # Standardize result columns
    results_df.columns = [c.lower() for c in results_df.columns]
    
    logging.info("Merging datasets...")
    # Merge Laps with Results on drivernumber (which maps to driver in laps) and race_year/event_name
    # FastF1 lap data has 'drivernumber'
    
    merge_cols = ['drivernumber', 'race_year', 'event_name']
    
    # Ensure columns exist before merging
    avail_merge_cols = [c for c in merge_cols if c in laps_df.columns and c in results_df.columns]
    
    merged_df = pd.merge(laps_df, results_df, on=avail_merge_cols, how='left', suffixes=('_lap', '_res'))
    
    # Drop rows without lap times to clean up NaNs (e.g., Red Flags)
    merged_df = merged_df.dropna(subset=['lap_time_seconds'])
    
    cleaned_path = os.path.join(data_dir, f'cleaned_data_{year}.csv')
    merged_df.to_csv(cleaned_path, index=False)
    logging.info(f"Cleaned dataset saved to {cleaned_path} with {len(merged_df)} rows.")
    
    return merged_df

if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    clean_and_merge_data(2025, target_dir)
