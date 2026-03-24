import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_feature_dataset(year: int, data_dir: str):
    """
    Extracts relevant features for machine learning models and saves them to master_dataset.csv.
    Required columns: driver, circuit, lap, tire_compound, track_temperature, lap_time_seconds, position, pit_stop, race_year.
    """
    year_dir = os.path.join(data_dir, 'training_data', str(year))
    cleaned_path = os.path.join(year_dir, f'cleaned_data_{year}.csv')
    if not os.path.exists(cleaned_path):
        logging.error(f"Cleaned data for {year} not found. Run preprocessing first.")
        return None
        
    logging.info(f"Loading cleaned data from {cleaned_path}")
    df = pd.read_csv(cleaned_path)
    
    # Required columns map
    required_columns = {
        'driver': 'abbreviation',
        'driver_id': 'drivernumber',
        'team': 'teamname',
        'circuit': 'circuit_lap', # Depending on the suffix after merge
        'lap': 'lapnumber',
        'tire_compound': 'compound',
        'track_temperature': 'tracktemp',
        'lap_time_seconds': 'lap_time_seconds',
        'position': 'position_lap',
        'pit_stop': 'pit_stop',
        'race_year': 'race_year',
        'round_number': 'round_number_lap'
    }
    
    # Map available columns to our requested target feature names
    features_df = pd.DataFrame()
    for target_col, source_col in required_columns.items():
        # Handle suffixes if column names collided during merge
        if source_col in df.columns:
            features_df[target_col] = df[source_col]
        elif target_col == 'circuit' and 'circuit' in df.columns:
            features_df[target_col] = df['circuit']
        elif target_col == 'race_year' and 'race_year' in df.columns:
            features_df[target_col] = df['race_year']
        elif target_col == 'position' and 'position' in df.columns:
             features_df[target_col] = df['position']
        else:
            logging.warning(f"Could not find exact source column '{source_col}' for target '{target_col}'. Trying alternatives.")
            features_df[target_col] = np.nan
            
    # Handle missing track temps if any (fill with median or mean for simplicity here)
    if 'track_temperature' in features_df.columns:
        features_df['track_temperature'] = pd.to_numeric(features_df['track_temperature'], errors='coerce')
        features_df['track_temperature'] = features_df['track_temperature'].fillna(method='ffill')
        
    # Drop rows that still have crucial missing target variables
    features_df = features_df.dropna(subset=['lap_time_seconds', 'driver', 'lap'])
    
    master_path = os.path.join(data_dir, 'master_dataset.csv')
    
    # If the master dataset exists, we append to it, otherwise create it.
    if os.path.exists(master_path):
        logging.info(f"Appending data to existing {master_path}")
        features_df.to_csv(master_path, mode='a', header=False, index=False)
    else:
        logging.info(f"Creating new {master_path}")
        features_df.to_csv(master_path, index=False)
        
    logging.info(f"Master dataset populated successfully for {year}. Total rows added: {len(features_df)}")
    return features_df

if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    build_feature_dataset(2025, target_dir)
