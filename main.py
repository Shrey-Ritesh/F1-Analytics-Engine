import argparse
import os
import logging
from ingestion.fetch_data import setup_cache, fetch_season_data
from preprocessing.clean_data import clean_and_merge_data
from feature_engineering.build_features import build_feature_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline(year: int):
    """
    Orchestrates the data pipeline for a specific year:
    1. Ingestion
    2. Preprocessing
    3. Feature Engineering
    """
    logging.info(f"--- STARTING PIPELINE FOR YEAR {year} ---")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    
    # Setup Data Directory & Cache
    cache_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    setup_cache(cache_dir)
    
    # 1. Ingestion
    fetch_season_data(year, data_dir)
    
    # 2. Preprocessing
    clean_and_merge_data(year, data_dir)
    
    # 3. Feature Engineering
    build_feature_dataset(year, data_dir)
    # 4. Extended Feature Expansion
    from feature_engineering.expand_features import expand_features
    expand_features(data_dir)
    
    logging.info(f"--- PIPELINE FOR YEAR {year} COMPLETE ---")

def verify_master_dataset():
    """
    Basic verification to ensure master_dataset.csv is created correctly.
    """
    master_path = os.path.join(os.path.dirname(__file__), 'data', 'master_dataset.csv')
    if not os.path.exists(master_path):
        logging.error(f"Verification failed: {master_path} does not exist.")
        return
        
    import pandas as pd
    try:
        df = pd.read_csv(master_path)
        logging.info(f"Verification passed: {master_path} exists with {len(df)} rows.")
        logging.info(f"Columns present: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Verification failed when reading {master_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Data Pipeline")
    parser.add_argument('--year', type=int, default=2025, help="Race season year to process.")
    args = parser.parse_args()
    
    run_pipeline(args.year)
    verify_master_dataset()
