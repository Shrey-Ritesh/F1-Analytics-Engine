import argparse
import os
import logging
import pandas as pd
from ingestion.fetch_data import setup_cache, fetch_multiple_seasons
from preprocessing.clean_data import clean_and_merge_data
from feature_engineering.build_features import build_feature_dataset
from feature_engineering.expand_features import expand_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TARGET_SEASONS = [2023, 2024, 2025]

def run_pipeline():
    """
    Orchestrates the full data pipeline:
      Phase 1 — ingestion → preprocessing → feature engineering
      Phase 2 — fuel correction, circuit DNA, model training
    """
    logging.info(f"--- STARTING MULTI-SEASON PIPELINE {TARGET_SEASONS} ---")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    # Clean previous master to avoid duplicate appends spanning across reruns
    master_path = os.path.join(data_dir, 'master_dataset.csv')
    if os.path.exists(master_path):
        os.remove(master_path)

    # Setup Data Directory & Cache
    cache_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    setup_cache(cache_dir)

    # 1. Multi Season Ingestion
    fetch_multiple_seasons(TARGET_SEASONS, data_dir)

    # 2. Iterate preprocessor mapping explicitly mapping year splits
    for year in TARGET_SEASONS:
        clean_and_merge_data(year, data_dir)
        build_feature_dataset(year, data_dir)

    # 3. Validation & Deduplication
    master_df = pd.read_csv(master_path)
    master_df = master_df.drop_duplicates(subset=['race_year', 'round_number', 'driver', 'lap'])
    master_df.to_csv(master_path, index=False)
    logging.info(f"Deduplicated Master Dataset explicitly saved. Total Base Rows: {len(master_df)}")

    # 4. Multi-season Feature Engineering
    logging.info("Engaging Machine Learning expansion module universally...")
    expand_features(data_dir)

    # Validate final engineered structure
    validate_multi_season(data_dir)

    # 5. Phase 2 — fuel correction + circuit DNA
    run_phase2(data_dir)

    logging.info(f"--- MULTI-SEASON PIPELINE COMPLETE ---")


def run_phase2(data_dir: str):
    """
    Phase 2 pipeline:
      - Apply fuel corrections (fuel_load_kg, fuel_corrected_lap_time, tire_deg_rate_v2)
      - Build circuit DNA fingerprints + archetypes
      - Train race outcome and qualifying models
    """
    logging.info("--- PHASE 2: Fuel correction, Circuit DNA, Model training ---")

    features_path = os.path.join(data_dir, 'f1_features_dataset.csv')
    df = pd.read_csv(features_path)

    # 5a. Fuel model — add corrected columns alongside originals (never overwrite)
    logging.info("Applying fuel model corrections...")
    from model.fuel_model.fuel_model import apply_all
    df_corrected = apply_all(df)
    df_corrected.to_csv(features_path, index=False)
    logging.info(
        f"Saved fuel-corrected dataset: "
        f"+fuel_load_kg, +fuel_corrected_lap_time, +tire_deg_rate_v2 "
        f"({len(df_corrected)} rows)"
    )

    # 5b. Circuit DNA — build fingerprints and cluster into archetypes
    logging.info("Building circuit DNA fingerprints...")
    from model.circuit_dna.circuit_dna import run as run_dna
    fp_df, archetypes = run_dna(save=True)
    logging.info(f"Circuit DNA complete: {len(fp_df)} circuits, 4 archetypes")

    # 5c. Train race outcome model
    logging.info("Training race outcome model...")
    from model.race_outcome.race_outcome_model import train as train_race_outcome
    train_race_outcome()

    # 5d. Train qualifying model
    logging.info("Training qualifying model...")
    from model.qualifying_model.qualifying_model import train as train_qualifying
    train_qualifying()

    logging.info("--- PHASE 2 COMPLETE ---")

def validate_multi_season(data_dir):
    f_path = os.path.join(data_dir, 'f1_features_dataset.csv')
    df = pd.read_csv(f_path)
    
    row_count = len(df)
    year_counts = df['race_year'].value_counts()
    unique_circuits = df['circuit'].unique()
    
    # Circuit comparison
    c_2025 = set(df[df['race_year'] == 2025]['circuit'].unique())
    c_prior = set(df[df['race_year'] < 2025]['circuit'].unique())
    
    overlap = c_2025.intersection(c_prior)
    new_circuits = c_2025.difference(c_prior)
    
    logging.info("\n=== 4. Merged Dataset Validation ===")
    logging.info(f"Total row count (Feature Expanded dataset):   {row_count}")
    logging.info(f"\nRows per season:\n{year_counts.to_string()}")
    logging.info(f"\nUnique circuits total: {len(unique_circuits)}")
    logging.info(f"Unique circuits overlapping (in 2025 and prior): {len(overlap)}")
    logging.info(f"Circuits genuinely structurally new to 2025 model (FLAGGED): {len(new_circuits)}")
    if len(new_circuits) > 0:
        logging.info(f"   --> New circuits: {new_circuits}")
        
    logging.info("====================================\n")

if __name__ == "__main__":
    run_pipeline()
