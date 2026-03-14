import pandas as pd
import numpy as np
import joblib
import os
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_v6_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    model_dir = os.path.join(project_root, 'model', 'lap_time_model')
    
    # Load Data
    df = pd.read_csv(data_path).fillna(0)
    df['compound_interaction'] = df['stint_progress_pct'] * df['compound_base_deg_rate']
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    circuit_mapping = {int(k): v for k, v in mappings.get('circuit', {}).items()}
    tire_mapping = {int(k): v for k, v in mappings.get('tire_compound', {}).items()}
    
    df['circuit_str'] = df['circuit'].map(circuit_mapping)
    df['tire_str'] = df['tire_compound'].map(tire_mapping)
    
    logging.info(f"Available race_year values in dataset:\n{df['race_year'].value_counts().to_string()}")
    
    # -------------------------------------------------------------------------
    # 1. Diagnose the fallback problem (v5 recreate)
    # -------------------------------------------------------------------------
    df['cv_group'] = df['race_year'].astype(str) + "_" + df['round_number'].astype(str)
    
    unique_years = df['race_year'].unique()
    
    if len(unique_years) > 1:
        logging.info("Multiple years found! Using Temporal Year-Based Split.")
        train_mask = df['race_year'] < df['race_year'].max()
        test_mask = df['race_year'] == df['race_year'].max()
    else:
        logging.info("Only 1 year found! Using Round-Based Temporal Split.")
        unique_groups = df['cv_group'].unique()
        split_idx = int(len(unique_groups) * 0.8) # Keep original split length
        train_groups = unique_groups[:split_idx]
        train_mask = df['cv_group'].isin(train_groups)
        test_mask = ~df['cv_group'].isin(train_groups)
        
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Outlier filter
    train_circuit_medians = train_df.groupby('circuit')['lap_time_seconds'].median()
    global_median = train_df['lap_time_seconds'].median()
    
    def apply_clean_mask(data_df):
        m1 = data_df['pit_stop'] == 0
        m2 = data_df['lap'] > 1
        m3 = ~data_df['tire_str'].isin(['INTERMEDIATE', 'WET'])
        mapped_medians = data_df['circuit'].map(train_circuit_medians).fillna(global_median)
        m4 = data_df['lap_time_seconds'] < (mapped_medians * 1.07)
        return m1 & m2 & m3 & m4
        
    train_clean_mask = apply_clean_mask(train_df)
    test_clean_mask = apply_clean_mask(test_df)
    
    train_df_clean = train_df[train_clean_mask].copy()
    test_df_clean = test_df[test_clean_mask].copy()
    
    # Recompute v5 Baselines to diagnose
    baseline_mask = (
        (train_df_clean['lap'] > 3) & 
        (train_df_clean['tire_age_laps'] >= 3) & 
        (train_df_clean['tire_age_laps'] <= 15) & 
        (train_df_clean['tire_str'].isin(['SOFT', 'MEDIUM', 'HARD']))
    )
    baseline_df_train = train_df_clean[baseline_mask]
    circuit_baselines = baseline_df_train.groupby('circuit')['lap_time_seconds'].median().to_dict()
    global_baseline = baseline_df_train['lap_time_seconds'].median()
    
    # Diagnostic Output
    logging.info("\n=== 1. Diagnose Fallback Problem ===")
    logging.info(f"Global Baseline Used: {global_baseline:.2f}s")
    for c_idx, grp in test_df_clean.groupby('circuit'):
        c_name = circuit_mapping.get(c_idx, str(c_idx))
        is_known = c_idx in circuit_baselines
        baseline_used = circuit_baselines.get(c_idx, global_baseline)
        actual_mean = grp['lap_time_seconds'].mean()
        error = baseline_used - actual_mean
        logging.info(f"Circuit: {c_name:<25} | Known: {str(is_known):<5} | Baseline: {baseline_used:>6.2f}s | Actual Mean: {actual_mean:>6.2f}s | Error: {error:>+6.2f}s")
        
    # -------------------------------------------------------------------------
    # 3. Retrain with Split
    # -------------------------------------------------------------------------
    
    # Apply baseline mapping
    def assign_baseline(data_df):
        data_df['circuit_baseline_pace'] = data_df['circuit'].map(circuit_baselines).fillna(global_baseline)
        return data_df
        
    train_df_clean = assign_baseline(train_df_clean)
    test_df_clean = assign_baseline(test_df_clean)
    
    exclude_cols = [
        'lap_time_seconds', 'final_race_position', 'podium_finish', 
        'race_winner', 'relative_pace_delta', 'expected_lap_time', 
        'tire_degradation_rate', 'cv_group', 'circuit_str', 'tire_str'
    ]
    
    X_train = train_df_clean.drop(columns=[c for c in exclude_cols if c in train_df_clean.columns])
    y_train = train_df_clean['lap_time_seconds']
    
    X_test = test_df_clean.drop(columns=[c for c in exclude_cols if c in test_df_clean.columns])
    y_test_true = test_df_clean['lap_time_seconds'].values
    
    model_v6 = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    
    logging.info("\n=== 3. Retraining Model v6 ===")
    model_v6.fit(X_train, y_train)
    
    pred_abs = model_v6.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_true, pred_abs))
    mae = mean_absolute_error(y_test_true, pred_abs)
    r2 = r2_score(y_test_true, pred_abs)
    
    residuals = y_test_true - pred_abs
    abs_residuals = np.abs(residuals)
    
    logging.info(f"Test RMSE: {rmse:.4f}s")
    logging.info(f"Test MAE:  {mae:.4f}s")
    logging.info(f"Test R²:   {r2:.4f}")
    
    logging.info("\n=== Residuals ===")
    logging.info(f"50th percentile abs: {np.percentile(abs_residuals, 50):.3f}s")
    logging.info(f"95th percentile abs: {np.percentile(abs_residuals, 95):.3f}s")
    
    # -------------------------------------------------------------------------
    # 5. Save & Summarize
    # -------------------------------------------------------------------------
    model_export_path = os.path.join(model_dir, 'lap_time_model_v6_temporal.pkl')
    joblib.dump(model_v6, model_export_path)
    
    circuit_str_baselines = {circuit_mapping.get(k, str(k)): v for k, v in circuit_baselines.items()}
    baseline_json_path = os.path.join(model_dir, 'circuit_baselines.json')
    with open(baseline_json_path, 'w') as f:
        json.dump(circuit_str_baselines, f, indent=4)
        
    logging.info("\n=== 5. Final Clean Summary ===")
    logging.info("Model version      | Train set        | Test set    | RMSE  | MAE   | R²")
    logging.info("-------------------+------------------+-------------+-------+-------+-----")
    logging.info("v4 (leaky)         | all races        | last 5      | 0.37s | 0.26s | 0.99")
    logging.info("v5 (honest)        | all races        | last 5      | 6.94s | 5.35s | -0.00")
    if len(unique_years) > 1:
        logging.info(f"v6 (temporal fix)  | < {df['race_year'].max()}          | == {df['race_year'].max()}  | {rmse:.2f}s | {mae:.2f}s | {r2:.2f}")
    else:
        logging.info(f"v6 (temporal fix)  | rounds 1-(n-5)   | last 5      | {rmse:.2f}s | {mae:.2f}s | {r2:.2f}")

if __name__ == "__main__":
    run_v6_pipeline()
