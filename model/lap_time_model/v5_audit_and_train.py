import pandas as pd
import numpy as np
import joblib
import os
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_v5_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    model_dir = os.path.join(project_root, 'model', 'lap_time_model')
    
    # -------------------------------------------------------------------------
    # Initial Data Prep & Split
    # -------------------------------------------------------------------------
    df = pd.read_csv(data_path).fillna(0)
    df['compound_interaction'] = df['stint_progress_pct'] * df['compound_base_deg_rate']
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    circuit_mapping = {int(k): v for k, v in mappings.get('circuit', {}).items()}
    tire_mapping = {int(k): v for k, v in mappings.get('tire_compound', {}).items()}
    
    df['circuit_str'] = df['circuit'].map(circuit_mapping)
    df['tire_str'] = df['tire_compound'].map(tire_mapping)
    
    # Sort chronologically
    df = df.sort_values(by=['race_year', 'round_number', 'lap'])
    
    # Match previous split definition
    df['cv_group'] = df['race_year'].astype(str) + "_" + df['round_number'].astype(str)
    unique_groups = df['cv_group'].unique()
    split_idx = int(len(unique_groups) * 0.8)
    train_groups = unique_groups[:split_idx]
    
    test_df = df[~df['cv_group'].isin(train_groups)].copy()
    train_df = df[df['cv_group'].isin(train_groups)].copy()
    
    # -------------------------------------------------------------------------
    # 1. Outlier Lap Audit
    # -------------------------------------------------------------------------
    old_model_path = os.path.join(model_dir, 'lap_time_model.pkl')
    if os.path.exists(old_model_path):
        old_model = joblib.load(old_model_path)
        exclude_cols_v4 = [
            'lap_time_seconds', 'final_race_position', 'podium_finish', 
            'race_winner', 'relative_pace_delta', 'expected_lap_time', 
            'tire_degradation_rate', 'cv_group', 'circuit_str', 'tire_str'
        ]
        
        X_test_v4 = test_df.drop(columns=[c for c in exclude_cols_v4 if c in test_df.columns], errors='ignore')
        
        pred_delta = old_model.predict(X_test_v4)
        pred_abs = pred_delta + test_df['expected_lap_time'].values
        true_abs = test_df['lap_time_seconds'].values
        
        test_df['abs_residual'] = np.abs(true_abs - pred_abs)
        outliers = test_df[test_df['abs_residual'] > 5.0]
        non_outliers = test_df[test_df['abs_residual'] <= 5.0]
        
        logging.info("\n=== 1. Outlier Lap Audit ===")
        logging.info(f"Total outliers (>5s residual): {len(outliers)}")
        logging.info("\nOutlier Tire Distribution:")
        logging.info(outliers['tire_str'].value_counts(normalize=True).to_string())
        logging.info("\nOutlier Pit Stop Flag Distribution (1=Pit Lap):")
        logging.info(outliers['pit_stop'].value_counts(normalize=True).to_string())
        logging.info("\nOutlier Lap 1 Distribution (True=Lap 1):")
        logging.info((outliers['lap'] == 1).value_counts(normalize=True).to_string())
        logging.info(f"\nMean lap time - Outliers: {outliers['lap_time_seconds'].mean():.2f}s")
        logging.info(f"Mean lap time - Non-Outliers: {non_outliers['lap_time_seconds'].mean():.2f}s")
        logging.info("============================\n")
    
    # -------------------------------------------------------------------------
    # 2. Training Set Filter
    # -------------------------------------------------------------------------
    # The 7% threshold for the lap time filter must be computed per-circuit from the TRAINING set only
    train_circuit_medians = train_df.groupby('circuit')['lap_time_seconds'].median()
    global_median = train_df['lap_time_seconds'].median()
    
    def apply_clean_mask(data_df):
        m1 = data_df['pit_stop'] == 0
        m2 = data_df['lap'] > 1
        m3 = ~data_df['tire_str'].isin(['INTERMEDIATE', 'WET'])
        
        # map training medians to data_df
        mapped_medians = data_df['circuit'].map(train_circuit_medians).fillna(global_median)
        m4 = data_df['lap_time_seconds'] < (mapped_medians * 1.07)
        
        return m1 & m2 & m3 & m4
        
    train_clean_mask = apply_clean_mask(train_df)
    test_clean_mask = apply_clean_mask(test_df)
    
    train_df_clean = train_df[train_clean_mask].copy()
    test_df_clean = test_df[test_clean_mask].copy()
    
    logging.info(f"Cleaning dropped {len(train_df) - len(train_df_clean)} training laps and {len(test_df) - len(test_df_clean)} test laps.")
    
    # -------------------------------------------------------------------------
    # 3. Non-leaky Circuit Anchor Feature
    # -------------------------------------------------------------------------
    baseline_mask = (
        (train_df['pit_stop'] == 0) & 
        (train_df['lap'] > 3) & 
        (train_df['tire_age_laps'] >= 3) & 
        (train_df['tire_age_laps'] <= 15) & 
        (train_df['tire_str'].isin(['SOFT', 'MEDIUM', 'HARD']))
    )
    baseline_df = train_df[baseline_mask]
    circuit_baselines = baseline_df.groupby('circuit')['lap_time_seconds'].median().to_dict()
    global_baseline = baseline_df['lap_time_seconds'].median()
    
    circuit_str_baselines = {circuit_mapping.get(k, str(k)): v for k, v in circuit_baselines.items()}
    baseline_json_path = os.path.join(model_dir, 'circuit_baselines.json')
    with open(baseline_json_path, 'w') as f:
        json.dump(circuit_str_baselines, f, indent=4)
    logging.info(f"Saved {len(circuit_str_baselines)} non-leaky circuit baselines to circuit_baselines.json")
    
    # Apply Baseline to DataFrames
    def assign_baseline(data_df):
        data_df['circuit_baseline_pace'] = data_df['circuit'].map(circuit_baselines).fillna(global_baseline)
        data_df['relative_pace_delta_clean'] = data_df['lap_time_seconds'] - data_df['circuit_baseline_pace']
        return data_df
        
    train_df_clean = assign_baseline(train_df_clean)
    test_df_clean = assign_baseline(test_df_clean)
    
    # -------------------------------------------------------------------------
    # 4. Target Variable Decision
    # -------------------------------------------------------------------------
    exclude_cols_clean = [
        'lap_time_seconds', 'final_race_position', 'podium_finish', 
        'race_winner', 'relative_pace_delta', 'expected_lap_time', 
        'tire_degradation_rate', 'cv_group', 'circuit_str', 'tire_str',
        'relative_pace_delta_clean', 'abs_residual'
    ]
    
    X_train_clean = train_df_clean.drop(columns=[c for c in exclude_cols_clean if c in train_df_clean.columns])
    X_test_clean = test_df_clean.drop(columns=[c for c in exclude_cols_clean if c in test_df_clean.columns])
    y_test_true = test_df_clean['lap_time_seconds'].values
    
    # Train Option A (Relative Delta target)
    y_train_A = train_df_clean['relative_pace_delta_clean']
    model_A = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_A.fit(X_train_clean, y_train_A)
    
    pred_delta_A = model_A.predict(X_test_clean)
    pred_abs_A = pred_delta_A + test_df_clean['circuit_baseline_pace'].values
    
    rmse_A = np.sqrt(mean_squared_error(y_test_true, pred_abs_A))
    mae_A = mean_absolute_error(y_test_true, pred_abs_A)
    r2_A = r2_score(y_test_true, pred_abs_A)
    
    # Train Option B (Absolute Target)
    y_train_B = train_df_clean['lap_time_seconds']
    model_B = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_B.fit(X_train_clean, y_train_B)
    
    pred_abs_B = model_B.predict(X_test_clean)
    
    rmse_B = np.sqrt(mean_squared_error(y_test_true, pred_abs_B))
    mae_B = mean_absolute_error(y_test_true, pred_abs_B)
    r2_B = r2_score(y_test_true, pred_abs_B)
    
    logging.info("\n=== 4. Target Variable Decision ===")
    logging.info(f"Option A (Delta target)   | RMSE: {rmse_A:.4f}s | MAE: {mae_A:.4f}s | R2: {r2_A:.4f}")
    logging.info(f"Option B (Absolute target)| RMSE: {rmse_B:.4f}s | MAE: {mae_B:.4f}s | R2: {r2_B:.4f}")
    
    # -------------------------------------------------------------------------
    # 5. Save and Summarize
    # -------------------------------------------------------------------------
    if rmse_A < rmse_B:
        logging.info("--> Option A wins! Predicting true delta + anchor is more robust.")
        final_model_path = os.path.join(model_dir, 'lap_time_model_v5_delta.pkl')
        joblib.dump(model_A, final_model_path)
        best_preds = pred_abs_A
    else:
        logging.info("--> Option B wins! Direct absolute prediction is more robust.")
        final_model_path = os.path.join(model_dir, 'lap_time_model_v5_absolute.pkl')
        joblib.dump(model_B, final_model_path)
        best_preds = pred_abs_B
        
    logging.info(f"Saved best model to {final_model_path}")
    
    # Final Residuals Print
    residuals = y_test_true - best_preds
    abs_residuals = np.abs(residuals)
    
    logging.info("\n=== 5. Final Sub-Sampled Residual Percentiles (Clean Laps) ===")
    logging.info(f"Test Set Size:       {len(test_df_clean)} laps")
    logging.info(f"Mean residual:       {np.mean(residuals):.3f}s")
    logging.info(f"Std of residuals:    {np.std(residuals):.3f}s")
    logging.info(f"5th percentile abs:  {np.percentile(abs_residuals, 5):.3f}s")
    logging.info(f"25th percentile abs: {np.percentile(abs_residuals, 25):.3f}s")
    logging.info(f"50th percentile abs: {np.percentile(abs_residuals, 50):.3f}s")
    logging.info(f"75th percentile abs: {np.percentile(abs_residuals, 75):.3f}s")
    logging.info(f"95th percentile abs: {np.percentile(abs_residuals, 95):.3f}s")
    logging.info(f"Outliers (>5s):      {np.sum(abs_residuals > 5.0)} laps")

if __name__ == "__main__":
    run_v5_pipeline()
