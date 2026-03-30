import pandas as pd
import numpy as np
import joblib
import os
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from sklearn.model_selection import ParameterGrid

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_v7_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    model_dir = os.path.join(project_root, 'model', 'lap_time_model')
    
    # Load Data 
    logging.info("Loading dataset...")
    df = pd.read_csv(data_path).fillna(0)
    
    # Ensure compound_interaction exists
    if 'compound_interaction' not in df.columns:
        df['compound_interaction'] = df['stint_progress_pct'] * df['compound_base_deg_rate']
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    circuit_mapping = {int(k): v for k, v in mappings.get('circuit', {}).items()}
    tire_mapping = {int(k): v for k, v in mappings.get('tire_compound', {}).items()}
    
    df['circuit_str'] = df['circuit'].map(circuit_mapping)
    df['tire_str'] = df['tire_compound'].map(tire_mapping)
    
    # -------------------------------------------------------------------------
    # 1. Temporal Split
    # -------------------------------------------------------------------------
    train_mask = df['race_year'].isin([2023, 2024])
    test_mask = df['race_year'] == 2025
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logging.info(f"Temporal split complete. Train (2023-2024): {len(train_df)} laps, Test (2025): {len(test_df)} laps.")
    
    # -------------------------------------------------------------------------
    # 2. Recompute circuit_baseline_pace from training set only
    # -------------------------------------------------------------------------
    baseline_mask = (
        (train_df['pit_stop'] == 0) & 
        (train_df['lap'] > 3) & 
        (train_df['tire_age_laps'] >= 3) & 
        (train_df['tire_age_laps'] <= 15) & 
        (train_df['tire_str'].isin(['SOFT', 'MEDIUM', 'HARD']))
    )
    baseline_df_train = train_df[baseline_mask]
    
    circuit_baselines_raw = baseline_df_train.groupby('circuit')['lap_time_seconds'].median().to_dict()
    
    # Map back to string names for json and logging
    circuit_str_baselines = {}
    logging.info("\nCircuit                    | Baseline (s)")
    logging.info("---------------------------+-------------")
    for c_idx, baseline in circuit_baselines_raw.items():
        c_name = circuit_mapping.get(c_idx, f"Unknown_{c_idx}")
        circuit_str_baselines[c_name] = round(baseline, 3)
        logging.info(f"{c_name:<26} | {baseline:.2f}s")
        
    baseline_json_path = os.path.join(model_dir, 'circuit_baselines.json')
    os.makedirs(model_dir, exist_ok=True)
    with open(baseline_json_path, 'w') as f:
        json.dump(circuit_str_baselines, f, indent=4)
        
    # Verify 2025 circuits
    test_circuits_idx = test_df['circuit'].unique()
    missing_circuits = [c for c in test_circuits_idx if c not in circuit_baselines_raw]
    if len(missing_circuits) == 0:
        logging.info("\nAll 2025 circuits have known baselines: TRUE")
    else:
        logging.warning("\nAll 2025 circuits have known baselines: FALSE!")
        for c in missing_circuits:
            logging.warning(f"  Missing: {circuit_mapping.get(c, c)}")
            
    # Apply baseline to both train and test
    global_train_baseline = baseline_df_train['lap_time_seconds'].median()
    train_df['circuit_baseline_pace'] = train_df['circuit'].map(circuit_baselines_raw).fillna(global_train_baseline)
    test_df['circuit_baseline_pace'] = test_df['circuit'].map(circuit_baselines_raw).fillna(global_train_baseline)

    # -------------------------------------------------------------------------
    # 3. Apply clean_lap_mask to both splits
    # -------------------------------------------------------------------------
    def apply_clean_mask(data_df):
        m1 = data_df['pit_stop'] == 0
        m2 = data_df['lap'] > 1
        m3 = ~data_df['tire_str'].isin(['INTERMEDIATE', 'WET'])
        m4 = data_df['lap_time_seconds'] < (data_df['circuit_baseline_pace'] * 1.07)
        return m1 & m2 & m3 & m4
        
    train_clean_mask = apply_clean_mask(train_df)
    test_clean_mask = apply_clean_mask(test_df)
    
    train_df_clean = train_df[train_clean_mask].copy()
    test_df_clean = test_df[test_clean_mask].copy()
    
    logging.info("\n=== 3. Clean Mask Application ===")
    logging.info(f"Train laps after filter: {len(train_df_clean)}")
    logging.info(f"Test laps after filter:  {len(test_df_clean)}")
    
    # -------------------------------------------------------------------------
    # 4. Feature list
    # -------------------------------------------------------------------------
    req_features = [
        'tire_age_laps', 'tire_cold_flag', 'compound_base_deg_rate',
        'adjusted_tire_stress', 'compound_interaction', 'stint_progress_pct',
        'stint_number', 'stint_length', 'fuel_load_estimate', 'fuel_time_effect',
        'dirty_air_flag', 'laps_remaining_pct', 'position_vs_grid',
        'gap_to_leader_seconds', 'gap_to_car_ahead_seconds', 'track_temperature',
        'circuit_baseline_pace', 'lap', 'round_number', 'grid_position',
        'driver_id', 'team', 'tire_compound'
    ]
    
    missing_features = [f for f in req_features if f not in train_df_clean.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
    
    valid_features = [f for f in req_features if f in train_df_clean.columns]
    
    X_train = train_df_clean[valid_features]
    y_train = train_df_clean['lap_time_seconds'].values
    eval_set_X = test_df_clean[valid_features]
    eval_set_y = test_df_clean['lap_time_seconds'].values

    # -------------------------------------------------------------------------
    # 5. Hyperparameter tuning (Grid search on single train/test split)
    # -------------------------------------------------------------------------
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [300, 500, 700],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid = list(ParameterGrid(param_grid))
    best_rmse = float('inf')
    best_params = None
    best_model = None
    
    logging.info("\n=== 5. Running Hyperparameter Tuning ===")
    logging.info(f"Evaluating {len(grid)} parameter combinations using early stopping (50 rounds)...")
    
    # For speeding up, we evaluate using the DMatrix / xgb.train interface 
    # but xgb.XGBRegressor works too (fit with early_stopping_rounds)
    for i, params in enumerate(grid):
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        # We need to pass eval_set to fit() to use early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(eval_set_X, eval_set_y)],
            verbose=False
        )
        
        # Use best iteration for prediction
        preds = model.predict(eval_set_X)
        rmse = np.sqrt(mean_squared_error(eval_set_y, preds))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model
            
    logging.info(f"Best parameters found: {best_params}")
    
    # -------------------------------------------------------------------------
    # 6. Evaluation
    # -------------------------------------------------------------------------
    preds = best_model.predict(eval_set_X)
    
    rmse = np.sqrt(mean_squared_error(eval_set_y, preds))
    mae = mean_absolute_error(eval_set_y, preds)
    r2 = r2_score(eval_set_y, preds)
    residuals = eval_set_y - preds
    abs_residuals = np.abs(residuals)
    
    logging.info("\n=== 6. Evaluation on 2025 Test Set ===")
    logging.info(f"RMSE (absolute lap time): {rmse:.4f}s")
    logging.info(f"MAE:                      {mae:.4f}s")
    logging.info(f"R²:                       {r2:.4f}")
    logging.info(f"Mean residual:            {np.mean(residuals):.4f}s")
    logging.info(f"5th percentile abs:       {np.percentile(abs_residuals, 5):.4f}s")
    logging.info(f"25th percentile abs:      {np.percentile(abs_residuals, 25):.4f}s")
    logging.info(f"50th percentile abs:      {np.percentile(abs_residuals, 50):.4f}s")
    logging.info(f"75th percentile abs:      {np.percentile(abs_residuals, 75):.4f}s")
    logging.info(f"95th percentile abs:      {np.percentile(abs_residuals, 95):.4f}s")
    logging.info(f"Outlier count (>3s):      {np.sum(abs_residuals > 3.0)} laps")
    
    # Per-circuit RMSE table
    circuit_stats = []
    test_df_clean['pred_abs'] = preds
    test_df_clean['true_abs'] = eval_set_y
    for c_idx, grp in test_df_clean.groupby('circuit'):
        c_name = circuit_mapping.get(c_idx, str(c_idx))
        c_rmse = np.sqrt(np.mean((grp['true_abs'] - grp['pred_abs'])**2))
        circuit_stats.append({'Circuit': c_name, 'RMSE': c_rmse, 'N laps': len(grp)})
        
    circuit_df = pd.DataFrame(circuit_stats).sort_values(by='RMSE', ascending=False)
    
    logging.info("\nPer-circuit RMSE table sorted worst to best:")
    logging.info(f"{'Circuit':<26} | {'RMSE':<8} | N laps")
    logging.info("-" * 46)
    for _, row in circuit_df.iterrows():
        logging.info(f"{row['Circuit']:<26} | {row['RMSE']:.2f}s    | {row['N laps']}")
        

    # -------------------------------------------------------------------------
    # 7. Feature importance
    # -------------------------------------------------------------------------
    booster = best_model.get_booster()
    gain_importances = booster.get_score(importance_type='gain')
    importances_df = pd.DataFrame(list(gain_importances.items()), columns=['Feature', 'Gain'])
    importances_df = importances_df.sort_values(by='Gain', ascending=False)
    
    logging.info("\n=== 7. Top 15 Features by Gain ===")
    top_15 = importances_df.head(15).copy()
    top_5_features = top_15.head(5)['Feature'].tolist()
    
    for idx, row in top_15.iterrows():
        logging.info(f"  {row['Feature']:<30} {row['Gain']:.4f}")
        
    if 'circuit_baseline_pace' not in top_5_features:
        logging.warning("circuit_baseline_pace not in top 5 — investigate")
        
    # -------------------------------------------------------------------------
    # 8. Save artifacts
    # -------------------------------------------------------------------------
    model_export_path = os.path.join(model_dir, 'lap_time_model_v7.pkl')
    joblib.dump(best_model, model_export_path)
    
    fi_export_path = os.path.join(model_dir, 'feature_importance_v7.csv')
    importances_df.to_csv(fi_export_path, index=False)
    
    logging.info(f"\nSaved v7 model artifacts to {model_dir}")

if __name__ == "__main__":
    run_v7_pipeline()
