import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_lap_time_model(data_path: str, model_dir: str):
    """
    Trains and evaluates tuned XGBoost model to predict lap_time_delta.
    Evaluates back on reconstructed lap_time_seconds.
    Saves it to models/lap_time_model.pkl
    """
    logging.info(f"Loading training data from {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Feature Engineering inside the script as requested
    logging.info("Engineering new features...")
    df['track_evolution'] = df['lap'] / df['laps_remaining'].replace(0, 1)
    df['dirty_air_flag'] = (df['gap_to_car_ahead_seconds'] < 1.5).astype(int)
    
    # New features for v3/v4
    df['drs_enabled'] = (df['lap'] > 2).astype(int)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    
    import json
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
            
        circuit_dict = mappings.get('circuit', {})
        
        street_circuits = ['Monaco Grand Prix', 'Singapore Grand Prix', 'Azerbaijan Grand Prix', 'Las Vegas Grand Prix', 'Miami Grand Prix', 'Saudi Arabian Grand Prix']
        high_speed_circuits = ['Italian Grand Prix', 'Belgian Grand Prix', 'British Grand Prix', 'Saudi Arabian Grand Prix', 'Qatar Grand Prix']
        technical_circuits = ['Hungarian Grand Prix', 'Spanish Grand Prix', 'Dutch Grand Prix', 'Emilia Romagna Grand Prix', 'Japanese Grand Prix', 'United States Grand Prix']
        power_circuits = ['Canadian Grand Prix', 'Austrian Grand Prix', 'Mexico City Grand Prix', 'Bahrain Grand Prix', 'Abu Dhabi Grand Prix', 'Chinese Grand Prix', 'Australian Grand Prix', 'São Paulo Grand Prix']
        
        def assign_track_type(circuit_name):
            if circuit_name in street_circuits: return 0
            if circuit_name in high_speed_circuits: return 1
            if circuit_name in technical_circuits: return 2
            if circuit_name in power_circuits: return 3
            return 2
            
        code_to_track_type = {}
        for code_str, name in circuit_dict.items():
            code_to_track_type[int(code_str)] = assign_track_type(name)
            
        df['track_type'] = df['circuit'].map(code_to_track_type)
    else:
        df['track_type'] = 0
    
    # V4: TARGET DELTA CREATION
    df['lap_time_delta'] = df['lap_time_seconds'] - df['expected_lap_time']
    
    # Remove Target Leakage
    logging.info("Removing target leakage columns...")
    leakage_cols = ['final_race_position', 'podium_finish', 'race_winner']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')
    
    target_col = 'lap_time_delta'
    absolute_target_col = 'lap_time_seconds'

    # Handle NaNs safely
    df = df.fillna(0)
    
    # Sort data
    df = df.sort_values(by=['race_year', 'round_number', 'lap'])
    
    # Split features and target
    y = df[target_col]
    
    # Drop both lap_time_seconds and lap_time_delta from X
    X = df.drop(columns=[target_col, absolute_target_col])
    groups = df['round_number']
    
    logging.info(f"GroupKFold split applied on {len(df['round_number'].unique())} total rounds.")
    logging.info(f"Total shapes -> X: {X.shape}, y: {y.shape}")
    
    if not XGB_AVAILABLE:
        logging.error("XGBoost is not available.")
        return

    # Hyperparameter tuning using RandomizedSearchCV with GroupKFold
    param_grid = {
        'max_depth': [5, 7, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [200, 300, 500],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    gkf = GroupKFold(n_splits=5)
    
    logging.info("\n--- Starting RandomizedSearchCV for XGBoost (5-fold GroupKFold) ---")
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=gkf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X, y, groups=groups)
    best_model = search.best_estimator_
    
    logging.info(f"Best Hyperparameters found: {search.best_params_}")
    
    mean_cv_rmse = -search.best_score_
    logging.info(f"Best Mean CV RMSE (predicting delta) during tuning: {mean_cv_rmse:.4f} seconds")
    
    # Test Evaluation
    rounds = sorted(df['round_number'].unique())
    split_idx = int(len(rounds) * 0.8)
    train_rounds = rounds[:split_idx]
    
    test_df = df[~df['round_number'].isin(train_rounds)]
    y_test_abs = test_df[absolute_target_col]
    X_test = test_df.drop(columns=[target_col, absolute_target_col])
    
    # Predict deltas
    predictions_delta = best_model.predict(X_test)
    
    # Reconstruct absolute lap times
    predictions_abs = predictions_delta + X_test['expected_lap_time']
    
    rmse = np.sqrt(mean_squared_error(y_test_abs, predictions_abs))
    mae = mean_absolute_error(y_test_abs, predictions_abs)
    r2 = r2_score(y_test_abs, predictions_abs)
    
    logging.info("\n=========================================")
    logging.info(f" FINAL XGBoost PERFORMANCE (on held-out future rounds, reconstructed)")
    logging.info(f" Mean CV RMSE (Delta): {mean_cv_rmse:.4f}")
    logging.info(f" Test RMSE:            {rmse:.4f} seconds")
    logging.info(f" Test MAE:             {mae:.4f} seconds")
    logging.info(f" Test R²:              {r2:.4f}")
    logging.info("=========================================\n")
    
    # Extract Top 20 Feature Importances
    importances = best_model.feature_importances_
    feature_names = X.columns
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(20)
    
    logging.info("Top 20 Feature Importances:")
    for idx, row in importances_df.iterrows():
        logging.info(f"  {row['Feature']:<30} {row['Importance']:.4f}")
        
    # Save Model
    os.makedirs(model_dir, exist_ok=True)
    model_export_path = os.path.join(model_dir, 'lap_time_model.pkl')
    logging.info(f"\nSaving tuned model to {model_export_path} ...")
    joblib.dump(best_model, model_export_path)
    logging.info("Model saved successfully. Pipeline complete.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    models_dir = os.path.join(project_root, 'model', 'lab_time_model')
    
    train_lap_time_model(dataset_path, models_dir)
