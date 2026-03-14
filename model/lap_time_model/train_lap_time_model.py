import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_lap_time_model(data_path: str, model_dir: str):
    """
    Trains and evaluates an XGBoost model predicting relative_pace_delta.
    """
    logging.info(f"Loading training data from {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Handle NaNs safely
    df = df.fillna(0)
    
    # 3a. Add a compound interaction feature inside the XGBoost pipeline
    # The interaction between how far in the stint we are and the tire degradation base rate
    df['compound_interaction'] = df['stint_progress_pct'] * df['compound_base_deg_rate']
    
    # Define Leaky/Excluded columns
    # We remove these as they provide unfair future glimpses or are part of the target calculation
    exclude_cols = [
        'lap_time_seconds',
        'final_race_position',
        'podium_finish',
        'race_winner',
        'relative_pace_delta',
        'expected_lap_time',
        'tire_degradation_rate' # LEAKY: Uses actual lap_time_seconds
    ]
    
    # Define Target Variables
    target_col = 'relative_pace_delta'
    absolute_target_col = 'lap_time_seconds' # For reconstruction evaluation
    
    # Sort data chronologically mapping
    df = df.sort_values(by=['race_year', 'round_number', 'lap'])
    
    # Validation strategy group definition
    # Combine race_year and round_number to stringently isolate each unique race globally
    df['cv_group'] = df['race_year'].astype(str) + "_" + df['round_number'].astype(str)
    groups = df['cv_group']
    
    # Determine the test splits. We'll simulate a chronological hold-out for true final testing.
    # GroupKFold handles the CV during tuning, but we want the final test on unseen data.
    unique_groups = df['cv_group'].unique()
    split_idx = int(len(unique_groups) * 0.8)
    train_groups = unique_groups[:split_idx]
    
    train_df = df[df['cv_group'].isin(train_groups)]
    test_df = df[~df['cv_group'].isin(train_groups)]
    
    logging.info(f"GroupKFold split prepared. {len(unique_groups)} total races.")
    logging.info(f"Training on {len(train_groups)} races. Held-out final testing on {len(unique_groups) - len(train_groups)} races.")
    
    # Extract Targets
    y_train = train_df[target_col]
    
    # Extract features, ensuring leaky exclusions are dropped completely.
    X_train = train_df.drop(columns=[c for c in exclude_cols if c in train_df.columns] + ['cv_group'], errors='ignore')
    groups_train = train_df['cv_group']
    
    if not XGB_AVAILABLE:
        logging.error("XGBoost is not available.")
        return

    # 3b. Tune XGBoost hyperparameters using GridSearchCV
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [300, 500],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    # n_splits=5 matches general industry practice for cross val grouping
    gkf = GroupKFold(n_splits=5)
    
    logging.info("\n--- Starting GridSearchCV for XGBoost (GroupKFold) ---")
    search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=gkf,
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    best_model = search.best_estimator_
    
    logging.info(f"Best Hyperparameters found: {search.best_params_}")
    
    # Mean CV RMSE
    mean_cv_rmse = -search.best_score_
    cv_std = search.cv_results_['std_test_score'][search.best_index_]
    
    # 4. Final Evaluation on Held-Out Test Set
    X_test = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns] + ['cv_group'], errors='ignore')
    
    # Predict the target (relative pace delta)
    predictions_delta = best_model.predict(X_test)
    
    # Reconstruct the absolute lap times
    # predicted lap time = predicted_delta + expected_lap_time
    expected_laps_test = test_df['expected_lap_time'].values
    predictions_abs = predictions_delta + expected_laps_test
    
    # True absolute lap times
    y_test_abs = test_df[absolute_target_col].values
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_abs, predictions_abs))
    mae = mean_absolute_error(y_test_abs, predictions_abs)
    r2 = r2_score(y_test_abs, predictions_abs)
    
    # 3c. Output a feature importance table sorted by gain
    # importance_type='gain' provides the fractional contribution of each feature to the model 
    booster = best_model.get_booster()
    gain_importances = booster.get_score(importance_type='gain')
    importances_df = pd.DataFrame(list(gain_importances.items()), columns=['Feature', 'Gain'])
    importances_df = importances_df.sort_values(by='Gain', ascending=False)
    
    # Save Feature Importance list
    os.makedirs(model_dir, exist_ok=True)
    fi_export_path = os.path.join(model_dir, 'feature_importance.csv')
    importances_df.to_csv(fi_export_path, index=False)
    
    logging.info("\n=========================================")
    logging.info(f" FINAL XGBOOST PERFORMANCE METRICS")
    logging.info(f" Cross-val RMSE:       {mean_cv_rmse:.4f} ± {cv_std:.4f}")
    logging.info(f" Test RMSE (Absolute): {rmse:.4f} seconds")
    logging.info(f" Test MAE  (Absolute): {mae:.4f} seconds")
    logging.info(f" Test R² Score:        {r2:.4f}")
    logging.info("=========================================\n")
    
    logging.info("--- Top 15 Features by Gain Importance ---")
    top_15 = importances_df.head(15)
    for idx, row in top_15.iterrows():
        logging.info(f"  {row['Feature']:<30} {row['Gain']:.4f}")
        
    # 5. Save the improved model
    model_export_path = os.path.join(model_dir, 'lap_time_model.pkl')
    logging.info(f"\nSaving tuned model to {model_export_path} ...")
    joblib.dump(best_model, model_export_path)
    logging.info("Model saved successfully. Pipeline complete.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    models_dir = os.path.join(project_root, 'model', 'lap_time_model')
    
    train_lap_time_model(dataset_path, models_dir)
