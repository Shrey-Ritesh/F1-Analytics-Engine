import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_driver_performance_metrics(data_path: str, mapping_path: str, output_path: str):
    logging.info(f"Loading dataset from {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Lap level metrics (Lap time mean and std)
    lap_stats = df.groupby('driver_id').agg(
        driver_avg_lap_time=('lap_time_seconds', 'mean'),
        driver_consistency_score=('lap_time_seconds', 'std')
    ).reset_index()
    
    # Fill NaN in consistency (if a driver only has 1 lap) with maximum std to penalize
    max_std = lap_stats['driver_consistency_score'].max()
    lap_stats['driver_consistency_score'] = lap_stats['driver_consistency_score'].fillna(max_std)
    
    # 2. Race level metrics (Win rate, podium rate, avg positions gained)
    # Group by driver_id and race to get unique races
    race_df = df.groupby(['driver_id', 'race_year', 'round_number']).agg(
        final_race_position=('final_race_position', 'first'),
        positions_gained=('positions_gained', 'first')
    ).reset_index()
    
    # Now group by driver_id to get race-level aggregates
    race_stats = race_df.groupby('driver_id').apply(lambda x: pd.Series({
        'total_races': len(x),
        'wins': (x['final_race_position'] == 1).sum(),
        'podiums': (x['final_race_position'] <= 3).sum(),
        'positions_gained_avg': x['positions_gained'].mean()
    })).reset_index()
    
    race_stats['driver_win_rate'] = race_stats['wins'] / race_stats['total_races']
    race_stats['driver_podium_rate'] = race_stats['podiums'] / race_stats['total_races']
    
    # 3. Merge metrics
    metrics_df = pd.merge(lap_stats, race_stats, on='driver_id')
    metrics_df.rename(columns={'driver_id': 'driver'}, inplace=True)
    
    # 4. Decode 'driver' back to string names using the unencoded dataset
    unencoded_path = os.path.join(os.path.dirname(data_path), '..', 'f1_features_dataset.csv')
    if os.path.exists(unencoded_path):
        features_df = pd.read_csv(unencoded_path)
        if 'driver' in features_df.columns and 'driver_id' in features_df.columns:
            driver_mapping = features_df.drop_duplicates(subset=['driver_id']).set_index('driver_id')['driver'].to_dict()
            metrics_df['driver'] = metrics_df['driver'].map(driver_mapping)
    else:
        # If unencoded isn't found, keep it as driver_id number but cast to string
        metrics_df['driver'] = metrics_df['driver'].astype(str)
    
    # 5. Normalization using MinMax scaling
    # We use a feature range of [0.01, 1.0] to strictly avoid division by zero in the specified formula
    scaler = MinMaxScaler(feature_range=(0.01, 1.0))
    
    metrics_df['driver_avg_lap_time_normalized'] = scaler.fit_transform(metrics_df[['driver_avg_lap_time']])
    metrics_df['driver_consistency_score_normalized'] = scaler.fit_transform(metrics_df[['driver_consistency_score']])
    metrics_df['driver_win_rate_normalized'] = scaler.fit_transform(metrics_df[['driver_win_rate']])
    metrics_df['driver_podium_rate_normalized'] = scaler.fit_transform(metrics_df[['driver_podium_rate']])
    
    # 6. Compute driver_performance_score
    metrics_df['driver_performance_score'] = (
        0.35 * (1 / metrics_df['driver_avg_lap_time_normalized']) +
        0.30 * (1 / metrics_df['driver_consistency_score_normalized']) +
        0.20 * metrics_df['driver_podium_rate_normalized'] +
        0.15 * metrics_df['driver_win_rate_normalized']
    )
    
    # 7. Select final columns and sort
    final_cols = [
        'driver', 'driver_avg_lap_time', 'driver_consistency_score', 
        'driver_win_rate', 'driver_podium_rate', 'positions_gained_avg', 
        'driver_performance_score'
    ]
    
    final_df = metrics_df[final_cols].sort_values(by='driver_performance_score', ascending=False)
    
    # 8. Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Driver performance metrics saved to {output_path}")
    
    # 9. Print Top 10
    logging.info("\n--- Top 10 Drivers by Performance Score ---")
    top_10 = final_df.head(10)
    for idx, row in top_10.iterrows():
        logging.info(f"{row['driver']:<25} Score: {row['driver_performance_score']:.2f} | Win Rate: {row['driver_win_rate']:.0%} | Avg Lap: {row['driver_avg_lap_time']:.2f}s")
        

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    output_path = os.path.join(project_root, 'data', 'processed', 'driver_performance_metrics.csv')
    
    build_driver_performance_metrics(data_path, mapping_path, output_path)
