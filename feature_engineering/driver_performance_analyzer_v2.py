import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_driver_performance_metrics_v2(data_path: str, output_path: str):
    logging.info(f"Loading dataset from {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Compute lap_time_delta
    # We define delta as actual lap time minus expected lap time
    df['lap_time_delta'] = df['lap_time_seconds'] - df['expected_lap_time']
    
    # 2. Lap level metrics
    lap_stats = df.groupby('driver_id').agg(
        driver_avg_delta=('lap_time_delta', 'mean'),
        driver_consistency_raw=('lap_time_delta', 'std')
    ).reset_index()
    
    max_std = lap_stats['driver_consistency_raw'].max()
    lap_stats['driver_consistency_raw'] = lap_stats['driver_consistency_raw'].fillna(max_std)
    
    # 3. Race level metrics
    race_df = df.groupby(['driver_id', 'race_year', 'round_number']).agg(
        final_race_position=('final_race_position', 'first')
    ).reset_index()
    
    race_stats = race_df.groupby('driver_id').apply(lambda x: pd.Series({
        'total_races': len(x),
        'wins': (x['final_race_position'] == 1).sum(),
        'podiums': (x['final_race_position'] <= 3).sum()
    })).reset_index()
    
    race_stats['win_rate_raw'] = race_stats['wins'] / race_stats['total_races']
    race_stats['podium_rate_raw'] = race_stats['podiums'] / race_stats['total_races']
    
    # 4. Merge metrics
    metrics_df = pd.merge(lap_stats, race_stats, on='driver_id')
    
    # 5. Add minimum race threshold
    logging.info(f"Drivers before threshold: {len(metrics_df)}")
    metrics_df = metrics_df[metrics_df['total_races'] >= 8].copy()
    logging.info(f"Drivers strictly >= 8 races: {len(metrics_df)}")
    
    # 6. Decode 'driver' back to string names using the unencoded dataset
    unencoded_path = os.path.join(os.path.dirname(data_path), '..', 'f1_features_dataset.csv')
    if os.path.exists(unencoded_path):
        features_df = pd.read_csv(unencoded_path)
        if 'driver' in features_df.columns and 'driver_id' in features_df.columns:
            driver_mapping = features_df.drop_duplicates(subset=['driver_id']).set_index('driver_id')['driver'].to_dict()
            metrics_df['driver'] = metrics_df['driver_id'].map(driver_mapping)
        else:
            metrics_df['driver'] = metrics_df['driver_id'].astype(str)
    else:
        metrics_df['driver'] = metrics_df['driver_id'].astype(str)
    
    # 7. Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    
    # pace_score: Negative delta means faster, thus lower values are better.
    # We negate the delta so that higher values (faster lap delta) become the top of the MinMax scale.
    metrics_df['pace_inverted'] = -metrics_df['driver_avg_delta']
    metrics_df['pace_score'] = scaler.fit_transform(metrics_df[['pace_inverted']])
    
    # consistency_score: Lower standard deviation means more consistent. 
    # We negate the raw STD to invert the scale, then apply MinMax.
    metrics_df['consistency_inverted'] = -metrics_df['driver_consistency_raw']
    metrics_df['consistency_score'] = scaler.fit_transform(metrics_df[['consistency_inverted']])
    
    # win_rate and podium_rate
    metrics_df['podium_rate'] = scaler.fit_transform(metrics_df[['podium_rate_raw']])
    metrics_df['win_rate'] = scaler.fit_transform(metrics_df[['win_rate_raw']])
    
    # 8. Compute driver_performance_score
    metrics_df['driver_performance_score'] = (
        0.40 * metrics_df['pace_score'] +
        0.25 * metrics_df['consistency_score'] +
        0.20 * metrics_df['podium_rate'] +
        0.15 * metrics_df['win_rate']
    )
    
    # 9. Select final columns and sort
    final_cols = [
        'driver', 'driver_performance_score', 'total_races', 'driver_avg_delta',
        'pace_score', 'consistency_score', 'podium_rate', 'win_rate'
    ]
    
    final_df = metrics_df[final_cols].sort_values(by='driver_performance_score', ascending=False)
    
    # 10. Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Corrected v2 metrics saved to {output_path}")
    
    # 11. Print Top 10
    logging.info("\n--- Top 10 Drivers by Performance Score (V2) ---")
    top_10 = final_df.head(10)
    for idx, row in top_10.iterrows():
        logging.info(
            f"{row['driver']:<5} | Score: {row['driver_performance_score']:>5.3f} | "
            f"Avg Delta: {row['driver_avg_delta']:>+6.3f}s | "
            f"Pace: {row['pace_score']:.2f} | Consist: {row['consistency_score']:.2f} | "
            f"Win: {row['win_rate']:.2f} | Podium: {row['podium_rate']:.2f}"
        )

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'driver_performance_metrics_v2.csv')
    
    build_driver_performance_metrics_v2(data_path, output_path)
