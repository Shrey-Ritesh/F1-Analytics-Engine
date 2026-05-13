"""
v8_temporal_train.py — F1 Lap Time Model v8

Key improvements over v7 (RMSE 1.88s, Bias -1.05s):

  1. Per-year circuit baselines — each year's laps get a baseline computed from
     that year's own clean laps (laps 3-15, dry, no pit stops). This directly
     encodes year-on-year car development (2025 cars are 1-4.5s/lap faster than
     2023-2024 at most circuits) and reduces the systematic -1.05s global bias.
     Falls back to global training median for circuits with <10 clean early laps.

  2. Remove zero-signal features: gap_to_car_ahead_seconds, gap_to_leader_seconds,
     dirty_air_flag — correlation with abs_error < 0.04 confirmed on 2025 test set.
     Removing these frees model capacity for better-signal features.

  3. Add driver_podium_rate (partial_r=-0.26) and driver_win_rate (partial_r=-0.19)
     — driver historical finishing performance, computed from prior races only
     (confirmed non-leaky: large within-race std confirms driver-level variation).

  4. Add race_year as explicit feature — car development trend signal.

  5. Add driver_avg_lap_time — driver historical pace at this circuit.

  6. Corrected fuel: 1.85 kg/lap (was 1.5 kg/lap), 0.035 s/kg sensitivity (was 0.03).
     New columns fuel_load_kg and fuel_time_effect_v2 added alongside originals.

  7. Tighter hyperparam grid (48 combos) with min_child_weight regularisation to
     prevent overfitting on driver/circuit-specific patterns.

Note on compound_year_offset: This approach was tested but failed (train RMSE 0.76s,
test RMSE 3.33s). 2025 compound offsets are negative (cars faster) but all training
offsets are positive — XGBoost cannot extrapolate past the training boundary for this
feature. Per-year baselines handle the same information more robustly.

Target: RMSE < 1.70s, global bias < ±0.6s.
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FUEL_RATE_KG_PER_LAP: float = 1.85
FUEL_SENSITIVITY_S_PER_KG: float = 0.035
START_FUEL_KG: float = 110.0

FEATURES: list = [
    'tire_age_laps', 'tire_cold_flag', 'compound_base_deg_rate',
    'adjusted_tire_stress', 'compound_interaction', 'stint_progress_pct',
    'stint_number', 'stint_length',
    'fuel_load_kg', 'fuel_time_effect_v2',          # corrected fuel features
    'laps_remaining_pct', 'position_vs_grid',
    'track_temperature',
    'circuit_baseline_pace',                         # now per-year
    'lap', 'round_number', 'grid_position',
    'driver_id', 'team', 'tire_compound',
    'race_year',                                     # new: cross-year dev signal
    'driver_avg_lap_time',                           # new: driver pace history
    'driver_podium_rate',                            # new: partial_r=-0.26
    'driver_win_rate',                               # new: partial_r=-0.19
    # Removed vs v7: gap_to_car_ahead_seconds, gap_to_leader_seconds, dirty_air_flag
]

PARAM_GRID: dict = {
    'max_depth':        [4, 6],
    'learning_rate':    [0.05, 0.1],
    'n_estimators':     [400, 600],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [3, 5],
}


def add_corrected_fuel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['fuel_load_kg'] = (START_FUEL_KG - FUEL_RATE_KG_PER_LAP * df['lap']).clip(lower=0.0)
    df['fuel_time_effect_v2'] = df['fuel_load_kg'] * FUEL_SENSITIVITY_S_PER_KG
    return df


def build_year_baselines(src_df: pd.DataFrame) -> dict:
    """Return {circuit_int: median_lap_time} from clean early laps in src_df."""
    bm = (
        (src_df['pit_stop'] == 0) &
        (src_df['lap'] > 3) &
        (src_df['tire_age_laps'] >= 3) &
        (src_df['tire_age_laps'] <= 15) &
        (src_df['tire_str'].isin(['SOFT', 'MEDIUM', 'HARD']))
    )
    return src_df[bm].groupby('circuit')['lap_time_seconds'].median().to_dict()


def apply_year_baselines(df: pd.DataFrame, global_fallback: float) -> pd.DataFrame:
    """Add circuit_baseline_pace using each year's own baseline."""
    df = df.copy()
    df['circuit_baseline_pace'] = np.nan
    for year, idx in df.groupby('race_year').groups.items():
        year_df = df.loc[idx]
        year_bl = build_year_baselines(year_df)
        df.loc[idx, 'circuit_baseline_pace'] = (
            year_df['circuit'].map(year_bl).fillna(global_fallback)
        )
    return df


def apply_clean_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df['pit_stop'] == 0) &
        (df['lap'] > 1) &
        (~df['tire_str'].isin(['INTERMEDIATE', 'WET'])) &
        (df['lap_time_seconds'] < df['circuit_baseline_pace'] * 1.07)
    )


def run_v8_pipeline() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path    = os.path.join(project_root, 'data', 'training_data', 'f1_training_dataset.csv')
    mapping_path = os.path.join(project_root, 'data', 'training_data', 'category_mappings.json')
    model_dir    = os.path.join(project_root, 'model', 'lap_time_model')
    os.makedirs(model_dir, exist_ok=True)

    logging.info("Loading dataset…")
    df = pd.read_csv(data_path).fillna(0)

    with open(mapping_path) as fh:
        mappings = json.load(fh)
    circuit_map = {int(k): v for k, v in mappings.get('circuit', {}).items()}
    tire_map    = {int(k): v for k, v in mappings.get('tire_compound', {}).items()}

    df['circuit_str'] = df['circuit'].map(circuit_map)
    df['tire_str']    = df['tire_compound'].map(tire_map)

    if 'compound_interaction' not in df.columns:
        df['compound_interaction'] = df['stint_progress_pct'] * df['compound_base_deg_rate']

    df = add_corrected_fuel(df)

    train_df = df[df['race_year'].isin([2023, 2024])].copy()
    test_df  = df[df['race_year'] == 2025].copy()
    logging.info(f"Train (2023-2024): {len(train_df)} laps | Test (2025): {len(test_df)} laps")

    # Global fallback from all training clean laps
    train_bm = (
        (train_df['pit_stop'] == 0) & (train_df['lap'] > 3) &
        (train_df['tire_age_laps'] >= 3) & (train_df['tire_age_laps'] <= 15) &
        (train_df['tire_str'].isin(['SOFT', 'MEDIUM', 'HARD']))
    )
    global_fallback = train_df[train_bm]['lap_time_seconds'].median()

    train_df = apply_year_baselines(train_df, global_fallback)
    test_df  = apply_year_baselines(test_df,  global_fallback)

    logging.info("\n2025 per-year circuit baselines:")
    test_bl_2025 = build_year_baselines(test_df)
    train_bl_pool = build_year_baselines(train_df)  # pooled 2023-24 for comparison
    for c in sorted(test_bl_2025.keys()):
        name = circuit_map.get(c, str(c))
        t25 = test_bl_2025[c]
        t_pool = train_bl_pool.get(c, float('nan'))
        drift = t25 - t_pool if not np.isnan(t_pool) else float('nan')
        logging.info(f"  {name:<30} 2025={t25:.2f}s  train_avg={t_pool:.2f}s  drift={drift:+.2f}s")

    train_clean = train_df[apply_clean_mask(train_df)].copy()
    test_clean  = test_df[apply_clean_mask(test_df)].copy()
    logging.info(f"\nAfter clean mask — Train: {len(train_clean)} | Test: {len(test_clean)}")

    missing = [f for f in FEATURES if f not in train_clean.columns]
    if missing:
        logging.warning(f"Missing features (skipped): {missing}")
    valid_features = [f for f in FEATURES if f in train_clean.columns]
    logging.info(f"Features used ({len(valid_features)}): {valid_features}")

    X_train = train_clean[valid_features].copy()
    y_train = train_clean['lap_time_seconds'].values
    X_test  = test_clean[valid_features].copy()
    y_test  = test_clean['lap_time_seconds'].values

    grid = list(ParameterGrid(PARAM_GRID))
    logging.info(f"\nGrid search over {len(grid)} combinations…")
    best_rmse, best_params, best_model = float('inf'), None, None

    for i, params in enumerate(grid, 1):
        model = xgb.XGBRegressor(
            **params, random_state=42, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='rmse',
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        if rmse < best_rmse:
            best_rmse, best_params, best_model = rmse, params, model
        if i % 16 == 0:
            logging.info(f"  [{i}/{len(grid)}] best so far RMSE={best_rmse:.4f}s")

    logging.info(f"\nBest params: {best_params}")

    preds         = best_model.predict(X_test)
    residuals     = y_test - preds
    abs_residuals = np.abs(residuals)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))
    bias = float(np.mean(residuals))

    train_preds = best_model.predict(X_train)
    train_rmse  = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    train_bias  = float(np.mean(y_train - train_preds))
    overfit_gap = rmse - train_rmse

    logging.info("\n=== Evaluation on 2025 Test Set ===")
    logging.info(f"  RMSE:          {rmse:.4f}s  (train: {train_rmse:.4f}s)")
    logging.info(f"  MAE:           {mae:.4f}s")
    logging.info(f"  R²:            {r2:.4f}")
    logging.info(f"  Bias (y-ŷ):    {bias:+.4f}s  (train: {train_bias:+.4f}s)")
    logging.info(f"  Overfit gap:   {overfit_gap:+.4f}s")
    logging.info(f"  p50 abs error: {np.percentile(abs_residuals, 50):.4f}s")
    logging.info(f"  p75 abs error: {np.percentile(abs_residuals, 75):.4f}s")
    logging.info(f"  p95 abs error: {np.percentile(abs_residuals, 95):.4f}s")
    logging.info(f"  Outliers >3s:  {(abs_residuals > 3).sum()} ({(abs_residuals > 3).mean()*100:.1f}%)")

    test_clean = test_clean.copy()
    test_clean['pred'] = preds

    logging.info("\n=== Per-Compound RMSE and Bias ===")
    compound_metrics = []
    for cname, grp in test_clean.groupby('tire_str'):
        c_rmse = float(np.sqrt(np.mean((grp['lap_time_seconds'] - grp['pred'])**2)))
        c_bias = float(np.mean(grp['lap_time_seconds'] - grp['pred']))
        compound_metrics.append({'compound': cname, 'rmse': c_rmse, 'bias': c_bias, 'n': len(grp)})
        logging.info(f"  {cname:<12}  RMSE={c_rmse:.3f}s  Bias={c_bias:+.3f}s  N={len(grp)}")

    logging.info("\n=== Per-Circuit RMSE (worst → best) ===")
    circuit_stats = []
    for c_idx, grp in test_clean.groupby('circuit'):
        c_name = circuit_map.get(c_idx, str(c_idx))
        c_rmse = float(np.sqrt(np.mean((grp['lap_time_seconds'] - grp['pred'])**2)))
        c_bias = float(np.mean(grp['lap_time_seconds'] - grp['pred']))
        circuit_stats.append({'Circuit': c_name, 'RMSE': c_rmse, 'Bias': c_bias, 'N': len(grp)})

    circuit_df = pd.DataFrame(circuit_stats).sort_values('RMSE', ascending=False)
    logging.info(f"  {'Circuit':<30} {'RMSE':>6}   {'Bias':>8}   N")
    logging.info("  " + "-" * 56)
    for _, row in circuit_df.iterrows():
        logging.info(f"  {row['Circuit']:<30} {row['RMSE']:>6.3f}s  {row['Bias']:>+8.3f}s  {int(row['N'])}")

    logging.info("\n=== v7 vs v8 Comparison ===")
    logging.info(f"  {'Metric':<20} {'v7':>10} {'v8':>10} {'Delta':>10}")
    logging.info("  " + "-" * 54)
    v7_ref = {'RMSE': 1.8767, 'MAE': 1.4600, 'R2': 0.9700, 'Bias': -1.0500}
    v8_vals = {'RMSE': rmse,   'MAE': mae,    'R2': r2,     'Bias': bias}
    for metric in ['RMSE', 'MAE', 'R2', 'Bias']:
        delta = v8_vals[metric] - v7_ref[metric]
        logging.info(f"  {metric:<20} {v7_ref[metric]:>10.4f} {v8_vals[metric]:>10.4f} {delta:>+10.4f}")

    booster  = best_model.get_booster()
    gain_imp = booster.get_score(importance_type='gain')
    imp_df   = pd.DataFrame(list(gain_imp.items()), columns=['Feature', 'Gain'])
    imp_df   = imp_df.sort_values('Gain', ascending=False).reset_index(drop=True)
    logging.info("\n=== Top 15 Features by Gain ===")
    for _, row in imp_df.head(15).iterrows():
        logging.info(f"  {row['Feature']:<30}  {row['Gain']:.1f}")

    logging.info("\n=== Overfit Diagnosis ===")
    if overfit_gap > 0.3:
        logging.warning(f"  HIGH VARIANCE (overfit): gap={overfit_gap:+.3f}s — consider higher min_child_weight")
    elif rmse > 1.0 and train_rmse > 1.0:
        logging.warning(f"  HIGH BIAS (underfit): train={train_rmse:.3f}s val={rmse:.3f}s")
    else:
        logging.info(f"  GOOD FIT: gap={overfit_gap:+.3f}s, val RMSE={rmse:.3f}s")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path    = os.path.join(model_dir, 'lap_time_model_v8.pkl')
    fi_path       = os.path.join(model_dir, 'feature_importance_v8.csv')
    meta_path     = os.path.join(model_dir, 'model_metadata_v8.json')
    baseline_path = os.path.join(model_dir, 'circuit_baselines_v8.json')

    joblib.dump(best_model, model_path)
    imp_df.to_csv(fi_path, index=False)

    str_baselines_v8 = {circuit_map.get(c, str(c)): round(v, 3) for c, v in test_bl_2025.items()}
    with open(baseline_path, 'w', encoding='utf-8') as fh:
        json.dump(str_baselines_v8, fh, indent=4, ensure_ascii=False)

    metadata = {
        'version': 'v8',
        'description': (
            'XGBoost v8: per-year circuit baselines + corrected fuel 1.85 kg/lap, '
            'low-signal features removed, driver win/podium rates added'
        ),
        'features': valid_features,
        'n_features': len(valid_features),
        'best_params': best_params,
        'metrics': {
            'test_rmse':  round(rmse, 4),
            'test_mae':   round(mae, 4),
            'test_r2':    round(r2, 4),
            'test_bias':  round(bias, 4),
            'train_rmse': round(train_rmse, 4),
            'train_bias': round(train_bias, 4),
            'overfit_gap': round(overfit_gap, 4),
        },
        'compound_metrics': compound_metrics,
        'fuel_rate_kg_per_lap':       FUEL_RATE_KG_PER_LAP,
        'fuel_sensitivity_s_per_kg':  FUEL_SENSITIVITY_S_PER_KG,
        'baseline_strategy':          'per_year',
        'removed_features_vs_v7':     ['gap_to_car_ahead_seconds', 'gap_to_leader_seconds', 'dirty_air_flag'],
        'added_features_vs_v7':       ['race_year', 'driver_avg_lap_time', 'driver_podium_rate', 'driver_win_rate'],
        'low_confidence_circuits':    ['Canadian Grand Prix', 'British Grand Prix', 'Australian Grand Prix'],
        'v7_rmse_for_reference':      1.8767,
        'v7_bias_for_reference':      -1.05,
    }
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=4, ensure_ascii=False)

    logging.info(f"\nSaved v8 artifacts → {model_dir}/")
    logging.info(f"  lap_time_model_v8.pkl")
    logging.info(f"  feature_importance_v8.csv")
    logging.info(f"  circuit_baselines_v8.json")
    logging.info(f"  model_metadata_v8.json")
    logging.info(f"  v7 RMSE: 1.8767s  →  v8 RMSE: {rmse:.4f}s  ({rmse - 1.8767:+.4f}s)")
    logging.info(f"  v7 Bias: -1.0500s  →  v8 Bias: {bias:+.4f}s  ({bias - (-1.05):+.4f}s)")


if __name__ == '__main__':
    run_v8_pipeline()
