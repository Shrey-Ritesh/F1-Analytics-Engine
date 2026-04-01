import numpy as np
import time
import pandas as pd

FEATURE_ORDER = [
    'tire_age_laps', 'tire_cold_flag', 'compound_base_deg_rate',
    'adjusted_tire_stress', 'compound_interaction', 'stint_progress_pct',
    'stint_number', 'stint_length', 'fuel_load_estimate', 'fuel_time_effect',
    'dirty_air_flag', 'laps_remaining_pct', 'position_vs_grid',
    'gap_to_leader_seconds', 'gap_to_car_ahead_seconds', 'track_temperature',
    'circuit_baseline_pace', 'lap', 'round_number', 'grid_position',
    'driver_id', 'team', 'tire_compound'
]

COMPOUND_DEG_RATE = {
    'SOFT': 0.035, 'MEDIUM': 0.022, 'HARD': 0.014
}

# Tire degradation plateau — after this many laps the compound reaches thermal
# equilibrium and lap time stops worsening meaningfully.
# Used to cap tire_age in the feature matrix so the linear model doesn't
# compound degradation beyond realistic levels on long stints.
# Values derived from real F1 stint length distributions (95th percentile).
TIRE_AGE_PLATEAU = {
    'SOFT': 10, 'MEDIUM': 16, 'HARD': 22
}

# Degradation feature scale applied at inference time.
# The XGBoost model was trained on individual lap times where tired tyres
# ARE slower, but real race drivers manage tyres — they don't push to the
# absolute limit on every lap of a long stint. Summing per-lap predictions
# for a 47-lap HARD stint therefore over-states the total time penalty by
# ~3-4x vs reality. Scale factor 0.30 corrects this at inference without
# retraining the model.
DEG_FEATURE_SCALE = 0.30

def build_feature_matrix(strategies, driver_id, team_encoded, circuit, grid_position, 
                         total_laps, circuit_baseline, compound_encoding, track_temperature=35.0):
    
    n_strats = len(strategies)
    row_count = sum(total_laps - s['n_stops'] for s in strategies)
    
    matrix = np.zeros((row_count, len(FEATURE_ORDER)), dtype=np.float32)
    strategy_row_boundaries = np.zeros(n_strats + 1, dtype=np.int32)
    
    current_row = 0
    
    for s_idx, strategy in enumerate(strategies):
        pit_laps = set(strategy['pit_laps'])
        compounds = strategy['compounds']
        
        stint_boundaries = [0] + strategy['pit_laps'] + [total_laps]
        stint_lengths = [stint_boundaries[i+1] - stint_boundaries[i] for i in range(len(stint_boundaries)-1)]
        
        stint_number = 1
        current_stint_idx = 0
        tire_age = 1
        
        for lap in range(1, total_laps + 1):
            if lap in pit_laps:
                stint_number += 1
                current_stint_idx += 1
                tire_age = 1
                continue
            
            comp = compounds[current_stint_idx]
            base_deg = COMPOUND_DEG_RATE[comp]
            exp_len = max(1, stint_lengths[current_stint_idx])

            # Cap tire age at the compound plateau for degradation features.
            # Real tires reach equilibrium — the linear model would otherwise
            # keep predicting worsening lap times indefinitely on long stints,
            # causing a spurious ~88s gap between stop counts at Bahrain.
            plateau    = TIRE_AGE_PLATEAU.get(comp, tire_age)
            capped_age = min(tire_age, plateau)
            # Also cap the stint_length feature so the model doesn't see a
            # 47-lap stint as inherently worse than a 22-lap stint
            capped_len = min(exp_len, plateau)

            prog_pct = capped_age / capped_len

            fuel_load = max(0.0, 110.0 - (lap * 1.7))

            matrix[current_row, 0] = capped_age
            matrix[current_row, 1] = 1.0 if capped_age <= 2 else 0.0
            matrix[current_row, 2] = base_deg * DEG_FEATURE_SCALE
            matrix[current_row, 3] = capped_age * base_deg * DEG_FEATURE_SCALE
            matrix[current_row, 4] = prog_pct * base_deg * DEG_FEATURE_SCALE
            matrix[current_row, 5] = prog_pct
            matrix[current_row, 6] = stint_number
            matrix[current_row, 7] = capped_len
            matrix[current_row, 8] = fuel_load
            matrix[current_row, 9] = fuel_load * 0.03
            matrix[current_row, 10] = 0.0
            matrix[current_row, 11] = (total_laps - lap) / total_laps
            matrix[current_row, 12] = 0.0
            matrix[current_row, 13] = 0.0
            matrix[current_row, 14] = 1.5
            matrix[current_row, 15] = track_temperature
            matrix[current_row, 16] = circuit_baseline
            matrix[current_row, 17] = lap
            matrix[current_row, 18] = 1.0
            matrix[current_row, 19] = grid_position
            matrix[current_row, 20] = driver_id
            matrix[current_row, 21] = team_encoded
            matrix[current_row, 22] = compound_encoding[comp]
            
            current_row += 1
            tire_age += 1
            
        strategy_row_boundaries[s_idx + 1] = current_row
        
    return matrix, strategy_row_boundaries


import pandas as pd

def simulate_all_strategies(strategies, driver_id, team_encoded,
                            circuit, grid_position, total_laps,
                            model, circuit_baselines, pit_loss_estimates,
                            compound_encoding, track_temperature=35.0,
                            circuit_profiles=None):
                            
    t0 = time.time()
    
    circuit_baseline = circuit_baselines.get(circuit, circuit_baselines.get('__global_fallback__', 84.0))
    circuit_pit_loss = pit_loss_estimates.get(circuit, pit_loss_estimates.get('__global_fallback__', 23.1))

    # Traffic cost per pit stop: each stop loses track position, costing time
    # fighting through backmarkers after rejoining. Scaled by overtaking difficulty
    # so Monaco/Singapore penalise extra stops more than Austria/Bahrain.
    # Coefficient 12s chosen so that Bahrain (difficulty 0.70) → ~8.4s/stop,
    # which narrows the 2→3 stop gap from ~88s to a realistic ~15-25s range.
    overtaking_difficulty = 0.70  # neutral fallback
    if circuit_profiles and circuit in circuit_profiles:
        overtaking_difficulty = circuit_profiles[circuit].get('overtaking_difficulty', 0.70)
    traffic_cost_per_stop = overtaking_difficulty * 15.0
    
    matrix, boundaries = build_feature_matrix(
        strategies, driver_id, team_encoded, circuit, grid_position, 
        total_laps, circuit_baseline, compound_encoding, track_temperature
    )
    
    df_matrix = pd.DataFrame(matrix, columns=FEATURE_ORDER)
    preds = model.predict(df_matrix)
    preds += 1.05 
    
    race_times = []
    for i in range(len(strategies)):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        laps_sum   = float(np.sum(preds[start_idx:end_idx]))
        pit_losses = strategies[i]['n_stops'] * circuit_pit_loss
        traffic    = strategies[i]['n_stops'] * traffic_cost_per_stop
        race_times.append(laps_sum + pit_losses + traffic)
        
    t1 = time.time()
    print(f"Simulating {len(strategies)} strategies...")
    print(f"Feature matrix shape: {matrix.shape}")
    print(f"Batch prediction complete in {t1 - t0:.2f}s")
    
    return race_times
