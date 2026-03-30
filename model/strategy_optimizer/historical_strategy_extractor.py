import pandas as pd
import numpy as np
import os
import json
import math

def extract_race_strategies(df):
    strategies = []
    
    # Group logically by race_year, circuit, driver, driver_id
    grouped = df.groupby(['race_year', 'circuit', 'driver', 'driver_id'])
    
    for (year, circuit, driver, driver_id), group in grouped:
        group = group.sort_values('lap')
        pit_laps = group[group['pit_stop'] == 1]['lap'].tolist()
        
        compounds = []
        stint_boundaries = [0] + pit_laps + [group['lap'].max()]
        
        for i in range(len(stint_boundaries) - 1):
            start = stint_boundaries[i]
            end = stint_boundaries[i+1]
            stint_laps = group[(group['lap'] > start) & (group['lap'] <= end)]
            if len(stint_laps) > 0:
                comp = stint_laps['tire_compound'].mode()
                if len(comp) > 0:
                    compounds.append(str(comp.iloc[0]))
                else:
                    compounds.append('UNKNOWN')
            else:
                compounds.append('UNKNOWN')
                
        final_pos = group['final_race_position'].iloc[-1] if 'final_race_position' in group.columns else -1
        
        strategies.append({
            'race_year': year,
            'circuit': circuit,
            'driver': driver,
            'driver_id': driver_id,
            'n_stops': len(pit_laps),
            'pit_laps': pit_laps,
            'compounds': compounds,
            'final_position': final_pos,
            'total_laps': group['lap'].max()
        })
        
    strategies_df = pd.DataFrame(strategies)
    print(f"Total race strategies reconstructed: {len(strategies_df)}")
    print(strategies_df.head(10).to_string())
    return strategies_df

def validate_reconstruction(strategies_df):
    print("\n--- Validation ---")
    zero_stop = strategies_df[strategies_df['n_stops'] == 0]
    print(f"Zero-stop strategies: {len(zero_stop)}")
    print("(These are likely DNFs or data gaps — inspect and flag)")
    
    crazy_stops = strategies_df[(strategies_df['n_stops'] > 3) & (strategies_df['circuit'] != 'Qatar Grand Prix')]
    crazy_stops = pd.concat([crazy_stops, strategies_df[(strategies_df['n_stops'] > 4) & (strategies_df['circuit'] == 'Qatar Grand Prix')]])
    print(f"\nStrategies with >3 stops (excluding Qatar >4): {len(crazy_stops)}")
    print(crazy_stops[['race_year','circuit','driver','n_stops']].to_string())
    print("(>3 stops = likely wet race chaos or data error)")
    
    def check_clean(row):
        if row['n_stops'] == 0:
            return 'dnf'
        # Qatar is allowed up to 4 stops historically
        limit = 4 if row['circuit'] == 'Qatar Grand Prix' else 3
        if row['n_stops'] > limit:
            return 'suspect'
        for c in row['compounds']:
            if pd.isna(c) or c == 'UNKNOWN' or c == 'nan':
                return 'suspect'
        return 'clean'
        
    strategies_df['data_quality'] = strategies_df.apply(check_clean, axis=1)
    suspect_count = len(strategies_df[strategies_df['data_quality'] == 'suspect'])
    dnf_count = len(strategies_df[strategies_df['data_quality'] == 'dnf'])
    print(f"\nStrategies flagged as suspect (>4 stops or unknown compound): {suspect_count}")
    
    print("\nStop count distribution across all races:")
    print(strategies_df['n_stops'].value_counts().sort_index())
    return strategies_df

def build_circuit_strategy_profiles(strategies_df):
    clean_df = strategies_df[strategies_df['data_quality'] == 'clean']
    profiles = {}
    
    for circuit, group in clean_df.groupby('circuit'):
        total_races = len(group)
        if total_races == 0: continue
            
        stops_dist = (group['n_stops'].value_counts() / total_races).to_dict()
        
        pit_windows = {}
        for stops in group['n_stops'].unique():
            subset = group[group['n_stops'] == stops]
            if len(subset) == 0: continue
            
            pit_windows[stops] = {}
            for i in range(stops):
                laps_for_stop_i = [p[i] for p in subset['pit_laps'] if len(p) > i]
                if laps_for_stop_i:
                    pit_windows[stops][f'stop_{i+1}'] = (
                        float(np.mean(laps_for_stop_i)),
                        float(np.std(laps_for_stop_i) or 1.0)
                    )
                    
        comp_seqs = ['->'.join(map(str, c)) for c in group['compounds']]
        top_compounds_counts = pd.Series(comp_seqs).value_counts(normalize=True)
        top_compounds = list(zip(top_compounds_counts.index, top_compounds_counts.values))[:3]
        
        winning_row = group[group['final_position'] == 1]
        winning_strategy = {}
        if not winning_row.empty:
            w_row = winning_row.iloc[0]
            winning_strategy = {
                'n_stops': int(w_row['n_stops']),
                'pit_laps': w_row['pit_laps'],
                'compounds': w_row['compounds']
            }
            
        corr = group['final_position'].corr(group['n_stops'])
        if pd.isna(corr): corr = 0.0
        ot_difficulty = 1.0 - corr
        ot_difficulty = float(max(0.0, min(1.0, ot_difficulty)))
        
        profiles[circuit] = {
            'stop_distribution': {int(k): float(v) for k, v in stops_dist.items()},
            'pit_windows': {int(k): v for k, v in pit_windows.items()},
            'top_compounds': [(str(k), float(v)) for k, v in top_compounds],
            'winning_strategy': winning_strategy,
            'overtaking_difficulty': ot_difficulty,
            'sample_size': int(total_races)
        }
        
    print("\nCircuit                    | Dominant stops | Top compound seq      | OT difficulty")
    print("---------------------------+----------------+-----------------------+--------------")
    for c_name, p in profiles.items():
        dom_stop = max(p['stop_distribution'].items(), key=lambda x: x[1])
        dom_stop_str = f"{dom_stop[0]}-stop ({dom_stop[1]*100:.0f}%)"
        
        top_comp = p['top_compounds'][0][0] if p['top_compounds'] else "N/A"
        if len(top_comp) > 21: top_comp = top_comp[:18] + '...'
            
        print(f"{str(c_name):<26} | {dom_stop_str:<14} | {top_comp:<21} | {p['overtaking_difficulty']:.2f}")
        
    return profiles

def score_strategy_prior(strategy, circuit, circuit_profiles):
    profile = circuit_profiles.get(circuit)
    if not profile:
        return {'prior_score': 0.5, 'stop_score': 0.5, 'pit_window_score': 0.5, 'compound_score': 0.5, 'stop_count_pct': 0.5, 'pit_window_z': []}
        
    n_stops = strategy['n_stops']
    
    stop_dist = {str(k): v for k,v in profile['stop_distribution'].items()}
    p_stops = stop_dist.get(str(n_stops), 0.01)
    
    window_scores = []
    z_scores = []
    
    windows_dict = {str(k): v for k,v in profile['pit_windows'].items()}
    windows = windows_dict.get(str(n_stops), {})
    
    for idx, pit_lap in enumerate(strategy['pit_laps']):
        stop_key = f'stop_{idx+1}'
        if stop_key in windows:
            mean, std = windows[stop_key]
            std = max(1.0, std) 
            z = (pit_lap - mean) / std
            z_scores.append(round(z, 2))
            w_score = math.exp(-0.5 * (z**2))
            window_scores.append(w_score)
        else:
            z_scores.append(0.0)
            window_scores.append(0.01)
            
    if len(window_scores) > 0:
        pit_window_score = sum(window_scores) / len(window_scores)
    else:
        pit_window_score = 1.0 if n_stops == 0 else 0.01
        
    comp_str = '->'.join(strategy['compounds'])
    matched = 0.01
    for seq, freq in profile['top_compounds']:
        if seq == comp_str:
            matched = freq
            break
            
    prior_score = (0.35 * p_stops) + (0.40 * pit_window_score) + (0.25 * matched)
    
    return {
        'prior_score': round(prior_score, 4),
        'stop_score': round(p_stops, 4),
        'pit_window_score': round(pit_window_score, 4),
        'compound_score': round(matched, 4),
        'stop_count_pct': round(p_stops, 4),
        'pit_window_z': z_scores
    }

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'f1_features_dataset.csv')
    df = pd.read_csv(data_path)
    
    print("=== Step 1: Reconstruct Strategies ===")
    s_df = extract_race_strategies(df)
    
    print("\n=== Step 2: Validate Data ===")
    s_df = validate_reconstruction(s_df)
    
    print("\n=== Step 3: Build Circuit Profiles ===")
    profiles = build_circuit_strategy_profiles(s_df)
    
    print("\n=== 5. Validate Scoring Function ===")
    test_a = {'n_stops': 2, 'pit_laps': [16, 34], 'compounds': ['SOFT', 'MEDIUM', 'HARD']}
    res_a = score_strategy_prior(test_a, 'Bahrain Grand Prix', profiles)
    print("\nTest A (Typical 2-stop):", res_a)
    print("PASSED" if res_a['prior_score'] > 0.55 else f"FAILED (Score: {res_a['prior_score']})")
    
    test_b = {'n_stops': 3, 'pit_laps': [5, 11, 17], 'compounds': ['SOFT', 'HARD', 'HARD', 'HARD']}
    res_b = score_strategy_prior(test_b, 'Bahrain Grand Prix', profiles)
    print("\nTest B (Unrealistic 3-stop):", res_b)
    print("PASSED" if res_b['prior_score'] < 0.20 else f"FAILED (Score: {res_b['prior_score']})")
    
    test_c = {'n_stops': 1, 'pit_laps': [30], 'compounds': ['MEDIUM', 'HARD']}
    res_c = score_strategy_prior(test_c, 'Monaco Grand Prix', profiles)
    print("\nTest C (Monaco 1-stop):", res_c)
    print("PASSED" if res_c['prior_score'] > 0.50 else f"FAILED (Score: {res_c['prior_score']})")
    
    print("\n=== Step 6: Save Outputs ===")
    out_csv = os.path.join(project_root, 'data', 'processed', 'historical_strategies.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    s_df.to_csv(out_csv, index=False)
    
    out_json = os.path.join(project_root, 'model', 'strategy_optimizer', 'circuit_strategy_profiles.json')
    with open(out_json, 'w') as f:
        json.dump(profiles, f, indent=4)
        
    print("=== HISTORICAL STRATEGY EXTRACTOR SUMMARY ===")
    print(f"Races processed:              {len(df[['race_year', 'circuit']].drop_duplicates())}")
    print(f"Strategies reconstructed:     {len(s_df)}")
    print(f"Clean strategies:             {len(s_df[s_df['data_quality'] == 'clean'])}")
    print(f"Suspect/DNF flagged:          {len(s_df[s_df['data_quality'] != 'clean'])}")
    print(f"Circuits profiled:            {len(profiles)}")
    print(f"Profiles saved to:            model/strategy_optimizer/circuit_strategy_profiles.json")
