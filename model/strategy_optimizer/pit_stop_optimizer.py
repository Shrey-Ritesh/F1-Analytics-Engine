import os
import json
import joblib
import itertools
import time
from pathlib import Path
import numpy as np

from model.strategy_optimizer.race_simulator import simulate_all_strategies
from model.strategy_optimizer.regulations import get_circuit_rules, validate_strategy

from model.strategy_optimizer.historical_strategy_extractor import score_strategy_prior

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model', 'lap_time_model')

def _load_json(filename):
    with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        return json.load(f)

LAP_TIME_MODEL = joblib.load(os.path.join(MODEL_DIR, 'lap_time_model_v7.pkl'))
CIRCUIT_BASELINES = _load_json('circuit_baselines.json')
PIT_LOSS_ESTIMATES = _load_json('pit_loss_estimates.json')
MODEL_METADATA = _load_json('model_metadata.json')

# Empirical stint length bounds from 2023-2025 F1 historical data
# Min = 25th percentile, Max = 95th percentile of observed real stints
# Source: data/processed/historical_strategies.csv
STINT_BOUNDS = {
    'SOFT':   {'min': 9,  'max': 27},
    'MEDIUM': {'min': 13, 'max': 35},
    'HARD':   {'min': 19, 'max': 47},
}

def _check_stint_bounds(pit_laps, compounds, total_laps):
    """
    Checks whether all stints in a strategy fall within
    empirical compound-specific length bounds.
    Returns (True, None) if valid.
    Returns (False, reason_string) if invalid.
    """
    boundaries = [0] + list(pit_laps) + [total_laps]

    for i, compound in enumerate(compounds):
        stint_len = boundaries[i+1] - boundaries[i]

        if compound not in STINT_BOUNDS:
            continue  # skip INTERMEDIATE/WET — no bounds for wet

        bounds = STINT_BOUNDS[compound]

        if stint_len < bounds['min']:
            return False, (
                f"{compound} stint {i+1}: {stint_len} laps "
                f"< minimum {bounds['min']}"
            )
        if stint_len > bounds['max']:
            return False, (
                f"{compound} stint {i+1}: {stint_len} laps "
                f"> maximum {bounds['max']}"
            )

    return True, None

PROFILES_PATH = Path(PROJECT_ROOT) / 'model' / 'strategy_optimizer' / 'circuit_strategy_profiles.json'
with open(PROFILES_PATH) as f:
    circuit_profiles = json.load(f)
print(f"Circuit profiles loaded: {len(circuit_profiles)} circuits")

with open(os.path.join(PROJECT_ROOT, 'data', 'training_data', 'category_mappings.json'), 'r') as f:
    CATEGORY_MAPPINGS = json.load(f)

_tire_map = CATEGORY_MAPPINGS.get('tire_compound', {})
COMPOUND_ENCODING = {val: int(key) for key, val in _tire_map.items() if val in ['SOFT', 'MEDIUM', 'HARD']}
if not COMPOUND_ENCODING:
    COMPOUND_ENCODING = {'HARD': 0, 'MEDIUM': 2, 'SOFT': 3}

def print_circuit_rules(circuit):
    rules = get_circuit_rules(circuit)
    print(f"\n=== Rules: {circuit} ===")
    print(f"  Min stops:        {rules['min_stops']}")
    print(f"  Min compounds:    {rules['min_compounds']}")
    print(f"  Max laps per set: {rules.get('max_laps_per_set', 'None')}")

def enumerate_strategies(total_laps, start_compound, n_stops, circuit, min_stint_laps=5, pit_lap_step=1):
    valid_strategies = []
    pruned_count = 0

    if n_stops >= 3 and pit_lap_step == 1:
        pit_lap_step = 2

    pit_lap_domain = range(min_stint_laps, total_laps - min_stint_laps + 2, pit_lap_step)
    
    for pit_laps in itertools.combinations(pit_lap_domain, n_stops):
        if any(pit_laps[i] - pit_laps[i-1] < min_stint_laps for i in range(1, len(pit_laps))):
            continue
            
        if total_laps - pit_laps[-1] < min_stint_laps:
            continue
            
        compounds_domain = ['SOFT', 'MEDIUM', 'HARD']
        for upcoming in itertools.product(compounds_domain, repeat=n_stops):
            compounds = [start_compound] + list(upcoming)
            
            strategy = {
                'n_stops': n_stops,
                'pit_laps': list(pit_laps),
                'compounds': compounds
            }
            
            valid, reason = _check_stint_bounds(
                strategy['pit_laps'],
                strategy['compounds'],
                total_laps
            )
            if not valid:
                pruned_count += 1
                continue
            
            result = validate_strategy(strategy, circuit, total_laps)
            if not result['valid']:
                continue
                
            valid_strategies.append(strategy)
                
    print(f"{n_stops}-stop: {len(valid_strategies)} valid strategies "
          f"({pruned_count} pruned by stint bounds)")
    return valid_strategies, pruned_count

def optimize_strategy(driver_id, team_encoded, circuit, grid_position, total_laps, 
                      start_compound='SOFT', n_stops_range=(1, 2, 3), track_temperature=35.0):
                      
    rules = get_circuit_rules(circuit)
    print_circuit_rules(circuit)
    
    effective_range = tuple(n for n in n_stops_range if n >= rules['min_stops'])
    if effective_range != n_stops_range:
        print(f"WARNING: Stop range adjusted: {n_stops_range} -> {effective_range} ({circuit} regulation)")
    n_stops_range = effective_range
    
    all_strategies = []
    total_pruned = 0
    for n_stops in n_stops_range:
        strats, pruned = enumerate_strategies(total_laps, start_compound, n_stops, circuit, min_stint_laps=5)
        all_strategies.extend(strats)
        total_pruned += pruned

    total_before = len(all_strategies) + total_pruned
    print(f"Total strategies to evaluate: {len(all_strategies)} "
          f"({total_pruned} pruned of {total_before})")
    
    race_times = simulate_all_strategies(
        all_strategies, driver_id, team_encoded, circuit, grid_position, total_laps,
        LAP_TIME_MODEL, CIRCUIT_BASELINES, PIT_LOSS_ESTIMATES, COMPOUND_ENCODING,
        track_temperature, circuit_profiles=circuit_profiles
    )
    
    for i, strategy in enumerate(all_strategies):
        strategy['total_race_time'] = race_times[i]

        if circuit in circuit_profiles:
            prior_breakdown = score_strategy_prior(
                strategy, circuit, circuit_profiles
            )
            strategy['prior_score']       = prior_breakdown['prior_score']
            strategy['stop_score']        = prior_breakdown['stop_score']
            strategy['pit_window_score']  = prior_breakdown['pit_window_score']
            strategy['compound_score']    = prior_breakdown['compound_score']
            strategy['pit_window_z']      = prior_breakdown['pit_window_z']
        else:
            strategy['prior_score']       = 0.5
            strategy['stop_score']        = 0.5
            strategy['pit_window_score']  = 0.5
            strategy['compound_score']    = 0.5
            strategy['pit_window_z']      = []
            
    times = np.array([s['total_race_time'] for s in all_strategies])
    time_min = times.min()
    time_max = times.max()

    for s in all_strategies:
        s['time_score'] = round(
            1.0 - (s['total_race_time'] - time_min) / (time_max - time_min + 1e-9),
            4
        )
        s['combined_score'] = round(
            (s['time_score'] * 0.50) + (s['prior_score'] * 0.50),
            4
        )

    physics_ranked = sorted(
        all_strategies, key=lambda x: x['total_race_time']
    )

    combined_ranked = sorted(
        all_strategies, key=lambda x: x['combined_score'], reverse=True
    )

    best_physics_time = physics_ranked[0]['total_race_time']
    for s in physics_ranked:
        s['delta_to_best_physics'] = round(
            s['total_race_time'] - best_physics_time, 2
        )

    for i, s in enumerate(combined_ranked):
        s['combined_rank'] = i + 1

    return {
        'physics_ranked': physics_ranked,
        'combined_ranked': combined_ranked,
        'circuit_profile': circuit_profiles.get(circuit, {}),
        'total_evaluated': len(all_strategies),
        'total_pruned': total_pruned,
        'total_before': total_before,
    }

def print_strategy_report(results, circuit, total_laps, top_n=10):
    physics_ranked  = results['physics_ranked']
    combined_ranked = results['combined_ranked']
    profile         = results['circuit_profile']

    print(f"\n{'='*65}")
    print(f"  PIT STOP STRATEGY REPORT")
    print(f"  Circuit:    {circuit}")
    print(f"  Race laps:  {total_laps}")
    print(f"  Strategies: {results['total_evaluated']} evaluated")
    print(f"{'='*65}")

    print(f"\n--- PHYSICS RANKING (pure predicted race time) ---")
    print(f"{'Rank':<5} {'Stops':<6} {'Pit laps':<16} "
          f"{'Compounds':<28} {'Race time':<11} {'Delta'}")
    print("-" * 75)
    for i, s in enumerate(physics_ranked[:top_n]):
        compounds_str = ' -> '.join(s['compounds'])
        pit_str = str(s['pit_laps'])
        print(f"{i+1:<5} {s['n_stops']:<6} {pit_str:<16} "
              f"{compounds_str:<28} {s['total_race_time']:.2f}s"
              f"   +{s['delta_to_best_physics']:.2f}s")

    print(f"\n--- COMBINED RANKING (physics 50% + historical prior 50%) ---")
    print(f"{'Rank':<5} {'Stops':<6} {'Pit laps':<16} "
          f"{'Compounds':<28} {'Score':<8} {'Race time':<11} {'Prior'}")
    print("-" * 85)
    for i, s in enumerate(combined_ranked[:top_n]):
        compounds_str = ' -> '.join(s['compounds'])
        pit_str = str(s['pit_laps'])
        print(f"{i+1:<5} {s['n_stops']:<6} {pit_str:<16} "
              f"{compounds_str:<28} {s['combined_score']:<8.4f} "
              f"{s['total_race_time']:.2f}s   {s['prior_score']:.3f}")

    print(f"\n--- BEST STRATEGY PER STOP COUNT (combined ranking) ---")
    for n in sorted(set(s['n_stops'] for s in combined_ranked)):
        best = next(s for s in combined_ranked if s['n_stops'] == n)
        compounds_str = ' -> '.join(best['compounds'])
        print(f"  {n}-stop: pit laps {str(best['pit_laps']):<16} "
              f"compounds: {compounds_str:<28} "
              f"score: {best['combined_score']:.4f}  "
              f"time: {best['total_race_time']:.2f}s")

    stop_bests = {}
    for s in combined_ranked:
        if s['n_stops'] not in stop_bests:
            stop_bests[s['n_stops']] = s['total_race_time']
    stop_counts = sorted(stop_bests.keys())
    print()
    for i in range(len(stop_counts) - 1):
        a, b = stop_counts[i], stop_counts[i+1]
        delta = stop_bests[a] - stop_bests[b]
        print(f"  {a}-stop vs {b}-stop delta: "
              f"{delta:+.2f}s (positive = {b}-stop is faster)")

    if profile:
        print(f"\n--- HISTORICAL PROFILE: {circuit} ---")
        dist = profile.get('stop_distribution', {})
        dominant = max(dist, key=dist.get)
        print(f"  Dominant strategy:     "
              f"{dominant}-stop ({float(dist[dominant])*100:.0f}% "
              f"of historical races)")
        windows = profile.get('pit_windows', {})
        if str(dominant) in windows:
            w = windows[str(dominant)]
            for stop_key, (mean, std) in w.items():
                print(f"  Historical {stop_key}:    "
                      f"lap {mean:.1f} ± {std:.1f}")
        top_c = profile.get('top_compounds', [])
        if top_c:
            print(f"  Top compound sequence: {top_c[0][0]}")
        ot = profile.get('overtaking_difficulty', None)
        if ot is not None:
            label = ('high' if ot > 0.85
                     else 'moderate' if ot > 0.65
                     else 'low')
            
            ot_text = ('track position critical' if ot > 0.85 else 'track position matters' if ot > 0.65 else 'overtaking feasible')
            print(f"  Overtaking difficulty: "
                  f"{ot:.2f} ({label} — {ot_text})")
        winning = profile.get('winning_strategy', {})
        if winning:
            w_compounds = ' -> '.join(winning.get('compounds', []))
            print(f"  Winning strategy ref:  "
                  f"{winning.get('n_stops')}-stop | "
                  f"laps {winning.get('pit_laps')} | "
                  f"{w_compounds}")

    metadata_path = Path(MODEL_DIR) / 'model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        low_conf = metadata.get('low_confidence_circuits', [])
        if circuit in low_conf:
            print(f"\n  WARNING: {circuit} is flagged low confidence.")
            print(f"    Absolute times may vary +/- 4.0s.")
            print(f"    Strategy deltas remain reliable.")

def assess_realism(res, circuit, total_laps):
    p_rank = res['physics_ranked'][0]
    c_rank = res['combined_ranked'][0]

    total_before = res.get('total_before', res['total_evaluated'])
    total_after  = res['total_evaluated']
    total_pruned = res.get('total_pruned', 0)
    pruning_rate = (total_pruned / total_before * 100) if total_before > 0 else 0

    print("\n  === REALISM ASSESSMENT ===")
    print(f"  Strategies before bounds filter: {total_before}")
    print(f"  Strategies after bounds filter:  {total_after}")
    print(f"  Pruning rate:                    {pruning_rate:.0f}%")
    print(f"\n  Physics rank #1:  {p_rank['pit_laps']} {p_rank['compounds']}")
    print(f"  Combined rank #1: {c_rank['pit_laps']} {c_rank['compounds']}")

    checks = []
    boundaries = [0] + c_rank['pit_laps'] + [total_laps]
    stint_lens = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]

    if circuit == 'Bahrain Grand Prix':
        first_pit = c_rank['pit_laps'][0]
        checks.append(('Bahrain combined #1 first pit between lap 9-20',  9 <= first_pit <= 20))
        checks.append(('Bahrain combined #1 no stint under 9 laps',        min(stint_lens) >= 9))

    elif circuit == 'Monaco Grand Prix':
        first_pit = c_rank['pit_laps'][0]
        checks.append(('Monaco  combined #1 first pit before lap 40',      first_pit <= 40))
        hard_stints = [stint_lens[i] for i, c in enumerate(c_rank['compounds']) if c == 'HARD']
        checks.append(('Monaco  combined #1 no HARD stint under 19 laps',
                        all(s >= 19 for s in hard_stints) if hard_stints else True))

    elif circuit == 'Qatar Grand Prix':
        first_pit = c_rank['pit_laps'][0]
        checks.append(('Qatar   combined #1 first pit between lap 12-28',  12 <= first_pit <= 28))
        checks.append(('Qatar   combined #1 no stint over 25 laps',         max(stint_lens) <= 25))

    print("\n  Checks:")
    passed = 0
    for label, result in checks:
        mark   = 'x' if result else ' '
        status = 'PASS' if result else 'FAIL'
        print(f"  [{mark}] {label:<50} {status}")
        if result:
            passed += 1

    print(f"\n  Overall: {passed}/{len(checks)} checks passed")
    if passed < len(checks):
        print('  FAILED CHECKS — tighten SOFT min from 9 to 12 and rerun')

if __name__ == '__main__':
    print("\n--- SCENARIO A: Bahrain Grand Prix ---")
    t0 = time.time()
    res_a = optimize_strategy(
        driver_id=1, team_encoded=3, circuit='Bahrain Grand Prix',
        grid_position=1, total_laps=57, start_compound='SOFT'
    )
    if res_a:
        print_strategy_report(res_a, 'Bahrain Grand Prix', 57)
        assess_realism(res_a, 'Bahrain Grand Prix', 57)
            
    tot_a = time.time() - t0
    print(f"\nScenario A total optimization wall time: {tot_a:.2f}s")
    
    print("\n--- SCENARIO B: Monaco Grand Prix ---")
    t1 = time.time()
    res_b = optimize_strategy(
        driver_id=1, team_encoded=3, circuit='Monaco Grand Prix',
        grid_position=3, total_laps=78, start_compound='MEDIUM'
    )
    if res_b:
        print_strategy_report(res_b, 'Monaco Grand Prix', 78)
        assess_realism(res_b, 'Monaco Grand Prix', 78)
    tot_b = time.time() - t1
    print(f"\nScenario B total optimization wall time: {tot_b:.2f}s")
    
    print("\n--- SCENARIO C: Qatar Grand Prix ---")
    t2 = time.time()
    res_c = optimize_strategy(
        driver_id=1, team_encoded=3, circuit='Qatar Grand Prix',
        grid_position=3, total_laps=57, start_compound='MEDIUM'
    )
    if res_c:
        print_strategy_report(res_c, 'Qatar Grand Prix', 57)
        assess_realism(res_c, 'Qatar Grand Prix', 57)
    tot_c = time.time() - t2
    print(f"\nScenario C total optimization wall time: {tot_c:.2f}s")
