import json
with open('model/strategy_optimizer/circuit_strategy_profiles.json') as f:
    profiles = json.load(f)

for circuit in ['Bahrain Grand Prix', 'Monaco Grand Prix', 'Qatar Grand Prix']:
    p = profiles[circuit]
    print(f"\n{circuit}")
    print(f"  Stop distribution: {p['stop_distribution']}")
    print(f"  Pit windows: {p['pit_windows']}")
    print(f"  Sample size: {p['sample_size']}")
