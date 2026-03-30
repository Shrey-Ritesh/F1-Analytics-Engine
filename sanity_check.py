import json
with open('model/strategy_optimizer/circuit_strategy_profiles.json') as f:
    profiles = json.load(f)

for circuit in ['Dutch Grand Prix', 'Canadian Grand Prix']:
    p = profiles[circuit]
    print(f"\n{circuit}")
    print(f"  Sample size:       {p['sample_size']}")
    print(f"  Stop distribution: {p['stop_distribution']}")
    print(f"  Winning strategy:  {p['winning_strategy']}")
