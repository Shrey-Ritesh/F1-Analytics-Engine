"""
Sanity checks on circuit strategy profiles.
Validates sample sizes and winning strategies for key circuits.
"""
import json
from pathlib import Path

# Resolve path relative to this file's location
profiles_path = Path(__file__).parent.parent / 'circuit_strategy_profiles.json'

with open(profiles_path) as f:
    profiles = json.load(f)

for circuit in ['Dutch Grand Prix', 'Canadian Grand Prix']:
    p = profiles[circuit]
    print(f"\n{circuit}")
    print(f"  Sample size:       {p['sample_size']}")
    print(f"  Stop distribution: {p['stop_distribution']}")
    print(f"  Winning strategy:  {p['winning_strategy']}")
