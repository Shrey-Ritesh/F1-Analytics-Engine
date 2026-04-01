"""
Inspect historical prior scores for key circuits.
Prints stop distribution and pit window analysis.
"""
import json
from pathlib import Path

# Resolve path relative to this file's location
profiles_path = Path(__file__).parent.parent / 'circuit_strategy_profiles.json'

with open(profiles_path) as f:
    profiles = json.load(f)

for circuit in ['Bahrain Grand Prix', 'Monaco Grand Prix', 'Qatar Grand Prix']:
    p = profiles[circuit]
    print(f"\n{circuit}")
    print(f"  Stop distribution: {p['stop_distribution']}")
    print(f"  Pit windows: {p['pit_windows']}")
    print(f"  Sample size: {p['sample_size']}")
