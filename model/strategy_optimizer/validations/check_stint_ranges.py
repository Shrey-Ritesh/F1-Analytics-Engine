"""
Analyze stint length distribution from historical strategies.
Computes percentiles and breakdowns by compound and stint position.
"""
import pandas as pd
import ast
from pathlib import Path

# Resolve path relative to this file's location
strategies_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'historical_strategies.csv'

strategies = pd.read_csv(strategies_path)

records = []
for _, row in strategies[strategies['data_quality'] == 'clean'].iterrows():
    pit_laps = ast.literal_eval(row['pit_laps'])
    compounds = ast.literal_eval(row['compounds'])
    total_laps = row['total_laps']

    # Build stint boundaries
    boundaries = [0] + pit_laps + [total_laps]

    for i, compound in enumerate(compounds):
        stint_len = boundaries[i+1] - boundaries[i]
        records.append({
            'circuit': row['circuit'],
            'compound': compound,
            'stint_length': stint_len,
            'stint_number': i + 1,
            'n_stops': row['n_stops']
        })

stints_df = pd.DataFrame(records)

# Filter out wet compounds
stints_df = stints_df[
    stints_df['compound'].isin(['SOFT', 'MEDIUM', 'HARD'])
]

print("=== Stint length stats by compound ===")
print(stints_df.groupby('compound')['stint_length'].describe().round(1))

print("\n=== Stint length percentiles by compound ===")
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    data = stints_df[stints_df['compound'] == compound]['stint_length']
    print(f"\n{compound}:")
    print(f"  5th pct:  {data.quantile(0.05):.0f} laps")
    print(f"  25th pct: {data.quantile(0.25):.0f} laps")
    print(f"  median:   {data.quantile(0.50):.0f} laps")
    print(f"  75th pct: {data.quantile(0.75):.0f} laps")
    print(f"  95th pct: {data.quantile(0.95):.0f} laps")

print("\n=== Opening stint lengths by compound ===")
opening = stints_df[stints_df['stint_number'] == 1]
print(opening.groupby('compound')['stint_length'].describe().round(1))

print("\n=== Final stint lengths by compound ===")
# Final stint = last compound index
# Approximate: stints where stint_number = n_stops + 1
final = stints_df[
    stints_df['stint_number'] == stints_df['n_stops'] + 1
]
print(final.groupby('compound')['stint_length'].describe().round(1))
