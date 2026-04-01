"""
Inspect pit stop data from the feature dataset.
Prints count of pit events by circuit and year.
"""
import pandas as pd
import os
from pathlib import Path

# Resolve path relative to this file's location
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'f1_features_dataset.csv'

df = pd.read_csv(data_path)

pit_data = df[df['pit_stop'] == 1][
    ['race_year', 'circuit', 'driver', 'lap',
     'tire_compound', 'stint_number']
].sort_values(['circuit', 'race_year', 'driver', 'lap'])

print(pit_data.groupby(['circuit', 'race_year'])['driver'].count().to_string())
