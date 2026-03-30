import pandas as pd
import os

data_path = os.path.join(os.path.dirname(__file__), 'data', 'f1_features_dataset.csv')
df = pd.read_csv(data_path)

pit_data = df[df['pit_stop'] == 1][
    ['race_year', 'circuit', 'driver', 'lap', 
     'tire_compound', 'stint_number']
].sort_values(['circuit', 'race_year', 'driver', 'lap'])

print(pit_data.groupby(['circuit', 'race_year'])['driver'].count().to_string())
