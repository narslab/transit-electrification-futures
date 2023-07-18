import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

df = pd.read_csv(r'../../results/trips-mapped-into-blocks.csv', low_memory=False)

# Ensure Start_time and End_time are datetime objects
df['Start_time'] = pd.to_datetime(df['Start_time'])
df['End_time'] = pd.to_datetime(df['End_time'])

# Sort DataFrame
df.sort_values(by=['Start_time', 'End_time'], inplace=True)

# Reset index after sorting
df.reset_index(drop=True, inplace=True)

# Initialize an empty array for block_ids
df['block_id'] = np.nan

# Initialize block_id
block_id = 1

# Initialize the progress bar
pbar = tqdm(total=len(df))

# Iterate over DataFrame grouped by 'Route'
for route, group in df.groupby('Route'):
    group.sort_values(by=['Start_time', 'End_time'], inplace=True)

    # Iterate over rows in group
    i = 0
    while i < len(group):
        # Start a bundle
        bundle = [i]
        
        # Try adding trips to the bundle
        while len(bundle) < 10:
            # Find the next trip that starts within 0.6 minutes after the last one in the bundle ends
            next_trip = group[(group['Start_time'] - group.iloc[bundle[-1]]['End_time'] <= timedelta(minutes=0.6)) & 
                              (group['block_id'].isna()) &
                              (group.index > bundle[-1])]
            
            if not next_trip.empty:
                bundle.append(next_trip.index[0])
                pbar.update(1)  # Update the progress bar
            else:
                break

        # If a bundle of at least 2 was formed, assign a block_id to it
        if len(bundle) >= 2:
            df.loc[bundle, 'block_id'] = block_id
            block_id += 1

        # Move to the next trip that doesn't have a block_id yet
        unassigned_trips = group[group['block_id'].isna()]
        if not unassigned_trips.empty:
            i = unassigned_trips.index[0]
        else:
            break

# Close the progress bar
pbar.close()

df.to_csv(r'../../results/trips-mapped-into-blocks-improved.csv', index=False)
