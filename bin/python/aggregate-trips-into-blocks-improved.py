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

# Initialize variables
block_id = 1
i = 0

# Initialize the progress bar
pbar = tqdm(total=len(df))

# Iterate over DataFrame rows
while i < len(df):
    # Start a bundle
    bundle = [i]
    
    # Try adding trips to the bundle
    while len(bundle) < 10:
        # Find the next trip that starts within 0.6 minutes after the last one in the bundle ends and shares the same route
        next_trip = df[(df['Start_time'] - df.at[bundle[-1], 'End_time'] <= timedelta(minutes=0.6)) & 
                       (df['Route'] == df.at[bundle[-1], 'Route']) & 
                       (df['block_id'].isna()) &
                       (df.index > bundle[-1])]

        if not next_trip.empty:
            bundle.append(next_trip.index[0])
        else:
            break

    # If a bundle of at least 2 was formed, assign a block_id to it
    if len(bundle) >= 2:
        df.loc[bundle, 'block_id'] = block_id
        block_id += 1

    # Move to the next trip that doesn't have a block_id yet
    i = df[df['block_id'].isna()].index[0]
    
    # Update the progress bar
    pbar.update(len(bundle))

# Close the progress bar
pbar.close()

df.to_csv(r'../../results/trips-mapped-into-blocks-improved.csv', index=False)
