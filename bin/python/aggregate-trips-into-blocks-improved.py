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
    for j in range(i + 1, len(df)):
        if df.at[j, 'block_id'] is not np.nan:
            continue
        if (df.at[j, 'Start_time'] - df.at[bundle[-1], 'End_time'] <= timedelta(minutes=0.6)) and (df.at[j, 'Route'] == df.at[bundle[-1], 'Route']):
            bundle.append(j)
            pbar.update(1)  # Update the progress bar
            if len(bundle) == 10:
                break
        else:
            break

    # If a bundle of at least 2 was formed, assign a block_id to it
    if len(bundle) >= 2:
        df.loc[bundle, 'block_id'] = block_id
        block_id += 1

    # Move to the next trip that doesn't have a block_id yet
    i = j + 1

# Close the progress bar
pbar.close()

df.to_csv(r'../../results/trips-mapped-into-blocks-improved.csv', index=False)
