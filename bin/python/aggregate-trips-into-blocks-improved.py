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

# Get unique routes
routes = df['Route'].unique()

# Initialize the outer progress bar
with tqdm(total=len(routes), desc='Routes', position=0) as pbar_routes:

    # Iterate over the unique routes
    for route in routes:
        # Work on a subset of df with only the current route
        group = df[df['Route'] == route].copy()
        
        # Initialize the inner progress bar
        with tqdm(total=len(group), desc='Trips in route '+str(route), position=1, leave=False) as pbar_trips:
            
            # Start with the first trip
            i = group.index[0]
            
            # Iterate over DataFrame rows of the group
            while i in group.index:
                # Start a bundle
                bundle = [i]
                
                # Try adding trips to the bundle
                while len(bundle) < 10:
                    # Find the next trip that starts within 0.6 minutes after the last one in the bundle ends
                    next_trip = group[(group['Start_time'] - group.at[bundle[-1], 'End_time'] <= timedelta(minutes=0.6)) & 
                                      (group['block_id'].isna()) &
                                      (group.index > bundle[-1])]
                    
                    if not next_trip.empty:
                        bundle.append(next_trip.index[0])
                    else:
                        break
                
                # If a bundle of at least 2 was formed, assign a block_id to it
                if len(bundle) >= 2:
                    df.loc[bundle, 'block_id'] = block_id
                    block_id += 1
                
                # Update the inner progress bar
                pbar_trips.update(len(bundle))
                
                # Move to the next trip that doesn't have a block_id yet
                unassigned_trips = group[group['block_id'].isna()]
                if not unassigned_trips.empty:
                    i = unassigned_trips.index[0]
                else:
                    break

        # Update the outer progress bar
        pbar_routes.update(1)

df.to_csv(r'../../results/trips-mapped-into-blocks-improved.csv', index=False)
