import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

df = pd.read_csv(r'../../results/trips-mapped-into-blocks.csv', low_memory=False)

df['Start_time'] = pd.to_datetime(df['Start_time'])
df['End_time'] = pd.to_datetime(df['End_time'])

df.sort_values(by=['Start_time', 'End_time'], inplace=True)
df.reset_index(drop=True, inplace=True)

df['block_id'] = np.nan

block_id = 1
routes = df['Route'].unique()

with tqdm(total=len(routes), desc='Routes', position=0) as pbar_routes:
    for route in routes:
        group = df[df['Route'] == route].copy()
        with tqdm(total=len(group), desc='Trips in route '+str(route), position=1, leave=False) as pbar_trips:
            i = group.index[0]
            while i is not None and i in group.index:
                bundle = [i]
                while len(bundle) < 10:
                    next_trip = group[(group['Start_time'] - group.at[bundle[-1], 'End_time'] <= timedelta(minutes=0.6)) & 
                      (group['block_id'].isna()) &
                      (group.index > bundle[-1])]

                    if not next_trip.empty:
                        bundle.append(next_trip.index[0])
                    else:
                        break

            if len(bundle) >= 2:
                df.loc[bundle, 'block_id'] = block_id
                group.loc[bundle, 'block_id'] = block_id  # update 'block_id' in 'group' DataFrame as well
                print("A new bundle formed:", block_id, "len is:", len(bundle))
                block_id += 1
                
                pbar_trips.update(len(bundle))

                unassigned_trips = group[group['block_id'].isna()]
                if not unassigned_trips.empty:
                    i = unassigned_trips.index[0]
                else:
                    i = None
        pbar_routes.update(1)

df.to_csv(r'../../results/trips-mapped-into-blocks-improved.csv', index=False)
