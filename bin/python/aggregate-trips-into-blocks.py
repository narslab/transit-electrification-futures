import pandas as pd
from tqdm import tqdm

df = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-CDB.csv', low_memory=False)

# convert ServiceDateTime to datetime type if it's not
df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])

# filter for the date
df = df[df['Date'] == '2021-10-29']

# Group by TripKey and get first and last entry for each trip
df_grouped = df.sort_values('ServiceDateTime').groupby('TripKey')
trip_starts = df_grouped.first().reset_index()[['TripKey', 'Stop', 'ServiceDateTime', 'Route']]
trip_ends = df_grouped.last().reset_index()[['TripKey', 'Stop', 'ServiceDateTime']]

# Merge start and end dataframes and sort by start time
trips_df = pd.merge(trip_starts, trip_ends, on='TripKey', suffixes=('_start', '_end'))
trips_df = trips_df.sort_values('ServiceDateTime_start')

# =============================================================================
# # Fetch the pairwise distance between two stops
# def get_travel_time(stop1, stop2):
#     if stop1 == stop2:
#         return 0
#     try:
#         distance = distance_df.loc[((distance_df['Stop1'] == stop1) & (distance_df['Stop2'] == stop2)) | 
#                                    ((distance_df['Stop1'] == stop2) & (distance_df['Stop2'] == stop1)), 'Distance'].values[0]
#         # Assuming distance is in miles and speed is in mph
#         travel_time_hours = distance / 25
#         # Convert travel time to minutes
#         travel_time_minutes = travel_time_hours * 60
#         return travel_time_minutes
#     except IndexError:
#         print(f"No distance found for pair of stops: {stop1} and {stop2}")
#         raise Exception("No distance found")
# =============================================================================

blocks = []
current_block = []

# iterate through each unique route
for route in tqdm(trips_df['Route'].unique()):
    # filter trips for current route
    trips_of_route = trips_df[trips_df['Route'] == route].copy().reset_index(drop=True)

    # Initialize the first block with the first trip
    blocks = [[trips_of_route.iloc[0]]]

    for i in range(1, len(trips_of_route)):
        next_trip = trips_of_route.iloc[i]
        
        if blocks[-1]:  # check if last block is not empty
            last_trip = blocks[-1][-1]

            # Check if the next trip can be added to the current block based on time and size
            can_add_next_trip = next_trip['ServiceDateTime_start'] == last_trip['ServiceDateTime_end'] and len(blocks[-1]) < 10

            # If the next trip can be added, add it to the current block
            if can_add_next_trip:
                blocks[-1].append(next_trip)
            # If the next trip can't be added to the current block, start a new block with this trip
            else:
                blocks.append([next_trip])
        else:
            # If the last block is empty, add the next trip to it
            blocks[-1].append(next_trip)

        # If current block already has 10 trips, start a new block with next trip
        if len(blocks[-1]) == 10:
            blocks.append([])

# Remove last block if it is empty
if not blocks[-1]:
    blocks = blocks[:-1]

# Flatten list of blocks to dataframe for saving, including additional info
block_df = pd.DataFrame([(i, trip['TripKey'], trip['Route'], trip['ServiceDateTime_start'], trip['ServiceDateTime_end'], trip['Stop_start'], trip['Stop_end']) for i, block in enumerate(blocks) for trip in block], columns=['block_id', 'TripKey', 'Route', 'Start_time', 'End_time', 'First_stop', 'Last_stop'])

block_df.to_csv(r'../../results/trips-mapped-into-blocks-consecutive-scheduling-approach.csv', index=False)
