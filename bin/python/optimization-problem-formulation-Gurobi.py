import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time
import psutil
import os
from tqdm import tqdm
import gurobipy as grb



start = time.time()

# Get the current process
process = psutil.Process(os.getpid())

# Get the total memory available in bytes
total_memory = psutil.virtual_memory().total

# Convert bytes to GB
total_memory_gb = total_memory / (1024 ** 3)

print(f'Total memory: {total_memory_gb} GB')

# Define function to report memory and CPU usage
def report_usage():
    # Get the percentage of CPU usage by this process
    cpu_percent = process.cpu_percent(interval=1)

    # Get the memory usage by this process
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # in bytes

    print(f"Process CPU usage: {cpu_percent}%")
    print(f"Process memory usage: {memory_usage / (1024 * 1024)} MB")  # convert bytes to MB


# Read dataframes of all-CDB, all-HEB, and all BEB with runs included
df_CDB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-CDB.csv', low_memory=False)
df_HEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-HEB.csv', low_memory=False)
df_BEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-BEB.csv', low_memory=False)


# Find the date with the maximum number of unique trips
date_with_max_trips = df_CDB.groupby('Date')['TripKey'].nunique().idxmax()
max_trips = df_CDB.groupby('Date')['TripKey'].nunique().max()
print("date_with_max_trips is:", date_with_max_trips)
print("max_trips is:", max_trips)
report_usage()

# Filter dataframes by the day with max trips
df_CDB = df_CDB.loc[df_CDB['Date']==date_with_max_trips]
df_HEB = df_HEB.loc[df_HEB['Date']==date_with_max_trips]
df_BEB = df_BEB.loc[df_BEB['Date']==date_with_max_trips]
report_usage()

# Group the dataframes to capture first and last stop, start and ened time and route for each trip
df_CDB = df_CDB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'Powertrain':'first'
}).reset_index()

df_HEB = df_HEB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'Powertrain':'first'
}).reset_index()

df_BEB = df_BEB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'Powertrain':'first'
}).reset_index()

# Renaming columns
df_CDB.columns = ['_'.join(col).strip() for col in df_CDB.columns.values]
df_HEB.columns = ['_'.join(col).strip() for col in df_HEB.columns.values]
df_BEB.columns = ['_'.join(col).strip() for col in df_BEB.columns.values]
df_CDB = df_CDB.rename(columns={'TripKey_': 'TripKey'})
df_CDB = df_CDB.rename(columns={'Date_': 'Date'})
df_CDB = df_CDB.rename(columns={'Route_first': 'Route'})
df_CDB = df_CDB.rename(columns={'Energy_sum': 'Energy'})
df_CDB = df_CDB.rename(columns={'Powertrain_first': 'Powertrain'})
df_HEB = df_HEB.rename(columns={'TripKey_': 'TripKey'})
df_HEB = df_HEB.rename(columns={'Date_': 'Date'})
df_HEB = df_HEB.rename(columns={'Route_first': 'Route'})
df_HEB = df_HEB.rename(columns={'Energy_sum': 'Energy'})
df_HEB = df_HEB.rename(columns={'Powertrain_first': 'Powertrain'})
df_BEB = df_BEB.rename(columns={'TripKey_': 'TripKey'})
df_BEB = df_BEB.rename(columns={'Date_': 'Date'})
df_BEB = df_BEB.rename(columns={'Route_first': 'Route'})
df_BEB = df_BEB.rename(columns={'Energy_sum': 'Energy'})
df_BEB = df_BEB.rename(columns={'Powertrain_first': 'Powertrain'})
print("Dataframe columns list:", df_CDB.columns)

# Define parameters
#D = len(set(df_CDB['Date'].unique()))  # Create a set of unique dates
Y = 13  # Years in simulation
max_number_of_buses = 1000 # 213*4 (current numnumber of fleet*4, assuming buses are going to be replaced with electric at most with ratio of 1:4)


# Maximum daily charging capacity in year y
M_cap = {y: val for y, val in enumerate([5600, 8400, 10500, 12950, 15400, 18900] + [float('inf')] * (Y - 6))}

# Set of scenarios
#S = {'low-cap', 'mid-cap', 'high-cap'}
S = {'low-cap'}

# Define R and Rho
R = df_CDB['Route'].nunique()
Rho = max_trips

# The cost of purchasing a new bus
cost_inv = {
    ('B', y): 0.9 for y in range(Y)
}  # in million dollars
cost_inv.update({
    ('H', y): 1.3 for y in range(Y)
})  # in million dollars
cost_inv.update({
    ('C', y): 0 for y in range(Y)
})  # Assuming no cost for existing CDB buses

# Max investment per scenario per year
C_max = {
    'low-cap': 7,  # in million dollars
#    'mid-cap': 14,  # in million dollars
#    'high-cap': 21  # in million dollars
}

# The maximum yearly investment
M_inv = {
    (s, y): C_max[s]
    for y in range(Y) for s in S
}

# Battery capacity of an electric bus
cap = 350  # Battery capacity of an electric bus assumed to be 350 kW 

# Total number of fleet from each powertrain in year 0
N = {
    ('C', 0): 189,
    ('H', 0): 9,
    ('B', 0): 15,
}

# Define reasonable time and distance lags between two trips
trips_time_lag = 10 # min
trips_distance_lag = 5 # min

# Averga speed of buses
mean_v = 25 #mph

# Groupby to compute energy consumption for each unique vehicle, date, route, and trip key
# then, create the 'Diesel' column based on the condition for 'Powertrain'

# For df_CDB
energy_CDB = df_CDB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_CDB['Diesel'] = (energy_CDB['Powertrain'].isin(['conventional', 'hybrid']) * energy_CDB['Energy'])
energy_CDB_dict = energy_CDB.set_index(['TripKey']).to_dict('index')

# For df_HEB
energy_HEB = df_HEB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_HEB['Diesel'] = (energy_HEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_HEB['Energy'])
energy_HEB_dict = energy_HEB.set_index(['TripKey']).to_dict('index')

# For df_BEB
energy_BEB = df_BEB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = (energy_BEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_BEB['Energy'])
energy_BEB_dict = energy_BEB.set_index(['TripKey']).to_dict('index')

### Combine three dicts and save all trip information in a csv file
combined_dict = {}
combined_dict.update(energy_CDB_dict)
combined_dict.update(energy_HEB_dict)
combined_dict.update(energy_BEB_dict)

# Convert the dictionary to a DataFrame
df_combined_dict = pd.DataFrame.from_dict(combined_dict, orient='index')

# Write the DataFrame to a CSV file
df_combined_dict.to_csv(r'../../results/busiest-day-trips-info.csv', index=False)

# Now delete the DataFrame to free up memory
del df_CDB
del df_HEB
del df_BEB
gc.collect()
print("Done deleting unneccesary dataframes")
report_usage()

# Create a model
model = Model('Minimize fleet diesel consumption')
print("Done creating the model")
report_usage()

# Enable logging and print progress
model.setParam('OutputFlag', 1)

# Set solver parameters
model.setParam('MIPFocus', 1)  # This parameter lets control the  In. The options are: 1. Finds feasible solutions quickly. This is useful if the problem is difficult to solve and you're satisfied with any feasible solution. 2: Works to improve the best bound. This is useful when the best objective bound is poor. 3: Tries to prove optimality of the best solution found. This is useful if you're sure a nearly-optimal solution exists and you want the solver to focus on proving its optimality.
model.setParam('Heuristics', 0.5)  # Controls the effort put into MIP heuristics (range is 0 to 1). A higher value means more effort is put into finding solutions, but at the cost of slower overall performance. 
#model.setParam('Presolve', 1)  # This parameter controls the presolve level. Presolve is a phase during which the solver tries to simplify the model before the actual optimization takes place. A higher presolve level means the solver puts more effort into simplification, which can often reduce solving time. (-1: automatic (default) - Gurobi will decide based on the problem characteristics whether to use presolve or not.0: no presolve. 1: conservative presolve. 2: aggressive presolve.)
#model.setParam('MIPGap', 0.01) # This parameter sets the relative gap for the MIP search termination. The solver will stop as soon as the relative gap between the lower and upper objective bound is less than this value. The lower this value, the closer to optimality the solution has to be before the solver stops.  
model.setParam('Threads', 72)  # Set number of threads to be used for parallel processing.

# Adjust the Model Tolerance
model.setParam('FeasibilityTol', 1e-3)  # Feasibility tolerance. The default value is 1e-6
model.setParam('IntFeasTol', 1e-2)  # Integrality Tolerance. The default value is 1e-5
model.setParam('OptimalityTol', 1e-3)  # Optimality Tolerance. The default value is 1e-6

# Adjust time limit
#model.setParam('TimeLimit', 86400)  # in seconds. The default value is infinity

# Change the variable selection strategy: sets the method used to solve the LP relaxation at each node in the MIP tree
#model.setParam('NodeMethod', 0)  # 0: Automatic (let Gurobi choose), 1: Primal simplex, 2: Dual simplex. 

# Change the variable selection strategy used at each node in the MIP tree
#model.setParam('VarBranch', 0)  # -1: Pseudo-reduced cost branching, 0: Automatic (let Gurobi choose), 1: Maximum infeasibility, 2: Strong branching, 3: Pseudo-reduced cost branching, 4: Maximum infeasibility on unsatisfied (MaxInf*), 5: Pseudo-shadow price (PSP), 6: Strong branching on variables with high pseudo costs 

# Change the aggressiveness of the cut generation during the Branch-and-Cut process of solving mixed integer programs (MIPs)
#model.setParam('Cuts', 2)  #  Cutting planes are additional constraints that can potentially improve the LP relaxation of the problem, thus leading to a quicker solution. A higher value means more aggressive cut generation, but this could potentially slow down the solver because of the extra overhead.
#model.setParam('Cuts', -1)  # No cuts
#model.setParam('Cuts', 1)  # Conservative level
#model.setParam('Cuts', 2)  # Aggressive level
#model.setParam('Cuts', 3)  # Very aggressive level


print("Done setting model parameters")
report_usage()

# Additional keys for buses and years
bus_keys = range(max_number_of_buses)
year_keys = range(Y)
#day_keys = range(D)
#route_keys = range(R)
#run_keys = range(Rho)

# Decision variables
keys_CDB = list(energy_CDB_dict.keys())
keys_HEB = list(energy_HEB_dict.keys())
keys_BEB = list(energy_BEB_dict.keys())
print("len_keys_CDB",len(keys_CDB))
print("len_keys_HEB",len(keys_HEB))
print("len_keys_BEB",len(keys_BEB))
print("Done setting necessary keys")
report_usage()

print("Number of CDB variables:",len(bus_keys)*len(year_keys)*len(keys_CDB))
print("Number of HEB variables:",len(bus_keys)*len(year_keys)*len(keys_HEB))
print("Number of BEB variables:",len(bus_keys)*len(year_keys)*len(keys_BEB))


# Decision variables which include two additional indices for buses (i) and years (y)
x_CDB = model.addVars(S, bus_keys, year_keys, keys_CDB, vtype=GRB.BINARY, name='x_CDB')
x_HEB = model.addVars(S, bus_keys, year_keys, keys_HEB, vtype=GRB.BINARY, name='x_HEB')
x_BEB = model.addVars(S, bus_keys, year_keys, keys_BEB, vtype=GRB.BINARY, name='x_BEB')
print("Done setting x variables")
report_usage()

# Define y_CDB, y_HEB, and y_BEB as the number of each type of bus at each year under each scenario
y_CDB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_CDB')
y_HEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_HEB')
y_BEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_BEB')
print("Done setting y variables")
report_usage()

# Decision Variables for bus types
z_CDB = model.addVars(S, bus_keys, year_keys, vtype=GRB.BINARY, name="z_CDB")
z_HEB = model.addVars(S, bus_keys, year_keys, vtype=GRB.BINARY, name="z_HEB")
z_BEB = model.addVars(S, bus_keys, year_keys, vtype=GRB.BINARY, name="z_BEB")
print("Done setting z variables")
report_usage()

# Variables indicating the sequence of trips for each bus
u = model.addVars(S, bus_keys, year_keys, [(key, 'CDB') for key in keys_CDB] +
                                          [(key, 'HEB') for key in keys_HEB] +
                                          [(key, 'BEB') for key in keys_BEB], 
                  vtype=GRB.INTEGER, name='u')
#u_CDB = model.addVars(S, bus_keys, year_keys, keys_CDB, vtype=GRB.INTEGER, name='u_CDB')
#u_HEB = model.addVars(S, bus_keys, year_keys, keys_HEB, vtype=GRB.INTEGER, name='u_HEB')
#u_BEB = model.addVars(S, bus_keys, year_keys, keys_BEB, vtype=GRB.INTEGER, name='u_BEB')

### The variable u represents the position of a trip in the set of assigned trips to a bus. If a bus serves n trips, the trips should be numbered from 1 to n, in the order they are served.

print("Done setting u variables")
report_usage()

model.setObjective(
(quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[s, i, y, key] for s in S for key in keys_CDB for i in bus_keys for y in year_keys if key in energy_CDB_dict]) +
 quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[s, i, y, key] for s in S for key in keys_HEB for i in bus_keys for y in year_keys if key in energy_HEB_dict]) +
 quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[s, i, y, key] for s in S for key in keys_BEB for i in bus_keys for y in year_keys if key in energy_BEB_dict])),
    GRB.MINIMIZE
)

print("Done setting objective function")
report_usage()

### Compute the time and distance between each two trips
distance_df = pd.read_csv(r'../../results/stops-pairwise-distances.csv', low_memory=False)

# Define a function to get the distance between two stops
def get_distance(stop1, stop2):
    distance_row = distance_df[(distance_df['Stop1'] == stop1) & (distance_df['Stop2'] == stop2)]
    if distance_row.empty:
        return 200  # Return 200 miles when there is no direct path
    else:
        return distance_row.iloc[0]['Distance']

# Convert the string to datetime format
df_combined_dict['ServiceDateTime_max'] = pd.to_datetime(df_combined_dict['ServiceDateTime_max'])
df_combined_dict['ServiceDateTime_min'] = pd.to_datetime(df_combined_dict['ServiceDateTime_min'])

# Compute the duration of each trip in minutes
df_combined_dict['Duration'] = (df_combined_dict['ServiceDateTime_max'] - df_combined_dict['ServiceDateTime_min']).dt.total_seconds() / 60

# Compute the distance between the first and last stop of each trip
df_combined_dict['Distance'] = df_combined_dict.apply(lambda row: get_distance(row['Stop_first'], row['Stop_last']), axis=1)


## Define Constraints

# Constraint 1: Linking the number of each type of bus at each year variable with trip assignment variables
model.addConstrs(
   (y_CDB[s, y] == quicksum(x_CDB[s, i, y, key] for i in bus_keys for key in keys_CDB) for s in S for y in year_keys)
,
    name="C1_CDB"
)
model.addConstrs(
    (y_HEB[s, y] == quicksum(x_HEB[s, i, y, key] for i in bus_keys for key in keys_HEB) for s in S for y in year_keys)
,
    name="C1_HEB"
)
model.addConstrs(
    (y_BEB[s, y] == quicksum(x_BEB[s, i, y, key] for i in bus_keys for key in keys_BEB) for s in S for y in year_keys)
,
    name="C1_BEB"
)

### Aditional explanation: 
#x_CDB[i, y, d, r, rho] >= 1 is a binary condition that checks if bus 'i' is used at least once in a trip during year 'y'. This will return True (or 1) if bus 'i' is used, and False (or 0) otherwise. 
#Regardless of the values of d, r, and rho, the result of this expression is either 1 (if the bus 'i' is used at least once) or 0 (if the bus 'i' is not used at all).
print("Done defining constraint 1")
report_usage()


# Constraint 2: Linking the bus type variable with trip assignment variables
###you can only assign a trip to a bus of type CDB if that bus is available
model.addConstrs(
    (x_CDB[s, i, y, key] <= z_CDB[s, i, y] for s in S for i in bus_keys for y in year_keys for key in keys_CDB),
    name="C2_CDB"
)
model.addConstrs(
    (x_HEB[s, i, y, key] <= z_HEB[s, i, y] for s in S for i in bus_keys for y in year_keys for key in keys_HEB),
    name="C2_HEB"
)
model.addConstrs(
    (x_BEB[s, i, y, key] <= z_BEB[s, i, y] for s in S for i in bus_keys for y in year_keys for key in keys_BEB),
    name="C2_BEB"
)

print("Done defining constraint 2")
report_usage()


# Constraint 3: Linking the sequence variable with trip assignment variables
###the sequence in which a bus serves its trips cannot exceed the total number of trips assigned to that bus
# =============================================================================
# model.addConstrs(
#     (u[s, i, y, key, 'CDB'] <= quicksum(x_CDB[s, i, y, key] for key in keys_CDB) for s in S for i in bus_keys for y in year_keys for key in keys_CDB),
#     name="C3_CDB"
# )
# model.addConstrs(
#     (u[s, i, y, key, 'HEB'] <= quicksum(x_HEB[s, i, y, key] for key in keys_HEB) for s in S for i in bus_keys for y in year_keys for key in keys_HEB),
#     name="C3_HEB"
# )
# model.addConstrs(
#     (u[s, i, y, key, 'BEB'] <= quicksum(x_BEB[s, i, y, key] for key in keys_BEB) for s in S for i in bus_keys for y in year_keys for key in keys_BEB),
#     name="C3_BEB"
# )
# 
# =============================================================================

# Improve constaint 3 to do pre-calculation and see the progress 
# Pre-calculate sums
sums_CDB = {(s, i, y): quicksum(x_CDB[s, i, y, key] for key in keys_CDB) for s in S for i in bus_keys for y in year_keys}
sums_HEB = {(s, i, y): quicksum(x_HEB[s, i, y, key] for key in keys_HEB) for s in S for i in bus_keys for y in year_keys}
sums_BEB = {(s, i, y): quicksum(x_BEB[s, i, y, key] for key in keys_BEB) for s in S for i in bus_keys for y in year_keys}

pbar = tqdm(total=3*len(S)*len(bus_keys)*len(year_keys)*len(keys_CDB))  # assuming all keys have the same length

# Add constraints
for bus_type, sums, keys in [('CDB', sums_CDB, keys_CDB), ('HEB', sums_HEB, keys_HEB), ('BEB', sums_BEB, keys_BEB)]:
    for s in S:
        for i in bus_keys:
            for y in year_keys:
                for key in keys:
                    model.addConstr(u[s, i, y, key, bus_type] <= sums[(s, i, y)], name=f"C3_{bus_type}")
                    pbar.update()
pbar.close()

print("Done defining constraint 3")
report_usage()

# Constraint 4: The sum of decision variables for each bus and year across all powertrains should be <= 1
model.addConstrs(
    (z_CDB[s, i, y] + z_HEB[s, i, y] + z_BEB[s, i, y] <= 1 for s in S for i in bus_keys for y in year_keys),
    name="C4"
)
print("Done defining constraint 4")
report_usage()

# Constraint 5: Each trip is assigned to exactly one bus
unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
model.addConstrs(
    (
        (
            quicksum(x_CDB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_CDB_dict) +
            quicksum(x_HEB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_HEB_dict) +
            quicksum(x_BEB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_BEB_dict)
        ) == 1 for key in unique_keys
    ), 
    name="C5"
)
print("Done defining constraint 5")
report_usage()

# Constraint 6: Maximum daily charging capacity
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB.sum(s, i, y, key) for s in S for i in bus_keys for key in keys_BEB) <= M_cap[y] for y in year_keys),
    name="C6"
)
print("Done defining constraint 6")
report_usage()

# Constraint 7: Bus Range Constraint- Add a constraint that ensures that a bus doesn't exceed its maximum range in a day
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB[s, i, y, key] for s in S for i in bus_keys for key in keys_BEB) <= cap for y in year_keys),
    name="C7"
)
print("Done defining constraint 7")
report_usage()

# Constraint 8: Maximum yearly investment
model.addConstrs(
    (quicksum(cost_inv[('C', year)] * y_CDB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('H', year)] * y_HEB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('B', year)] * y_BEB.sum(year, '*') for year in range(y + 1)) <= M_inv[s, y]
    for y in year_keys for s in S),
    name="C8"
)
print("Done defining constraint 8")
report_usage()

 
# Constraint 9: Enforce the sequence of the trips. If u[s, i, y, (t, bus_type)] represents the sequence number of trip t of type bus_type assigned to bus i in year y under scenario s
bus_types = ['CDB', 'HEB', 'BEB']
for s in S:
    for i in bus_keys:
        for y in year_keys:
            for bus_type in bus_types:
                if bus_type == 'CDB':
                    keys = keys_CDB
                    x = x_CDB
                elif bus_type == 'HEB':
                    keys = keys_HEB
                    x = x_HEB
                else:  # bus_type == 'BEB'
                    keys = keys_BEB
                    x = x_BEB
                sorted_trips = sorted(keys, key=lambda x: df_combined_dict.loc[x,'ServiceDateTime_min'])
                for j in range(len(sorted_trips) - 1):
                    model.addConstr(u[s, i, y, sorted_trips[j], bus_type] <= u[s, i, y, sorted_trips[j + 1], bus_type], 'sequence_c9')
                    
print("Done defining constraint 9")
report_usage()


# Constraint 10: The start times of each trip in the sequence of all trips assigned to a unique bus is greater than equal to the start time of the previous trip plus the time it takes from the last stop of the first trip to the first stop of the second trip
df_combined_dict['ServiceDateTime_min'] = df_combined_dict['ServiceDateTime_min'].apply(lambda x: x.timestamp())
df_combined_dict['ServiceDateTime_max'] = df_combined_dict['ServiceDateTime_max'].apply(lambda x: x.timestamp())
sorted_trips_CDB = sorted(keys_CDB, key=lambda x: df_combined_dict.loc[x,'ServiceDateTime_min'])
sorted_trips_HEB = sorted(keys_HEB, key=lambda x: df_combined_dict.loc[x,'ServiceDateTime_min'])
sorted_trips_BEB = sorted(keys_BEB, key=lambda x: df_combined_dict.loc[x,'ServiceDateTime_min'])

bus_types_keys = {'CDB': (keys_CDB, x_CDB, sorted_trips_CDB),
                  'HEB': (keys_HEB, x_HEB, sorted_trips_HEB),
                  'BEB': (keys_BEB, x_BEB, sorted_trips_BEB)}

pbar = tqdm(total=len(S)*len(bus_keys)*len(year_keys)*len(bus_types)*len(sorted_trips_CDB))  # assuming all sorted_trips have the same length, change if not
# Pre-calculate travel times
travel_times = {}
for bus_type in bus_types:
    keys, x, sorted_trips = bus_types_keys[bus_type]
    for j in range(1, len(sorted_trips)):
        trip1 = sorted_trips[j-1]
        trip2 = sorted_trips[j]
        travel_times[(trip1, trip2)] = (df_combined_dict.loc[trip1,'ServiceDateTime_max'] +
                                         get_distance(df_combined_dict.loc[trip1,'Stop_last'], df_combined_dict.loc[trip2,'Stop_first']) / mean_v)*3600

def create_constraint(bus_key, S, year_keys, bus_types_keys, df_combined_dict, travel_times):
    constraints = []
    for s in S:
        for y in year_keys:
            for bus_type in bus_types:
                keys, x, sorted_trips = bus_types_keys[bus_type]
                for j in range(1, len(sorted_trips)):
                    trip1 = sorted_trips[j-1]
                    trip2 = sorted_trips[j]
                    constraints.append(
                        x[s, bus_key, y, trip2] * df_combined_dict.loc[trip2,'ServiceDateTime_min'] >=
                        x[s, bus_key, y, trip1] * travel_times[(trip1, trip2)]
                    )
    return constraints

for bus_key in tqdm(bus_keys, desc="Generating constraints"):
    constraints = create_constraint(bus_key, S, year_keys, bus_types_keys, df_combined_dict, travel_times)
    for c in constraints:
        model.addConstr(c, 'travel_time_c10')


print("Done defining constraint 10")
report_usage()
     

# Print model statistics
model.update()
print(model)
        
# Tuning and Optimization
model.tune()
model.optimize()
report_usage()


# Prepare dictionaries of coeesicients to save
coeff_dict_CDB = {(s, i, y, key): energy_CDB_dict[key]['Diesel'] for s in S for key in keys_CDB for i in bus_keys for y in year_keys if key in energy_CDB_dict}
coeff_dict_HEB = {(s, i, y, key): energy_HEB_dict[key]['Diesel'] for s in S for key in keys_HEB for i in bus_keys for y in year_keys if key in energy_HEB_dict}
coeff_dict_BEB = {(s, i, y, key): energy_BEB_dict[key]['Diesel'] for s in S for key in keys_BEB for i in bus_keys for y in year_keys if key in energy_BEB_dict}

# Combine all the dictionaries into a dataframe
coeff_df = pd.DataFrame(list(coeff_dict_CDB.items()) + list(coeff_dict_HEB.items()) + list(coeff_dict_BEB.items()), columns=['Variable', 'Coefficient'])


vars = model.getVars()

# Create DataFrame directly from the variables and their values
if model.status == grb.GRB.Status.OPTIMAL:
    df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.X for v in vars]})
else:
    print('No solution found')

# Identify the smallest infeasible subset of constraints and variable bounds
if model.status == grb.GRB.INFEASIBLE:
    model.computeIIS()
    model.write("model.ilp")

# Generate a feasibility report
if model.status == grb.GRB.INFEASIBLE:
    model.feasRelaxS(0, False, False, True)
    model.optimize()

# Get optimal objective value
optimal_value = model.ObjVal

# Add this value to DataFrame
df_objective = pd.DataFrame({"Year": year_keys, "Objective_Value": [optimal_value]*len(year_keys)})

# Print objective value
print("optimal_value:",optimal_value)

# Save the DataFrame to a CSV file
df.to_csv(r'../../results/strategies-simulation-optimized-variables.csv', index=False)
coeff_df.to_csv(r'../../results/variable_coefficients.csv', index=False)

end = time.time()
report_usage()
print("Total time spend in seconds",end - start)