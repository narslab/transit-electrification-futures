import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time
import psutil
import os
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
df_CDB = pd.read_csv(r'../../results/computed-fuel-rates-all-CDB.csv', low_memory=False)
df_HEB = pd.read_csv(r'../../results/computed-fuel-rates-all-HEB.csv', low_memory=False)
df_BEB = pd.read_csv(r'../../results/computed-fuel-rates-all-BEB.csv', low_memory=False)

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

## Remove Trips with zero Energy consumption with CDB vehicles
# Find the TripKey values in df_CDB where sum of Energy column is zero when grouped by TripKey
tripkeys_to_remove = df_CDB.groupby('TripKey').filter(lambda group: group['Energy'].sum() == 0)['TripKey'].unique()

# Remove the rows from all three dataframes that match the TripKey values found above
df_CDB = df_CDB[~df_CDB['TripKey'].isin(tripkeys_to_remove)]
df_HEB = df_HEB[~df_HEB['TripKey'].isin(tripkeys_to_remove)]
df_BEB = df_BEB[~df_BEB['TripKey'].isin(tripkeys_to_remove)]
report_usage()

# Group the dataframes to capture first and last stop, start and ened time and route for each trip
df_CDB = df_CDB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'dist':'sum',
    'Powertrain':'first'
}).reset_index()

df_HEB = df_HEB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'dist':'sum',
    'Powertrain':'first'
}).reset_index()

df_BEB = df_BEB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'dist':'sum',
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
df_CDB = df_CDB.rename(columns={'dist_sum': 'dist'})
df_CDB = df_CDB.rename(columns={'Powertrain_first': 'Powertrain'})
df_HEB = df_HEB.rename(columns={'TripKey_': 'TripKey'})
df_HEB = df_HEB.rename(columns={'Date_': 'Date'})
df_HEB = df_HEB.rename(columns={'Route_first': 'Route'})
df_HEB = df_HEB.rename(columns={'Energy_sum': 'Energy'})
df_HEB = df_HEB.rename(columns={'dist_sum': 'dist'})
df_HEB = df_HEB.rename(columns={'Powertrain_first': 'Powertrain'})
df_BEB = df_BEB.rename(columns={'TripKey_': 'TripKey'})
df_BEB = df_BEB.rename(columns={'Date_': 'Date'})
df_BEB = df_BEB.rename(columns={'Route_first': 'Route'})
df_BEB = df_BEB.rename(columns={'Energy_sum': 'Energy'})
df_BEB = df_BEB.rename(columns={'dist_sum': 'dist'})
df_BEB = df_BEB.rename(columns={'Powertrain_first': 'Powertrain'})

# Define parameters
#D = len(set(df_CDB['Date'].unique()))  # Create a set of unique dates
Y = 13  # Years in simulation (including year 0)
max_number_of_buses = 1000 # 213*4 (current numnumber of fleet*4, assuming buses are going to be replaced with electric at most with ratio of 1:4)

# Batery capacity of an electric bus
battery_cap=350 #kWh 

# Maximum daily charging capacity in year y
#M_cap = [23, 23, 27, 38, 42, 52, 62, 74, 89, 107, 128, 154, 185]
M_cap = [23, 23, 27, 38, 42, 52, 73, 102, 143, 200, 280, 392, 548]


# Set of scenarios
#S = {'low-cap', 'mid-cap', 'high-cap'}
S = {'high-cap'}

# Define R and Rho
R = df_CDB['Route'].nunique()
Rho = max_trips

# The cost of purchasing a new bus
cost_inv = {
    ('C'): 0.5,  # in million dollars
    ('H'): 0.9,  # in million dollars
    ('B'): 1.3   # in million dollars
}

# Max investment per scenario per year
C_max = {
#    'low-cap': 10,  # in million dollars
#     'mid-cap': 20,  # in million dollars
    'high-cap': 30  # in million dollars
}

M_inv = {
    (s, y): C_max[s] for y in range(Y) for s in S
}

# Bus ranges
#range_CDB= 332-50 # in miles (NEW flyer XD40 tank cap= 473 liters or 125 gal, mean fuel economy = 2.66 MPG)
#range_HEB= 381-50 # in miles (minues 10 miles buffer to go to the garage), mean fuel economy = 3.05 MPG 
#range_BEB= 200-50 # in miles (minues 10 miles buffer to go to the garage), from the manual
range_CDB= 93-10 # mean actual values in miles 
range_HEB= 110-10 # mean actual values in miles 
range_BEB= 55-10 # mean actual values in miles 
#range_CDB= 129 # 3rd quantile actual values in miles 
#range_HEB= 133 # 3rd quantile actual values in miles 
#range_BEB= 85 # 3rd quantile actual values in miles 
# Total number of fleet from each powertrain in year 0
N = {
    ('C', 0): 189,
    ('H', 0): 9,
    ('B', 0): 15,
}

# Groupby to compute energy consumption for each unique vehicle, date, route, and trip key
# then, create the 'Diesel' column based on the condition for 'Powertrain'

# For df_CDB
energy_CDB = df_CDB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'dist': 'sum', 
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
    'dist': 'sum', 
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
    'dist': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = (energy_BEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_BEB['Energy'])
energy_BEB_dict = energy_BEB.set_index(['TripKey']).to_dict('index')


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
model.setParam('Presolve', 0)  # This parameter controls the presolve level. Presolve is a phase during which the solver tries to simplify the model before the actual optimization takes place. A higher presolve level means the solver puts more effort into simplification, which can often reduce solving time. (-1: automatic (default) - Gurobi will decide based on the problem characteristics whether to use presolve or not.0: no presolve. 1: conservative presolve. 2: aggressive presolve.)
model.setParam('Threads', 64)  # Set number of threads to be used for parallel processing.

print("Done setting model parameters")
report_usage()

# Additional keys for buses and years
bus_keys = range(max_number_of_buses)
year_keys = range(Y)

# Decision variables
keys_CDB = list(energy_CDB_dict.keys())
keys_HEB = list(energy_HEB_dict.keys())
keys_BEB = list(energy_BEB_dict.keys())
print("Done setting necessary keys")
report_usage()

print("Number of CDB variables:",len(year_keys)*len(keys_CDB))
print("Number of HEB variables:",len(year_keys)*len(keys_HEB))
print("Number of BEB variables:",len(year_keys)*len(keys_BEB))

# Decision variables which include two additional indices for buses (i) and years (y)
x_CDB = model.addVars(S, year_keys, keys_CDB, vtype=GRB.BINARY, name='x_CDB')
x_HEB = model.addVars(S, year_keys, keys_HEB, vtype=GRB.BINARY, name='x_HEB')
x_BEB = model.addVars(S, year_keys, keys_BEB, vtype=GRB.BINARY, name='x_BEB')
print("Done setting x variables")
report_usage()

# Define y_CDB, y_HEB, and y_BEB as the number of each type of bus at each year under each scenario
y_CDB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_CDB')
y_HEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_HEB')
y_BEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_BEB')
print("Done setting y variables")
report_usage()

# Define decision variables delta to show number of new buses
model.setObjective(
 (quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[s, y, key] for s in S for key in keys_CDB for y in year_keys]) +
  quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[s, y, key] for s in S for key in keys_HEB for y in year_keys]) +
  quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[s, y, key] for s in S for key in keys_BEB for y in year_keys])),
     GRB.MINIMIZE
 )

print("Done setting objective function")
report_usage()

## Define Constraints

# Constraint 1: Linking the number of each type of bus at each year variable with trip assignment variables
### ensure the number of buses of type X (y_X[s, y]) is at least the total distance traveled by all trips assigned to buses of type X divided by the range of the bus (rounded up to ensure all trips are covered).
# For CDB buses
for s in S:
    for y in year_keys:
        total_distance_CDB = quicksum(energy_CDB_dict[key]['dist'] * x_CDB[s, y, key] for key in keys_CDB)
        
        model.addConstr(
            y_CDB[s, y] * range_CDB >= total_distance_CDB, 
            name=f"C1_numberofCDB_ge_{s}_{y}"
        )
        model.addConstr(
            y_CDB[s, y] * range_CDB <= total_distance_CDB + range_CDB,
            name=f"C1_numberofCDBs_le_{s}_{y}"
        )

# For HEB buses
for s in S:
    for y in year_keys:
        total_distance_HEB = quicksum(energy_HEB_dict[key]['dist'] * x_HEB[s, y, key] for key in keys_HEB)
        
        model.addConstr(
            y_HEB[s, y] * range_HEB >= total_distance_HEB, 
            name=f"C1_numberofHEBs_ge_{s}_{y}"
        )
        model.addConstr(
            y_HEB[s, y] * range_HEB <= total_distance_HEB + range_HEB,
            name=f"C1_numberofHEBs_le_{s}_{y}"
        )

# For BEB buses
for s in S:
    for y in year_keys:
        total_distance_BEB = quicksum(energy_BEB_dict[key]['dist'] * x_BEB[s, y, key] for key in keys_BEB)
        
        model.addConstr(
            y_BEB[s, y] * range_BEB >= total_distance_BEB, 
            name=f"C1_numberofBEBs_ge_{s}_{y}"
        )
        model.addConstr(
            y_BEB[s, y] * range_BEB <= total_distance_BEB + range_BEB,
            name=f"C1_numberofBEBs_le_{s}_{y}"
        )

print("Done defining constraint 1")
report_usage()


# Constraint 2: Each trip is assigned to exactly one bus powertrain
for s in S:
    for y in year_keys:
        #for key in unique_keys:
        for key in keys_CDB:
            model.addConstr(
                x_CDB[s, y, key] + x_HEB[s, y, key] + x_BEB[s, y, key] == 1,
                name=f"C2_Trip{key}_S{s}_Y{y}: Each trip is assigned to only one powertrain"
            )

print("Done defining constraint 2")
report_usage()

# Constraint 3: Maximum daily charging capacity
for s in S:
     for y in year_keys:
         total_BEB = y_BEB[s,y]
         model.addConstr(total_BEB <= M_cap[y], name=f"C3: daily charging capacity_{y}_{s}")
 

print("Done defining constraint 3")
report_usage()

# Constraint 4: Maximum yearly investment
# Initial values for year 0
initial_fleet_CDB = 189
initial_fleet_HEB = 9
initial_fleet_BEB = 15

for s in S:
    for y in year_keys:
        z_HEB = model.addVar(vtype=GRB.BINARY, name=f"z_HEB_{s}_{y}")
        z_BEB = model.addVar(vtype=GRB.BINARY, name=f"z_BEB_{s}_{y}")

        positive_delta_fleet_HEB = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"positive_delta_HEB_{s}_{y}")
        positive_delta_fleet_BEB = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"positive_delta_BEB_{s}_{y}")

        # Adjustments for year 0 to use initial values
        if y == 0:
            delta_fleet_HEB = y_HEB[s, y] - initial_fleet_HEB
            delta_fleet_BEB = y_BEB[s, y] - initial_fleet_BEB
        else:
            delta_fleet_HEB = y_HEB[s, y] - y_HEB[s, y-1]
            delta_fleet_BEB = y_BEB[s, y] - y_BEB[s, y-1]

        big_M = 1e6
        model.addConstr(positive_delta_fleet_HEB >= delta_fleet_HEB)
        model.addConstr(positive_delta_fleet_HEB <= delta_fleet_HEB + big_M * (1 - z_HEB))
         
        model.addConstr(positive_delta_fleet_BEB >= delta_fleet_BEB)
        model.addConstr(positive_delta_fleet_BEB <= delta_fleet_BEB + big_M * (1 - z_BEB))
         
        model.addConstr(delta_fleet_HEB <= big_M * z_HEB)
        model.addConstr(delta_fleet_BEB <= big_M * z_BEB)
 
        HEB_investment = positive_delta_fleet_HEB * cost_inv[('H')]
        BEB_investment = positive_delta_fleet_BEB * cost_inv[('B')]
 
        total_investment = HEB_investment + BEB_investment
        model.addConstr(total_investment <= M_inv[s, y], name=f"C4: Max yearly investment_{y}_{s}")
 
print("Done defining constraint 4")
report_usage()


# Constraint 5: Total number of buses (y) (summed over all powertrain) per year cannot exceed 1000
for s in S:
    for y in year_keys:
        model.addConstr(
            y_CDB[s, y] + y_HEB[s, y] + y_BEB[s, y] <= max_number_of_buses,
            name=f"C5: TotalFleetSize_{y}_{s}"
        )
print("Done defining constraint 5")
report_usage()   

# Constraint 6: The number of hybrid buses (HEBs) for a given year cannot be more than the previous year
for s in S:
    for y in range(len(year_keys)):  # Start from the second year since the first year doesn't have a previous year
        if y == 0:
            initial_fleet_HEB = 9
            delta_HEB = y_HEB[s, y] - initial_fleet_HEB
            model.addConstr(
            delta_HEB <= 0,
            name=f"C6_NoIncreaseInHEBs_{s}_{y}"
            )
        else:
            delta_HEB = y_HEB[s, y] - y_HEB[s, y-1]
            model.addConstr(
            delta_HEB <= 0,
            name=f"C6_NoIncreaseInHEBs_{s}_{y}"
            )
print("Done defining constraint 6")
report_usage()    

# Print model statistics
model.update()
print(model)
        
# Tuning and Optimization
model.tune()
model.optimize()
report_usage()


# Prepare dictionaries of coeesicients to save
coeff_dict_CDB = {(s, y, key): energy_CDB_dict[key]['Diesel'] for s in S for key in keys_CDB for y in year_keys if key in energy_CDB_dict}
coeff_dict_HEB = {(s, y, key): energy_HEB_dict[key]['Diesel'] for s in S for key in keys_HEB for y in year_keys if key in energy_HEB_dict}
coeff_dict_BEB = {(s, y, key): energy_BEB_dict[key]['Diesel'] for s in S for key in keys_BEB for y in year_keys if key in energy_BEB_dict}

# Combine all the dictionaries into a dataframe
coeff_df = pd.DataFrame(list(coeff_dict_CDB.items()) + list(coeff_dict_HEB.items()) + list(coeff_dict_BEB.items()), columns=['Variable', 'Coefficient'])


vars = model.getVars()

# Create DataFrame directly from the variables and their values
if model.status == grb.GRB.Status.OPTIMAL:
    df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.X for v in vars]})
#    df.to_csv(r'../../results/balanced-transition-highcap-optimized-variables.csv', index=False)
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

df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.X for v in vars]})

# Save the DataFrame to a CSV file
df.to_csv(r'../../results/highcap-FE-optimized-variables-with-replacement-40percentBEBcharging.csv', index=False)
#coeff_df.to_csv(r'../../results/optimization-coefficients.csv', index=False)

end = time.time()
report_usage()
print("Total time spend in seconds",end - start)