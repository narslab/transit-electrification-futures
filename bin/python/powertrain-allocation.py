import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time
import psutil
import os
from tqdm import tqdm
import gurobipy as grb
from math import ceil


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
print("Dataframe columns list:", df_CDB.columns)

# Define parameters
#D = len(set(df_CDB['Date'].unique()))  # Create a set of unique dates
Y = 13  # Years in simulation
max_number_of_buses = 1000 # 213*4 (current numnumber of fleet*4, assuming buses are going to be replaced with electric at most with ratio of 1:4)

# Batery capacity of an electric bus
battery_cap=350 #kWh 

# Maximum daily charging capacity in year y
M_cap = {y: val for y, val in enumerate([5600, 8400, 10500, 12950, 15400, 18900] + [float('inf')] * (Y - 6))}

# Set of scenarios
#S = {'low-cap', 'mid-cap', 'high-cap'}
S = {'high-cap'}

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
#    'low-cap': 7,  # in million dollars
#     'mid-cap': 14,  # in million dollars
     'high-cap': 21  # in million dollars
}

# The maximum yearly investment
M_inv = {
    (s, y): C_max[s]
    for y in range(Y) for s in S
}

# Bus ranges
range_CDB= 690-10 # in miles ( NEW flyer XD40 tank cap= 473 liters or 125 gal, mean fuel economy = 5.52 MPG)
range_HEB= 701-10 # in miles (minues 20 miles buffer to go to the garage) 
range_BEB= 200-10 # in miles (minues 20 miles buffer to go to the garage)

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
#model.setParam('MIPFocus', 1)  # This parameter lets control the  In. The options are: 1. Finds feasible solutions quickly. This is useful if the problem is difficult to solve and you're satisfied with any feasible solution. 2: Works to improve the best bound. This is useful when the best objective bound is poor. 3: Tries to prove optimality of the best solution found. This is useful if you're sure a nearly-optimal solution exists and you want the solver to focus on proving its optimality.
#model.setParam('Heuristics', 0.5)  # Controls the effort put into MIP heuristics (range is 0 to 1). A higher value means more effort is put into finding solutions, but at the cost of slower overall performance. 
#model.setParam('Presolve', 1)  # This parameter controls the presolve level. Presolve is a phase during which the solver tries to simplify the model before the actual optimization takes place. A higher presolve level means the solver puts more effort into simplification, which can often reduce solving time. (-1: automatic (default) - Gurobi will decide based on the problem characteristics whether to use presolve or not.0: no presolve. 1: conservative presolve. 2: aggressive presolve.)
#model.setParam('MIPGap', 0.01) # This parameter sets the relative gap for the MIP search termination. The solver will stop as soon as the relative gap between the lower and upper objective bound is less than this value. The lower this value, the closer to optimality the solution has to be before the solver stops.  
model.setParam('Threads', 64)  # Set number of threads to be used for parallel processing.

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


print("Done setting u variables")
report_usage()

model.setObjective(
(quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[s, y, key] for s in S for key in keys_CDB for y in year_keys if key in energy_CDB_dict]) +
 quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[s, y, key] for s in S for key in keys_HEB for y in year_keys if key in energy_HEB_dict]) +
 quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[s, y, key] for s in S for key in keys_BEB for y in year_keys if key in energy_BEB_dict])),
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
        total_distance_CDB = quicksum(energy_CDB_dict[key]['dist'] * x_CDB[s, y, key] for key in keys_CDB if key in energy_CDB_dict)
        
        model.addConstr(
            y_CDB[s, y] * range_CDB >= total_distance_CDB, 
            name=f"C1_CDB_ge_{s}_{y}"
        )
        model.addConstr(
            y_CDB[s, y] * range_CDB < total_distance_CDB + range_CDB,
            name=f"C1_CDB_lt_{s}_{y}"
        )

# For HEB buses
for s in S:
    for y in year_keys:
        total_distance_HEB = quicksum(energy_HEB_dict[key]['dist'] * x_HEB[s, y, key] for key in keys_HEB if key in energy_HEB_dict)
        
        model.addConstr(
            y_HEB[s, y] * range_HEB >= total_distance_HEB, 
            name=f"C1_HEB_ge_{s}_{y}"
        )
        model.addConstr(
            y_HEB[s, y] * range_HEB < total_distance_HEB + range_HEB,
            name=f"C1_HEB_lt_{s}_{y}"
        )

# For BEB buses
for s in S:
    for y in year_keys:
        total_distance_BEB = quicksum(energy_BEB_dict[key]['dist'] * x_BEB[s, y, key] for key in keys_BEB if key in energy_BEB_dict)
        
        model.addConstr(
            y_BEB[s, y] * range_BEB >= total_distance_BEB, 
            name=f"C1_BEB_ge_{s}_{y}"
        )
        model.addConstr(
            y_BEB[s, y] * range_BEB < total_distance_BEB + range_BEB,
            name=f"C1_BEB_lt_{s}_{y}"
        )



print("Done defining constraint 1")
report_usage()


# Constraint 2: Each trip is assigned to exactly one bus
unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
for key in unique_keys:
    for s in S:
        for y in year_keys:
            model.addConstr(
                (
                    quicksum(x_CDB[s, y, key] for y in year_keys if key in energy_CDB_dict) +
                    quicksum(x_HEB[s, y, key] for y in year_keys if key in energy_HEB_dict) +
                    quicksum(x_BEB[s, y, key] for y in year_keys if key in energy_BEB_dict)
                ) == 1
            , name=f"C2_{key}_{s}_{y}")

print("Done defining constraint 2")
report_usage()

# Constraint 3: Maximum daily charging capacity
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB.sum(s, y, key) for s in S for key in keys_BEB) <= M_cap[y] for y in year_keys),
    name="C3"
)
print("Done defining constraint 3")
report_usage()

# Constraint 4: Bus Range Constraint- Add a constraint that ensures that a bus doesn't exceed its maximum range in a day
# For CDB buses
for y in year_keys:
    for s in S:
        model.addConstr(
            quicksum(energy_CDB_dict[key]['dist'] * x_CDB[s, y, key] for key in keys_CDB if key in energy_CDB_dict) <= range_CDB,
            name=f"C4_CDB_{s}_{y}"
        )

# For HEB buses
for y in year_keys:
    for s in S:
        model.addConstr(
            quicksum(energy_HEB_dict[key]['dist'] * x_HEB[s, y, key] for key in keys_HEB if key in energy_HEB_dict) <= range_HEB,
            name=f"C4_HEB_{s}_{y}"
        )

# For BEB buses
for y in year_keys:
    for s in S:
        model.addConstr(
            quicksum(energy_BEB_dict[key]['dist'] * x_BEB[s, y, key] for key in keys_BEB if key in energy_BEB_dict) <= range_BEB,
            name=f"C4_BEB_{s}_{y}"
        )
print("Done defining constraint 4")
report_usage()

# Constraint 5: Maximum yearly investment
model.addConstrs(
    (quicksum(cost_inv[('C', year)] * y_CDB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('H', year)] * y_HEB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('B', year)] * y_BEB.sum(year, '*') for year in range(y + 1)) <= M_inv[s, y]
    for y in year_keys for s in S),
    name="C5"
)
print("Done defining constraint 5")
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
df.to_csv(r'../../results/balanced-transition-highcap-optimized-variables.csv', index=False)
coeff_df.to_csv(r'../../results/variable_coefficients.csv', index=False)

end = time.time()
report_usage()
print("Total time spend in seconds",end - start)