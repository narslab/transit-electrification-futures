import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time
import psutil
import os

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

# Sample 7 random dates
#random_dates = df_CDB['Date'].sample(n=7, random_state=1).values

# Find the date with the maximum number of unique trips
date_with_max_trips = df_CDB.groupby('Date')['TripKey'].nunique().idxmax()
max_trips = df_CDB.groupby('Date')['TripKey'].nunique().max()
print("date_with_max_trips is:", date_with_max_trips)
print("max_trips is:", max_trips)
report_usage()

# Filter dataframes by random dates
#df_CDB = df_CDB[df_CDB['Date'].isin(date_with_max_trips)]
#df_HEB = df_HEB[df_HEB['Date'].isin(date_with_max_trips)]
#df_BEB = df_BEB[df_BEB['Date'].isin(date_with_max_trips)]
df_CDB = df_CDB.loc[df_CDB['Date']==date_with_max_trips]
df_HEB = df_HEB.loc[df_HEB['Date']==date_with_max_trips]
df_BEB = df_BEB.loc[df_BEB['Date']==date_with_max_trips]
report_usage()


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
#C_max = {
#    'low-cap': 7,  # in million dollars
#    'mid-cap': 14,  # in million dollars
#    'high-cap': 21  # in million dollars
#}

C_max = {
    'low-cap': 7,  # in million dollars
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

# Groupby to compute energy consumption for each unique vehicle, date, route, and trip key
# then, create the 'Diesel' column based on the condition for 'Powertrain'

# For df_CDB
energy_CDB = df_CDB.groupby(['Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_CDB['Diesel'] = (energy_CDB['Powertrain'].isin(['conventional', 'hybrid']) * energy_CDB['Energy'])
#energy_CDB_dict = energy_CDB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')
energy_CDB_dict = energy_CDB.set_index(['TripKey']).to_dict('index')

# For df_HEB
energy_HEB = df_HEB.groupby(['Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_HEB['Diesel'] = (energy_HEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_HEB['Energy'])
#energy_HEB_dict = energy_HEB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')
energy_HEB_dict = energy_HEB.set_index(['TripKey']).to_dict('index')

# For df_BEB
energy_BEB = df_BEB.groupby(['Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = (energy_BEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_BEB['Energy'])
#energy_BEB_dict = energy_BEB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')
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

# Set heuristic parameters
#model.setParam('MIPFocus', 1)  # This parameter lets control the MIP solver's focus. The options are: 1. Finds feasible solutions quickly. This is useful if the problem is difficult to solve and you're satisfied with any feasible solution. 2: Works to improve the best bound. This is useful when the best objective bound is poor. 3: Tries to prove optimality of the best solution found. This is useful if you're sure a nearly-optimal solution exists and you want the solver to focus on proving its optimality.
model.setParam('Heuristics', 0.1)  # Controls the effort put into MIP heuristics (range is 0 to 1). A higher value means more effort is put into finding solutions, but at the cost of slower overall performance. 
#model.setParam('Cuts', 2)  # This parameter controls the aggressiveness of cut generation. Cutting planes are additional constraints that can potentially improve the LP relaxation of the problem, thus leading to a quicker solution. A higher value means more aggressive cut generation, but this could potentially slow down the solver because of the extra overhead.
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
#print("keys_CDB",keys_CDB)
print("len_keys_CDB",len(keys_CDB))
#print("keys_HEB",keys_HEB)
print("len_keys_HEB",len(keys_HEB))
#print("keys_BEB",keys_BEB)
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

model.setObjective(
(quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[s, i, y, key] for s in S for key in keys_CDB for i in bus_keys for y in year_keys if key in energy_CDB_dict]) +
 quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[s, i, y, key] for s in S for key in keys_HEB for i in bus_keys for y in year_keys if key in energy_HEB_dict]) +
 quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[s, i, y, key] for s in S for key in keys_BEB for i in bus_keys for y in year_keys if key in energy_BEB_dict])),
    GRB.MINIMIZE
)

print("Done setting objective function")
report_usage()
     
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
    (y_BEB[s, y] == quicksum(x_BEB[s, i, y, key] for i in bus_keys for key in keys_CDB) for s in S for y in year_keys)
,
    name="C1_BEB"
)
### Aditional explanation: 
#x_CDB[i, y, d, r, rho] >= 1 is a binary condition that checks if bus 'i' is used at least once in a trip during year 'y'. This will return True (or 1) if bus 'i' is used, and False (or 0) otherwise. 
#Regardless of the values of d, r, and rho, the result of this expression is either 1 (if the bus 'i' is used at least once) or 0 (if the bus 'i' is not used at all).
print("Done defining constraint 1")
report_usage()

# Constraint 2: The sum of decision variables for each bus and year across all powertrains should be <= 1
model.addConstrs(
    (z_CDB[s, i, y] + z_HEB[s, i, y] + z_BEB[s, i, y] <= 1 for s in S for i in bus_keys for y in year_keys),
    name="C2"
)
print("Done defining constraint 2")
report_usage()


# Constraint 3: Each trip is assigned to exactly one bus
unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
model.addConstrs(
    (
        (
            quicksum(x_CDB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_CDB_dict) +
            quicksum(x_HEB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_HEB_dict) +
            quicksum(x_BEB[s, i, y, key] for s in S for i in bus_keys for y in year_keys if key in energy_BEB_dict)
        ) == 1 for key in unique_keys
    ), 
    name="C3"
)
print("Done defining constraint 3")
report_usage()

# Constraint 4: Maximum daily charging capacity
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB.sum(s, i, y, key) for s in S for i in bus_keys for key in keys_BEB) <= M_cap[y] for y in year_keys),
    name="C4"
)
print("Done defining constraint 4")
report_usage()

# Constraint 5: Daily energy consumption by each BEB should not exceed its battery capacity
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB[s, i, y, key] for s in S for i in bus_keys for key in keys_BEB) <= cap for y in year_keys),
    name="C5"
)
print("Done defining constraint 5")
report_usage()

# Constraint 6: Maximum yearly investment
model.addConstrs(
    (quicksum(cost_inv[('C', year)] * y_CDB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('H', year)] * y_HEB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('B', year)] * y_BEB.sum(year, '*') for year in range(y + 1)) <= M_inv[s, y]
    for y in year_keys for s in S),
    name="C6"
)
print("Done defining constraint 6")
report_usage()

# Print model statistics
#print("Number of variables:", model.NumVars)
#print("Number of binary variables:", model.NumBinVars)
#print("Number of integer variables:", model.NumIntVars)
#print("Number of constraints:", model.NumConstrs)
#print("Number of non-zero coefficients:", model.NumNZs)
#report_usage()

# Print model statistics
model.update()
print(model)
        
# Tuning and Optimization
model.tune()
model.optimize()
report_usage()


# Save the DataFrame to a CSV file
#df.to_csv(r'../../results/strategies-simulation-optimized-variables.csv', index=False)

# Prepare dictionaries of coeesicients to save
coeff_dict_CDB = {(s, i, y, key): energy_CDB_dict[key]['Diesel'] for s in S for key in keys_CDB for i in bus_keys for y in year_keys if key in energy_CDB_dict}
coeff_dict_HEB = {(s, i, y, key): energy_HEB_dict[key]['Diesel'] for s in S for key in keys_HEB for i in bus_keys for y in year_keys if key in energy_HEB_dict}
coeff_dict_BEB = {(s, i, y, key): energy_BEB_dict[key]['Diesel'] for s in S for key in keys_BEB for i in bus_keys for y in year_keys if key in energy_BEB_dict}

# Combine all the dictionaries into a dataframe
coeff_df = pd.DataFrame(list(coeff_dict_CDB.items()) + list(coeff_dict_HEB.items()) + list(coeff_dict_BEB.items()), columns=['Variable', 'Coefficient'])


vars = model.getVars()

# Create DataFrame directly from the variables and their values
df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.x for v in vars]})

# Get optimal objective value
optimal_value = model.ObjVal

# Add this value to DataFrame
df_objective = pd.DataFrame({"Year": year_keys, "Objective_Value": [optimal_value]*len(year_keys)})

# Print optimal decision variables
#print(df.to_string(index=False))

# Print objective value
print("optimal_value:",optimal_value)

# Save the DataFrame to a CSV file
df.to_csv(r'../../results/strategies-simulation-optimized-variables.csv', index=False)
#optimal_value.to_csv(r'../../results/strategies-simulation-optimized-objective.csv', index=False)
coeff_df.to_csv(r'../../results/variable_coefficients.csv', index=False)

end = time.time()
report_usage()
print("Total time spend in seconds",end - start)