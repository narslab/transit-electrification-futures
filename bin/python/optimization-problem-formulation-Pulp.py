import pandas as pd
from pulp import GUROBI, LpVariable, LpMinimize, LpProblem, lpSum, LpStatus, value
from tqdm import tqdm  
import time  
# from pulp import PULP_CBC_CMD,
#from pulp import *

# Read dataframes of all-CDB, all-HEB, and all BEB with runs included
df_CDB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-CDB.csv', low_memory=False)
df_HEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-HEB.csv', low_memory=False)
df_BEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-BEB.csv', low_memory=False)

# Convert 'Date' column to day of the year format
df_CDB['Date'] = pd.to_datetime(df_CDB['Date']).dt.dayofyear
df_HEB['Date'] = pd.to_datetime(df_HEB['Date']).dt.dayofyear
df_BEB['Date'] = pd.to_datetime(df_BEB['Date']).dt.dayofyear

# Define parameters
D = set(df_CDB['Date'].unique())  # Create a set of unique dates
Y = 13  # Years in simulation
max_number_of_buses = 500 # 213*2 (current numnumber of fleet*2, assuming buses are going to be replaced with electric at most with ratio of 1:2)

# Maximum daily charging capacity in year y
M_cap = {y: val for y, val in enumerate([5600, 8400, 10500, 12950, 15400, 18900] + [float('inf')] * (Y - 6))}

# Set of scenarios
S = {'low-cap', 'mid-cap', 'high-cap'}

# Define R and Rho
R = df_CDB['Route'].nunique()
Rho = int(df_CDB[df_CDB['run'] != float('inf')]['run'].max())

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
    'mid-cap': 14,  # in million dollars
    'high-cap': 21  # in million dollars
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
energy_CDB['Diesel'] = energy_CDB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_CDB_dict = energy_CDB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')

# For df_HEB
energy_HEB = df_HEB.groupby(['Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_HEB['Diesel'] = energy_HEB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_HEB_dict = energy_HEB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')

# For df_BEB
energy_BEB = df_BEB.groupby(['Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = energy_BEB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_BEB_dict = energy_BEB.set_index(['Date', 'Route', 'TripKey']).to_dict('index')


# Create an LP Problem
model = LpProblem('Minimize fleet diesel consumption', LpMinimize)

# Additional keys for buses and years
bus_keys = range(max_number_of_buses)
year_keys = range(Y)

# Decision variables
keys_CDB = list(energy_CDB_dict.keys())
keys_HEB = list(energy_HEB_dict.keys())
keys_BEB = list(energy_BEB_dict.keys())

# Decision variables which include two additional indices for buses (i) and years (y)
x_CDB = LpVariable.dicts('x_CDB', (bus_keys, year_keys, keys_CDB), lowBound=0, cat='Binary')
x_HEB = LpVariable.dicts('x_HEB', (bus_keys, year_keys, keys_HEB), lowBound=0, cat='Binary')
x_BEB = LpVariable.dicts('x_BEB', (bus_keys, year_keys, keys_BEB), lowBound=0, cat='Binary')

# Define y_CDB, y_HEB, and y_BEB as the number of each type of bus at each year under each scenario
y_CDB = LpVariable.dicts('y_CDB', (year_keys, S), lowBound=0, cat='Integer')
y_HEB = LpVariable.dicts('y_HEB', (year_keys, S), lowBound=0, cat='Integer')
y_BEB = LpVariable.dicts('y_BEB', (year_keys, S), lowBound=0, cat='Integer')


# Objective function for diesel consumption
model += (
    lpSum([energy_CDB_dict[key]['Diesel'] * x_CDB[i][y][key] for key in keys_CDB for i in bus_keys for y in year_keys]) +
    lpSum([energy_HEB_dict[key]['Diesel'] * x_HEB[i][y][key] for key in keys_HEB for i in bus_keys for y in year_keys]) +
    lpSum([energy_BEB_dict[key]['Diesel'] * x_BEB[i][y][key] for key in keys_BEB for i in bus_keys for y in year_keys])
), "Total Diesel Consumption"
     
## Define Constraints

# Constraint 1: Aaccounting for the relationship between the buses purchased each year and the trips that are assigned to these new buses. 
for s in  tqdm(S):
    for y in  year_keys:
        # CDB buses
        model += lpSum(x_CDB[s][i][y][key] for i in bus_keys for key in keys_CDB if key in x_CDB[s][i][y]) == y_CDB[(y, s)]

        # HEB buses
        model += lpSum(x_HEB[s][i][y][key] for i in bus_keys for key in keys_HEB if key in x_HEB[s][i][y]) == y_HEB[(y, s)]

        # BEB buses
        model += lpSum(x_BEB[s][i][y][key] for i in bus_keys for key in keys_BEB if key in x_BEB[s][i][y]) == y_BEB[(y, s)]


# Constraint 2: The sum of decision variables for each bus and year across all powertrains should be <= 1
for i in tqdm(bus_keys):
    for y in year_keys:
        model += (
            lpSum(x_CDB[i][y][key] for key in keys_CDB if key in x_CDB[i][y]) +
            lpSum(x_HEB[i][y][key] for key in keys_HEB if key in x_HEB[i][y]) +
            lpSum(x_BEB[i][y][key] for key in keys_BEB if key in x_BEB[i][y])
        ) <= 1

# Constraint 3: Only one bus can be assigned to each trip
# Assume unique_keys are all the unique combinations of (date, route, trip)
unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
for key in tqdm(unique_keys):
    model += (
        lpSum(x_CDB[i][y][key] for i in bus_keys for y in year_keys if key in x_CDB[i][y]) +
        lpSum(x_HEB[i][y][key] for i in bus_keys for y in year_keys if key in x_HEB[i][y]) +
        lpSum(x_BEB[i][y][key] for i in bus_keys for y in year_keys if key in x_BEB[i][y])
    ) <= 1

# Constraint 4: Total number of CDB, HEB, BEB should not exceed the total fleet size
# Assume max_buses_per_year is a constant limit
max_buses_per_year = max_number_of_buses
for y in tqdm(year_keys):
    model += (
        lpSum(y_CDB[(year, s)] for year in range(y + 1) for s in S) +
        lpSum(y_HEB[(year, s)] for year in range(y + 1) for s in S) +
        lpSum(y_BEB[(year, s)] for year in range(y + 1) for s in S)
    ) <= max_buses_per_year

# Constraint 5: Maximum daily charging capacity
for d in tqdm(D):
    for y in year_keys:
        model += (
            lpSum(energy_BEB_dict[key]['Energy'] * x_BEB[i][y][key] if key in x_BEB[i][y] else 0 for i in bus_keys for key in keys_BEB if key[1] == d)
        ) <= M_cap[y]

# Constraint 6: Maximum yearly investment
for y in year_keys:
    for s in S:
        model += (
            lpSum(cost_inv[('C', year)] * y_CDB[(year, s)] for year in range(y + 1)) +
            lpSum(cost_inv[('H', year)] * y_HEB[(year, s)] for year in range(y + 1)) +
            lpSum(cost_inv[('B', year)] * y_BEB[(year, s)] for year in range(y + 1))
        ) <= M_inv[(s, y)]
        
        
# Print model statistics
print("Number of variables: ", len(model.variables()))
print("Number of constraints: ", len(model.constraints))
                
# Debug the model
model.debug = True

# Solve the model
start_time = time.time()  # get the current time
#model.solve()  

# Use the msg parameter in the solve function to see the solver logs.
#model.solve(PULP_CBC_CMD(msg=1))  # For PuLP's built-in CBC solver
model.solve(GUROBI(msg=1))
print(f"Time taken by model.solve(): {time.time() - start_time} seconds")  # print the time difference


# Check status and print optimal value
print("Status:", LpStatus[model.status])
print("Optimal Cost:", value(model.objective))

df = pd.DataFrame(columns=["Variable", "Value"])

# Print optimal decision variables
for variable in model.variables():
    df = df.append({"Variable": variable.name, "Value": variable.varValue}, ignore_index=True)
    print("{} = {}".format(variable.name, variable.varValue))
    
# Append optimal objective value to DataFrame
df = df.append({"Variable": "Optimal Cost", "Value": value(model.objective)}, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(r'../../results/strategies-simulation-optimized-variables.csv', index=False)
