import pandas as pd
from gurobipy import Model, GRB, quicksum
from tqdm import tqdm  
import time  

# Read dataframes of all-CDB, all-HEB, and all BEB with runs included
df_CDB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-CDB.csv', low_memory=False)
df_HEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-HEB.csv', low_memory=False)
df_BEB = pd.read_csv(r'../../results/computed-fuel-rates-runs-all-BEB.csv', low_memory=False)

# Convert 'Date' column to day of the year format
df_CDB['Date'] = pd.to_datetime(df_CDB['Date']).dt.dayofyear
df_HEB['Date'] = pd.to_datetime(df_HEB['Date']).dt.dayofyear
df_BEB['Date'] = pd.to_datetime(df_BEB['Date']).dt.dayofyear

# Sample 30 random dates
random_dates = df_CDB['Date'].sample(n=30, random_state=1).values

# Filter dataframes by random dates
df_CDB = df_CDB[df_CDB['Date'].isin(random_dates)]
df_HEB = df_HEB[df_HEB['Date'].isin(random_dates)]
df_BEB = df_BEB[df_BEB['Date'].isin(random_dates)]

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


# Create a model
model = Model('Minimize fleet diesel consumption')

# Enable logging and print progress
model.setParam('OutputFlag', 1)

# Set heuristic parameters
model.setParam('MIPFocus', 1)  # 1: Focuses on finding feasible solutions. 2: focuses on improving the optimality of the best solution found. 3: focuses on proving the optimality of the best solution.
model.setParam('Heuristics', 0.5)  # Spend more time on heuristics (range is 0 to 1, with 1 being the most time)
model.setParam('Cuts', 2)  # Use more aggressive cutting planes. This parameter lets you modify the aggressiveness with which Gurobi generates cutting planes. These are additional constraints that can potentially improve the LP relaxation of the problem, leading to a quicker solution.
model.setParam('Presolve', 2)  # This parameter controls the aggressiveness of presolve reductions.  
model.setParam('Threads', 8)  # Set number of threads to be used for parallel processing.


# Additional keys for buses and years
bus_keys = range(max_number_of_buses)
year_keys = range(Y)

# Decision variables
keys_CDB = list(energy_CDB_dict.keys())
keys_HEB = list(energy_HEB_dict.keys())
keys_BEB = list(energy_BEB_dict.keys())

# Decision variables which include two additional indices for buses (i) and years (y)
x_CDB = model.addVars(bus_keys, year_keys, keys_CDB, vtype=GRB.BINARY, name='x_CDB')
x_HEB = model.addVars(bus_keys, year_keys, keys_HEB, vtype=GRB.BINARY, name='x_HEB')
x_BEB = model.addVars(bus_keys, year_keys, keys_BEB, vtype=GRB.BINARY, name='x_BEB')

# Define y_CDB, y_HEB, and y_BEB as the number of each type of bus at each year under each scenario
y_CDB = model.addVars(year_keys, S, vtype=GRB.INTEGER, name='y_CDB')
y_HEB = model.addVars(year_keys, S, vtype=GRB.INTEGER, name='y_HEB')
y_BEB = model.addVars(year_keys, S, vtype=GRB.INTEGER, name='y_BEB')

# Objective function for diesel consumption
#model.setObjective(
#    quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[i, y, key] for key in keys_CDB for i in bus_keys for y in year_keys]) +
#    quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[i, y, key] for key in keys_HEB for i in bus_keys for y in year_keys]) +
#    quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[i, y, key] for key in keys_BEB for i in bus_keys for y in year_keys]),
#    GRB.MINIMIZE
#)

# Objective function for diesel consumption
model.setObjective(
    12 * (quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[i, y, key] for key in keys_CDB for i in bus_keys for y in year_keys]) +
    quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[i, y, key] for key in keys_HEB for i in bus_keys for y in year_keys]) +
    quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[i, y, key] for key in keys_BEB for i in bus_keys for y in year_keys])),
    GRB.MINIMIZE
)

     
## Define Constraints

# Constraint 1: Accounting for the relationship between the buses purchased each year and the trips that are assigned to these new buses. 
model.addConstrs(
    (x_CDB.sum(i, y, '*') <= y_CDB.sum(y, '*') for i in bus_keys for y in year_keys for s in S),
    name="C1_CDB"
)
model.addConstrs(
    (x_HEB.sum(i, y, '*') <= y_HEB.sum(y, '*') for i in bus_keys for y in year_keys for s in S),
    name="C1_HEB"
)
model.addConstrs(
    (x_BEB.sum(i, y, '*') <= y_BEB.sum(y, '*') for i in bus_keys for y in year_keys for s in S),
    name="C1_BEB"
)

# Constraint 2: The sum of decision variables for each bus and year across all powertrains should be <= 1
model.addConstrs(
    (x_CDB.sum(i, y, '*') + x_HEB.sum(i, y, '*') + x_BEB.sum(i, y, '*') <= 1 for i in bus_keys for y in year_keys),
    name="C2"
)

# Constraint 3: Only one bus can be assigned to each trip
unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
model.addConstrs(
    (x_CDB.sum('*', '*', key) + x_HEB.sum('*', '*', key) + x_BEB.sum('*', '*', key) <= 1 for key in unique_keys),
    name="C3"
)

# Constraint 4: Total number of CDB, HEB, BEB should not exceed the total fleet size
model.addConstrs(
    (y_CDB.sum(y, '*') + y_HEB.sum(y, '*') + y_BEB.sum(y, '*') <= max_number_of_buses for y in year_keys),
    name="C4"
)

# Constraint 5: Maximum daily charging capacity
model.addConstrs(
    (quicksum(energy_BEB_dict[key]['Energy'] * x_BEB.sum(i, y, key) for i in bus_keys for key in keys_BEB if key[0] == d) <= M_cap[y] for d in D for y in year_keys),
    name="C5"
)


# Constraint 6: Maximum yearly investment
model.addConstrs(
    (quicksum(cost_inv[('C', year)] * y_CDB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('H', year)] * y_HEB.sum(year, '*') for year in range(y + 1)) +
    quicksum(cost_inv[('B', year)] * y_BEB.sum(year, '*') for year in range(y + 1)) <= M_inv[s, y]
    for y in year_keys for s in S),
    name="C6"
)

        
# Tuning and Optimization
model.tune()
model.optimize()

# Print optimal decision variables
df = pd.DataFrame(columns=["Variable", "Value"])
for v in model.getVars():
    df = df.append({"Variable": v.varName, "Value": v.x}, ignore_index=True)
    print(f'{v.varName} = {v.x}')

# Save the DataFrame to a CSV file
df.to_csv(r'../../results/strategies-simulation-optimized-variables.csv', index=False)