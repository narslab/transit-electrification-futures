import pandas as pd
from pulp import LpVariable, LpMinimize, LpProblem
from pulp import lpSum
from pulp import LpStatus, value

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

# Maximum daily charging capacity in year y
M_cap = {y: val for y, val in enumerate([5600, 8400, 10500, 12950, 15400, 18900] + [float('inf')] * (Y - 6))}

# Set of scenarios
S = {'low-cap', 'mid-cap', 'high-cap'}

# Define R and Rho
R = df_CDB['Route'].nunique()
Rho = df_CDB[df_CDB['run'] != float('inf')]['run'].max()

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
energy_CDB = df_CDB.groupby(['Vehicle', 'Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_CDB['Diesel'] = energy_CDB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_CDB_dict = energy_CDB.set_index(['Vehicle', 'Date', 'Route', 'TripKey']).to_dict('index')

# For df_HEB
energy_HEB = df_HEB.groupby(['Vehicle', 'Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_HEB['Diesel'] = energy_HEB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_HEB_dict = energy_HEB.set_index(['Vehicle', 'Date', 'Route', 'TripKey']).to_dict('index')

# For df_BEB
energy_BEB = df_BEB.groupby(['Vehicle', 'Date', 'Route', 'TripKey']).agg({'Energy': 'sum', 'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = energy_BEB.apply(lambda x: x['Energy'] if x['Powertrain'] in ['conventional', 'hybrid'] else 0, axis=1)
energy_BEB_dict = energy_BEB.set_index(['Vehicle', 'Date', 'Route', 'TripKey']).to_dict('index')

# Create an LP Problem
model = LpProblem('Minimize fleet cost', LpMinimize)

# Decision variables
keys_CDB = list(energy_CDB_dict.keys())
x_CDB = LpVariable.dicts('x_CDB', keys_CDB, lowBound=0, cat='Binary')

keys_HEB = list(energy_HEB_dict.keys())
x_HEB = LpVariable.dicts('x_HEB', keys_HEB, lowBound=0, cat='Binary')

keys_BEB = list(energy_BEB_dict.keys())
x_BEB = LpVariable.dicts('x_BEB', keys_BEB, lowBound=0, cat='Binary')

# Variables for new buses each year
keys_years_scenarios = [(y, s) for y in range(Y) for s in S]
y_CDB = LpVariable.dicts('y_CDB', keys_years_scenarios, lowBound=0, cat='Integer')
y_HEB = LpVariable.dicts('y_HEB', keys_years_scenarios, lowBound=0, cat='Integer')
y_BEB = LpVariable.dicts('y_BEB', keys_years_scenarios, lowBound=0, cat='Integer')

# Define Objective Function
model += lpSum([
    cost_inv[(p, y)] * (y_CDB[(y, s)] if p == 'C' else (
        y_HEB[(y, s)] if p == 'H' else
        y_BEB[(y, s)])) for p in ['C', 'H', 'B'] for y in range(Y) for s in S
] + [
    energy_CDB_dict[key]['Diesel'] * x_CDB[key] if key in energy_CDB_dict else (
        energy_HEB_dict[key]['Diesel'] * x_HEB[key] if key in energy_HEB_dict else (
            energy_BEB_dict[key]['Diesel'] * x_BEB[key] if key in energy_BEB_dict else 0))
    for key in keys_CDB + keys_HEB + keys_BEB
])

# Define Constraints

# Constraint 1: Only one bus can be assigned to each trip (either a CDB, HEB or BEB)
for key in keys_CDB + keys_HEB + keys_BEB:
    model += x_CDB[key] + x_HEB[key] + x_BEB[key] <= 1

# Constraint 2: Total number of CDB, HEB, BEB should not exceed the total fleet size
for y in range(Y):
    for s in S:
        model += lpSum(y_CDB[(year, s)] for year in range(y + 1)) + lpSum(
            y_HEB[(year, s)] for year in range(y + 1)) + lpSum(y_BEB[(year, s)] for year in range(y + 1)) <= 1000

# Constraint 3: Maximum daily charging capacity
for d in D:
    for y in range(Y):
        for s in S:
            model += lpSum(
                energy_BEB_dict[key]['Energy'] * x_BEB[key] if key in energy_BEB_dict else 0 for key in keys_BEB if
                key[1] == d) <= M_cap[y]

# Constraint 4: Maximum yearly investment
for y in range(Y):
    for s in S:
        model += lpSum(cost_inv[('C', year)] * y_CDB[(year, s)] for year in range(y + 1)) + lpSum(
            cost_inv[('H', year)] * y_HEB[(year, s)] for year in range(y + 1)) + lpSum(
            cost_inv[('B', year)] * y_BEB[(year, s)] for year in range(y + 1)) <= M_inv[(s, y)]

# Solve the model
model.solve()

# Check status and print optimal value
print("Status:", LpStatus[model.status])
print("Optimal Cost:", value(model.objective))
