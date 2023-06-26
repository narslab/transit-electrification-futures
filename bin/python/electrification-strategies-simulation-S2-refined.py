import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

# Read df_all_equal_powertrain energy consumption
df_conventional= pd.read_csv(r'../../results/computed-fuel-rates-all-CDB.csv', delimiter=',', skiprows=0, low_memory=False)
df_hybrid= pd.read_csv(r'../../results/computed-fuel-rates-all-HEB.csv', delimiter=',', skiprows=0, low_memory=False)
df_electric= pd.read_csv(r'../../results/computed-fuel-rates-all-BEB.csv', delimiter=',', skiprows=0, low_memory=False)

# Calculate total energy per vehicle for each DataFrame
total_energy_conventional = df_conventional.groupby('Vehicle')['Energy'].sum().reset_index()
total_energy_hybrid = df_hybrid.groupby('Vehicle')['Energy'].sum().reset_index()
total_energy_electric = df_electric.groupby('Vehicle')['Energy'].sum().reset_index()

# Get list of vehicles
vehicles = pd.concat([total_energy_conventional['Vehicle'], total_energy_hybrid['Vehicle'], total_energy_electric['Vehicle']]).unique()

# Get list of powertrains
powertrains = ['conventional', 'hybrid', 'electric']

# Create a binary variable for each vehicle*powertrain combination
variables = LpVariable.dicts("Choice",(powertrains, vehicles),0,1,LpBinary)

# Create the 'prob' variable to contain the problem data
prob = LpProblem("Vehicle Powertrain Problem", LpMinimize)

# The objective function is added to 'prob' first
prob += lpSum([variables[powertrain][vehicle] * energy 
               for powertrain, df in zip(powertrains, [total_energy_conventional, total_energy_hybrid, total_energy_electric])
               for vehicle, energy in zip(df['Vehicle'], df['Energy'])]), "Total Energy of Vehicles"
