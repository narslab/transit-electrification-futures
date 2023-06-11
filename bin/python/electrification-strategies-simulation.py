# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:13:37 2023

@author: Mahsa
"""

import pandas as pd
from pyomo.environ import *

# Filter the dataframe for the conventional powertrain
subset_df = df[df['Powertrain'] == 'conventional']

# Calculate the average daily energy consumption for each route
daily_energy = subset_df.groupby(['ServiceDateTime', 'Route'])['Energy'].sum().reset_index()
average_daily_energy = daily_energy.groupby('Route')['Energy'].mean().to_dict()

# Set up the Pyomo optimization model
model = ConcreteModel()

# Define decision variable
model.route = Var(average_daily_energy.keys(), within=Binary)

# Define objective function
model.energy_consumption = Objective(expr=sum(model.route[r] * average_daily_energy[r] for r in average_daily_energy.keys()), sense=minimize)

# Add constraints
n_routes = 87
model.route_selection = Constraint(expr=sum(model.route[r] for r in average_daily_energy.keys()) >= 1)
model.number_of_routes = Constraint(expr=sum(model.route[r] for r in average_daily_energy.keys()) == n_routes)

# Solve the optimization model using the GLPK solver
solver = SolverFactory('glpk')
solver.solve(model)

# Extract the selected routes and return them
def get_selected_routes(model):
    selected_routes = [r for r in model.route if model.route[r].value == 1]
    return selected_routes

selected_routes = get_selected_routes(model)
print("Selected Routes:", selected_routes)

