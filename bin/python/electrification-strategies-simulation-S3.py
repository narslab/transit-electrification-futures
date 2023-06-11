from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd
import energy_model_rb as emrb  

# Create the problem
problem = LpProblem(name="bus-optimization-problem", sense=LpMinimize)

# Define indexes
years = list(range(0, 14))
bus_types = ['BEB', 'HEB', 'CDB']

# Initialize the decision variables
x = {y: {b: LpVariable(name=f"x_{y}_{b}", lowBound=0, cat="Integer") for b in bus_types} for y in years}

# Define parameters
bus_costs = {'BEB':13000000, 'HEB':600000, 'CDB':0}
total_cost_limit = 40000000
charging_constraints = {1:8, 2:0, 3:4, 4:11, 5:4, 6:10}

# Initialize year 0 buses
x[0]['CDB'] = LpVariable(name="x_0_CDB", lowBound=189, upBound=189, cat='Integer')
x[0]['HEB'] = LpVariable(name="x_0_HEB", lowBound=9, upBound=9, cat='Integer')
x[0]['BEB'] = LpVariable(name="x_0_BEB", lowBound=15, upBound=15, cat='Integer')

# Add the objective function to the problem
problem += lpSum(0*x[y]['BEB'] + 3982*x[y]['HEB'] + 9714*(213 - (x[y]['BEB'] + x[y]['HEB'])) for y in years if y != 0)

# Add the charging constraints for new buses
for y in range(1, 7):
    problem += (x[y]['BEB'] - x[y-1]['BEB']) <= charging_constraints[y]

# Add the cost constraint for new buses
for y in range(1, 14):
    problem += bus_costs['HEB']*(x[y]['HEB'] - x[y-1]['HEB']) + bus_costs['BEB']*(x[y]['BEB'] - x[y-1]['BEB']) <= total_cost_limit

# Add the bus total constraint for each year
for y in years:
    problem += x[y]['BEB'] + x[y]['HEB'] + x[y]['CDB'] == 213 

# Add the CDB to HEB transition constraint
problem += x[13]['HEB'] == 0  # convert all HEB to BEB by the 13th year
problem += x[13]['BEB'] == 213   # convert all CDB and HEB to BEB by the 13th year
problem += x[13]['CDB'] == 0    # convert all CDB to BEB by the 13th year



# Solve the problem
status = problem.solve()

# Print the changes in number of buses each year
if LpStatus[status] == "Optimal":
    for y in years[1:]:  # We start from year 1 as there's no previous year for year 0
        for b in bus_types:
            change = x[y][b].value() - x[y-1][b].value()
            print(f"Change in number of {b} buses from year {y-1} to {y}: {change}")


### Change the fuel rate based on replaced fleet
# Read fuel rates
df = pd.read_csv(r'../../results/computed-fuel-rates-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)

# Create a DataFrame from the purchase plan
purchase_plan = pd.DataFrame({
    'Year': [2023, 2024],
    '#Purchased': [66, 57],
    'Powertrain': ['hybrid', 'hybrid'],
    'VehicleModel': ['NEW FLYER XDE40', 'NEW FLYER XDE40'],
    'VehiclWeight(lb)': [28250, 28250]
})

# Filter for conventional vehicles
df_conventional = df[df['Powertrain'] == 'conventional']

# Group by 'Vehicle' and sum 'Energy' column
energy_consumption = df_conventional.groupby('Vehicle')['Energy'].sum().reset_index()

# Sort the resulting DataFrame by 'Energy' column in descending order
energy_consumption_sorted = energy_consumption.sort_values('Energy', ascending=False).reset_index(drop=True)

# Initialize empty DataFrame to store vehicles to replace each year
vehicles_to_replace = pd.DataFrame(columns=['Year', 'Vehicle', 'Energy'])

# For each year in purchase plan, find vehicles to replace
for index, row in purchase_plan.iterrows():
    year = row['Year']
    num_purchased = row['#Purchased']
    for _ in range(num_purchased):
        if not energy_consumption_sorted.empty:
            vehicle_to_replace = pd.DataFrame(energy_consumption_sorted.iloc[0:1].copy()) # select first row as dataframe
            vehicle_to_replace['Year'] = year
            vehicles_to_replace = pd.concat([vehicles_to_replace, vehicle_to_replace], ignore_index=True)
            energy_consumption_sorted = energy_consumption_sorted.iloc[1:]

#print(vehicles_to_replace)
vehicles_to_replace.to_csv(r'../../results/S3-vehicles_to_replace.csv')

# Read trajectories
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)

# Read parameters
parameters_filename='params-oct2021-sep2022.yaml'

# Loop over the years in vehicles_to_replace DataFrame
for year in vehicles_to_replace['Year'].unique():
    # Get all vehicles to replace for this year
    vehicles_for_year = vehicles_to_replace[vehicles_to_replace['Year'] == year]

    # Update the Powertrain, VehicleModel, and VehicleWeight in df_trajectories for this year's vehicles
    for index, row in vehicles_for_year.iterrows():
        purchase_row = purchase_plan[purchase_plan['Year'] == year].iloc[0]
        df_trajectories.loc[df_trajectories['Vehicle'] == row['Vehicle'], 'Powertrain'] = purchase_row['Powertrain']
        df_trajectories.loc[df_trajectories['Vehicle'] == row['Vehicle'], 'VehicleModel'] = purchase_row['VehicleModel']
        df_trajectories.loc[df_trajectories['Vehicle'] == row['Vehicle'], 'VehiclWeight(lb)'] = purchase_row['VehiclWeight(lb)']
    
    # Compute energy for each year and save it
    df_input = df_trajectories.copy()
    output_csv_filename = f'computed-fuel-rates-S3-{year}.csv'
    emrb.compute_energy(parameters_filename, df_input, output_csv_filename)