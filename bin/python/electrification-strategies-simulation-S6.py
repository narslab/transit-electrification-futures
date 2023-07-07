import pandas as pd
import energy_model_rb as emrb  

# Read fuel rates
df = pd.read_csv(r'../../results/computed-fuel-rates-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)

# Create a DataFrame from the purchase plan
purchase_plan = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2025, 2026, 2027, 2027, 2028],
    '#Purchased': [8, 16, 10, 4, 11, 4, 4, 10],
    'Powertrain': ['electric', 'hybrid', 'hybrid', 'electric', 'electric', 'hybrid', 'electric', 'electric'],
    'VehicleModel': ['NEW FLYER XE40', 'NEW FLYER XDE40', 'NEW FLYER XDE40', 'NEW FLYER XE40', 'NEW FLYER XE40', 'NEW FLYER XDE40', 'NEW FLYER XE40', 'NEW FLYER XE40'],
    'VehiclWeight(lb)': [32770, 28250, 28250, 32770, 32770, 28250, 32770, 32770]
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
            vehicle_to_replace = energy_consumption_sorted.iloc[0].copy()
            vehicle_to_replace['Year'] = year
            vehicles_to_replace = vehicles_to_replace.append(vehicle_to_replace, ignore_index=True)
            #vehicles_to_replace = pd.concat([vehicles_to_replace, vehicle_to_replace], ignore_index=True)
            energy_consumption_sorted = energy_consumption_sorted.iloc[1:]

#print(vehicles_to_replace)
vehicles_to_replace.to_csv(r'../../results/S6-vehicles_to_replace.csv')

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
    output_csv_filename = f'computed-fuel-rates-S6-{year}.csv'
    emrb.compute_energy(parameters_filename, df_input, output_csv_filename)

