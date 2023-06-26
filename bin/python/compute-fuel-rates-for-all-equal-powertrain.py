import pandas as pd
import energy_model_rb as emrb  

# Read trajectories
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)

# Change all vehicles to hybrid 
df_trajectories['Powertrain'] = 'hybrid'
df_trajectories['VehicleModel'] = 'NEW FLYER XDE40'
df_trajectories['VehicleWeight(lb)'] = 28250


# Read parameters
parameters_filename='params-oct2021-sep2022.yaml'

# Compute energy for all hybrid
df_input = df_trajectories.copy()
output_csv_filename = r'../../results/computed-fuel-rates-all-HEB.csv'
emrb.compute_energy(parameters_filename, df_input, output_csv_filename)

# Change all vehicles to hybrid 
df_trajectories['Powertrain'] = 'conventional'
df_trajectories['VehicleModel'] = 'NEW FLYER XD40'
df_trajectories['VehicleWeight(lb)'] = 28250


# Compute energy for all conventional
df_input = df_trajectories.copy()
output_csv_filename = r'../../results/computed-fuel-rates-all-CDB.csv'
emrb.compute_energy(parameters_filename, df_input, output_csv_filename)

# Change all vehicles to electric 
df_trajectories['Powertrain'] = 'electric'
df_trajectories['VehicleModel'] = 'NEW FLYER XE40'
df_trajectories['VehicleWeight(lb)'] = 32770


# Compute energy for all electric
df_input = df_trajectories.copy()
output_csv_filename = r'../../results/computed-fuel-rates-all-BEB.csv'
emrb.compute_energy(parameters_filename, df_input, output_csv_filename)