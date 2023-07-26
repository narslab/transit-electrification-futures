import pandas as pd
import energy_model_whole_year as em  
#import energy_model_rb as em  


# Read trajectories
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)

# Read parameters
#parameters_filename='params-oct2021-sep2022.yaml'
parameters_filename='params.yaml'

### CDB
df_input = df_trajectories.copy()

# Change all vehicles to conventional 
df_input['Powertrain'] = 'conventional'
df_input['VehicleModel'] = 'NEW FLYER XD40'
df_input['VehicleWeight(lb)'] = 27180
print('CDB vehivle model', df_input['VehicleModel'].unique())

# Compute energy for all CDBs
output_csv_filename = r'../../results/computed-fuel-rates-all-CDB.csv'
em.compute_energy(parameters_filename, df_input, output_csv_filename)
print("Done all CDBS")

### HEB
df_input = df_trajectories.copy()

# Change all vehicles to hybrid 
df_input['Powertrain'] = 'hybrid'
df_input['VehicleModel'] = 'NEW FLYER XDE40'
df_input['VehicleWeight(lb)'] = 28250
print('HEB vehivle model', df_input['VehicleModel'].unique())

# Compute energy for all hybrid
output_csv_filename = r'../../results/computed-fuel-rates-all-HEB.csv'
em.compute_energy(parameters_filename, df_input, output_csv_filename)
print("Done all HEBS")

### BEB
df_input = df_trajectories.copy()

# Change all vehicles to electric 
df_input['Powertrain'] = 'electric'
df_input['VehicleModel'] = 'NEW FLYER XE40'
df_input['VehicleWeight(lb)'] = 32770
print('BEB vehivle model', df_input['VehicleModel'].unique())

# Compute energy for all electric
output_csv_filename = r'../../results/computed-fuel-rates-all-BEB.csv'
em.compute_energy(parameters_filename, df_input, output_csv_filename)