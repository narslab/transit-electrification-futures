import yaml
import pandas as pd
import numpy as np
import gc   # For manual garbage collection

f = open('params-oct2021-sep2022-new-equation-12212023.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Read only required columns for computation from trajectories df. Change this to the actual column names you use.
df = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)


df.speed = df.speed *0.44704 # Convert from mph to m/s
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)

# Remove accelerations other than -5 to 3 m/s^2
#total_rows = len(df)
#df = df[(df['Acceleration'] >= -5) & (df['Acceleration'] <= 3)]
# Calculating the 1st and 99th percentiles
quantile_1 = df['Acceleration'].quantile(0.0001)
quantile_99 = df['Acceleration'].quantile(0.9999)
#quantile_1 = df['Acceleration'].quantile(0.01)
#quantile_99 = df['Acceleration'].quantile(0.99)

# Printing only the 1st and 99th percentile values
print("quantile_1",quantile_1)
print("quantile_99",quantile_99)

# Trimming the data
df = df[(df['Acceleration'] >= quantile_1) & (df['Acceleration'] <= quantile_99)]


#remaining_rows = len(df)
#removed_rows = total_rows - remaining_rows
#removed_percentage = (removed_rows / total_rows) * 100
#print(f"Percentage of removed data: {removed_percentage:.2f}%")


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_electric=df.loc[df['Powertrain'] == 'electric'].copy()
print('Done with reading dataframes for CDB, HEB, and BEB')

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
A_f_cdb = p.frontal_area_cdb
A_f_heb = p.frontal_area_heb
A_f_beb = p.frontal_area_beb
g = p.gravitational_acceleration
C_r1 = p.rolling_resistance_coef1
C_r2 = p.rolling_resistance_coef2
eta_d_cdb = p.driveline_efficiency_d_dis
eta_d_heb = p.driveline_efficiency_d_dis
eta_d_beb = p.driveline_efficiency_d_beb
eta_batt = p.battery_efficiency
eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
gamma_beb=p.gamma

# Define power function for diesel vehicle
def power(df_input, hybrid=False, electric=False):
    if hybrid == True:
       A=A_f_heb
       eta_d = eta_d_heb
    elif electric == True:
        A=A_f_beb
        eta_d = eta_d_beb
    else:
       A=A_f_cdb 
       eta_d = eta_d_cdb
       
    df = df_input
    v = df.Speed
    a = df.Acceleration
    G = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    C_h = 1 - (0.085*H)
    #P_t = (v/float(1000*eta_d))*(rho*C_D*C_h*A*v*v +m(g(C_r1*v+C_r2+G)+1.2*a))
    P_t = (v / (1000 * eta_d)) * (rho * C_D * C_h * A * v * v + m * (g * (C_r1 * v + C_r2 + G) + 1.2 * a))

    return P_t

# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, hybrid=False):
	# Estimates fuel consumed (liters per second) 
    if hybrid == True:
        a0 = a0_heb 
        a1 = a1_heb 
        a2 = a2_heb 
    else:
        a0 = a0_cdb
        a1 = a1_cdb
        a2 = a2_cdb
    P_t = power(df_input, hybrid)
    FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    FC_t = fuelRate_d(df_input, hybrid)
    E_t = (FC_t * t)/3.78541 # to convert liters to gals
    return E_t


# # Define Energy consumption function for electric vehicle
# def energyConsumption_e(df_input, electric=True):
# 	# Estimates energy consumed (KWh)     
#      df = df_input
#      t = df.time_delta_in_seconds
#      P_t = power(df_input, electric)
#      #a = df.Acceleration
#      print("gamma_beb", gamma_beb)
#      eta_rb = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma_beb/abs(a))))
#      #eta_rb = df.Acceleration.apply(lambda a: 0 if a >= 0 else np.exp(-(gamma_beb/abs(a))))
#      #eta_rb = [0 if p >= 0 else np.exp(-(gamma_beb/abs(a_val))) for p, a_val in zip(P_t, a)]
#      #print("eta_rb", eta_rb)
#      #print("acceleratin", a)
#      E_t = t * P_t * eta_rb * eta_batt /(eta_m*3600)
#      return E_t

def energyConsumption_e(df_input, electric=True):
     # Estimates energy consumed (KWh)
     df = df_input
     t = df.time_delta_in_seconds
     P_t = power(df_input, electric) # Assuming this returns a Series of the same length as df

     # Ensure P_t is in the DataFrame for easy access
     df['P_t'] = P_t

     # Updated eta_rb calculation
     def calculate_eta_rb(row):
         if row['P_t'] >= 0 and row['Acceleration'] >= 0:
             return 1
         elif row['P_t'] < 0 and row['Acceleration'] >= 0:
             return 0
         else:
             return np.exp(-(gamma_beb / abs(row['Acceleration'])))

     eta_rb = df.apply(calculate_eta_rb, axis=1)

     E_t = t * df['P_t'] * eta_rb * eta_batt / (eta_m * 3600)
     return E_t

# def energyConsumption_e(df_input, electric=True):
#     # Estimates energy consumed (KWh)
#     df = df_input
#     t = df.time_delta_in_seconds
#     P_t = power(df_input, electric) # Assuming this returns a Series of the same length as df

#     # Ensure P_t is in the DataFrame for easy access
#     df['P_t'] = P_t

#     # Updated eta_rb calculation
#     def calculate_eta_rb(row):
#         if row['P_t'] >= 0:
#             return 1
#         elif row['P_t'] < 0 and row['Acceleration'] >= 0:
#             return 0
#         else:
#             return np.exp(-(gamma_beb / abs(row['Acceleration'])))

#     eta_rb = df.apply(calculate_eta_rb, axis=1)

#     E_t = t * df['P_t'] * eta_rb * eta_batt / (eta_m * 3600)
#     return E_t

# def energyConsumption_e(df_input, electric=True):
#     # Estimates energy consumed (KWh)
#     df = df_input

#     # Ensure P_t is in the DataFrame for easy access
#     df = df_input.copy()
#     df['P_t'] = power(df, electric)  # Assuming this returns a Series

#     # Updated eta_rb calculation
#     def calculate_eta_rb(row):
#         if row['P_t'] >= 0:
#             return 1
#         elif row['P_t'] < 0 and row['Acceleration'] >= 0:
#             return 0
#         else:
#             return np.exp(-(gamma_beb / abs(row['Acceleration'])))

#     eta_rb = df.apply(calculate_eta_rb, axis=1)

#     # Calculate E_t based on the updated cases
#     E_t = df['time_delta_in_seconds'] * df['P_t'] * eta_rb / (eta_m * 3600)

#     # Specific handling for the negative P_t and negative Acceleration case
#     negative_mask = (df['P_t'] < 0) & (df['Acceleration'] < 0)
#     E_t[negative_mask] *= eta_d_beb**2 * eta_batt

#     # Return the energy consumption values
#     return E_t

# Compute energy consumption for "Conventional", "hybrid" and "electric" buses
df_conventional['power']=power(df_conventional, hybrid=False, electric=False)
df_conventional['Energy']=energyConsumption_d(df_conventional)
print('Done calculating energy for CDB')
df_hybrid['power']=power(df_hybrid, hybrid=True, electric=False)
df_hybrid['Energy']=energyConsumption_d(df_hybrid, hybrid=True)
print('Done calculating energy for HEB')
df_electric['power']=power(df_electric, hybrid=False, electric=True)
df_electric['Energy']=energyConsumption_e(df_electric, electric=True)
print('Done calculating energy for BEB')


#megre subset dataframes 
df_final=pd.concat([df_conventional, df_hybrid, df_electric])
print('Done concatinating')

# Delete the no longer needed dataframes and perform garbage collection
del df_conventional, df_hybrid, df_electric
gc.collect()

# Sort dataframe
df_final.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)
print('Done sorting')

df_final.to_csv(r'../../results/computed-fuel-rates-oct2021-sep2022-12212023.csv')


