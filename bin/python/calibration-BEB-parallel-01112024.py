import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import itertools 


f = open('params-oct2021-sep2022-new-equation-12212023.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

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
#eta_d_beb = p.driveline_efficiency_d_beb
#eta_batt = p.battery_efficiency
#eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
#gamma_beb=p.gamma

# Define power function for diesel vehicle
def power(df_input, eta_d_beb, hybrid=False, electric=False):
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


# def energyConsumption_e(df_input, gamma_beb, eta_m, eta_d_beb,eta_batt, electric=True):
#      # Estimates energy consumed (KWh)
#      df = df_input
#      t = df.time_delta_in_seconds
#      P_t = power(df_input, electric) # Assuming this returns a Series of the same length as df

#      # Ensure P_t is in the DataFrame for easy access
#      df['P_t'] = P_t

#      # Updated eta_rb calculation
#      def calculate_eta_rb(row):
#          if row['P_t'] >= 0 and row['Acceleration'] >= 0:
#              return 1
#          elif row['P_t'] < 0 and row['Acceleration'] >= 0:
#              return 0
#          else:
#              return np.exp(-(gamma_beb / abs(row['Acceleration'])))

#      eta_rb = df.apply(calculate_eta_rb, axis=1)

#      E_t = t * df['P_t'] * eta_rb * eta_batt / (eta_m * 3600)
#      return eta_rb,E_t
 
def energyConsumption_e(df_input, gamma_beb, eta_m, eta_d_beb,eta_batt, electric=True):
     # Estimates energy consumed (KWh)
     df = df_input

     # Ensure P_t is in the DataFrame for easy access
     df = df_input.copy()
     df['P_t'] = power(df, electric)  # Assuming this returns a Series

     # Updated eta_rb calculation
     def calculate_eta_rb(row):
         if row['P_t'] >= 0:
             return 1
         elif row['P_t'] < 0 and row['Acceleration'] >= 0:
             return 0
         else:
             return np.exp(-(gamma_beb / abs(row['Acceleration'])))

     eta_rb = df.apply(calculate_eta_rb, axis=1)

     # Calculate E_t based on the updated cases
     E_t = df['time_delta_in_seconds'] * df['P_t'] * eta_rb / (eta_m * eta_batt * 3600)

     # Specific handling for the negative P_t and negative Acceleration case
     negative_mask = (df['P_t'] < 0) & (df['Acceleration'] < 0)
     E_t[negative_mask] *= eta_d_beb**2 * eta_batt**2 * eta_m**2

     # Return the energy consumption values
     return eta_rb,E_t

# Read computed fuel rates
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *0.44704 # Convert from mph to m/s
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_beb=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()
del df_trajectories

# Trimming the data
df_beb.rename(columns={"speed": "Speed", "acc": "Acceleration"}, inplace=True)
quantile_1 = df_beb['Acceleration'].quantile(0.0001)
quantile_99 = df_beb['Acceleration'].quantile(0.9999)
df_beb = df_beb[(df_beb['Acceleration'] >= quantile_1) & (df_beb['Acceleration'] <= quantile_99)]

# read validation df
df_validation = pd.read_excel(r'../../data/tidy/BEB-validation.xlsx')
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])

def process_dataframe(df, validation, gamma_beb, eta_m, eta_d_beb, eta_batt):
    df_new = df.copy()
    validation_new = validation.copy()

    eta_rb,df_new['Energy'] = energyConsumption_e(df, gamma_beb, eta_m, eta_d_beb, eta_batt)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])
    df_new = df_new.groupby(['Date', 'Vehicle'])[['Energy', 'dist']].sum().reset_index()


    df_integrated =  pd.merge(validation_new, df_new, 
                     left_on=['ServiceDateTime', 'Vehicle'], 
                     right_on=['Date', 'Vehicle'], 
                     how='left').copy()
    #df_integrated['residual']=df_integrated['trip']-df_integrated['Energy']
    df_integrated = df_integrated.dropna(subset=['trip', 'Energy'])

    # Drop rows where 'trip' or 'Predicted Energy' is 0
    df_integrated = df_integrated.query("trip != 0 and `Energy` != 0")
    
    df_integrated['actual_mpg']=df_integrated['dist']/df_integrated['trip'].copy()
    
    # Calculate the 5st and 95th percentiles
    percentile_5 = df_integrated['actual_mpg'].quantile(0.05)
    percentile_95 = df_integrated['actual_mpg'].quantile(0.95)

    # Filter the DataFrame
    df_integrated = df_integrated[(df_integrated['actual_mpg'] >= percentile_5) & (df_integrated['actual_mpg'] <= percentile_95)]

    return eta_rb,df_integrated

def process_single_combination(args):
    gamma, eta_m, eta_d_beb, eta_batt = args
    eta_rb, df_integrated = process_dataframe(df_beb.copy(), df_validation.copy(), gamma, eta_m, eta_d_beb, eta_batt)
    df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)
    RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
    MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'], df_train['Energy'])

    return gamma, eta_m, eta_d_beb, eta_batt, eta_rb, RMSE_Energy_train_current, MAPE_Energy_train_current

def calibrate_parameter_parallel(start_gamma, stop_gamma, n_points_gamma, start_eta_m, stop_eta_m, n_points_eta_m, start_eta_d_beb, stop_eta_d_beb, n_points_eta_d_beb, start_eta_batt, stop_eta_batt, n_points_eta_batt):
    decimal_places = 5
    gamma_values = np.around(np.linspace(start_gamma, stop_gamma, n_points_gamma), decimals=decimal_places)
    eta_m_values = np.around(np.linspace(start_eta_m, stop_eta_m, n_points_eta_m), decimals=decimal_places)
    eta_d_beb_values = np.around(np.linspace(start_eta_d_beb, stop_eta_d_beb, n_points_eta_d_beb), decimals=decimal_places)
    eta_batt_values = np.around(np.linspace(start_eta_batt, stop_eta_batt, n_points_eta_batt), decimals=decimal_places)

    parameter_combinations = list(itertools.product(gamma_values, eta_m_values, eta_d_beb_values, eta_batt_values))

    results = []
    with ProcessPoolExecutor(max_workers=64) as executor:
        for result in tqdm(executor.map(process_single_combination, parameter_combinations), total=len(parameter_combinations)):
            results.append(result)

    results_df = pd.DataFrame(results, columns=['parameter1_values', 'parameter2_values', 'parameter3_values', 'parameter4_values', 'eta_rb', 'RMSE_Energy_train', 'MAPE_Energy_train'])
    results_df.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_01122024_parallel.csv'))


    
#calibrate_parameter((0.00001,1, 500, 0.9,0.99, 5, 0.9,0.99, 5))
calibrate_parameter_parallel(0.00000001,0.01, 100, 0.70,1, 10,  0.70,1, 10, 0.70,1, 10)

