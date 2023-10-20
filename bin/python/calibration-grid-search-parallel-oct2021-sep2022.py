import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os


f = open('params-oct2021-sep2022.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
C_h= p.altitude_correction_factor
A_f_cdb = p.frontal_area_cdb
A_f_heb = p.frontal_area_heb
A_f_beb = p.frontal_area_beb
g = p.gravitational_acceleration
C_r = p.rolling_coefficient
c1 = p.rolling_resistance_coef1
c2 = p.rolling_resistance_coef2
eta_d_dis = p.driveline_efficiency_d_dis
eta_d_beb = p.driveline_efficiency_d_beb
P_mfo = p.idling_mean_fuel_pressure
omega = p.idling_speed
d = p.engine_displacement
Q = p.fuel_lower_heating_value
N = p.number_of_engine_cylinders
FE_city_p = p.fuel_economy_city
FE_hwy_p = p.fuel_economy_hwy
eps = p.epsilon
eta_batt = p.battery_efficiency
eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2 = p.alpha_2
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
b=p.beta
gamma_heb=p.gamma

# Define power function for diesel vehicle
def power_d(df_input, hybrid=False):
    if hybrid == True:
       A_f_d=A_f_heb
    else:
       A_f_d=A_f_cdb        
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    P_t = (1/float(3600*eta_d_dis)) * ((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr) * v
    return P_t


# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, a0, a1, hybrid=False):
    # Set the coefficients based on the hybrid flag
    if hybrid:
        a0_local = a0
        a1_local = a1
    else:
        a0_local = a0
        a1_local = a1

    # Compute power
    P_t = power_d(df_input, hybrid=hybrid)

    # Vectorized computation of fuel rate based on the condition
    condition = P_t >= 0
    FC_t = np.where(condition, a0_local + a1_local*P_t + a2*P_t*P_t, a0_local)

    # Adjust the value if it's a hybrid vehicle
    if hybrid:
        FC_t *= b

    return FC_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input,a0, a1, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    if hybrid == True:
        FC_t = fuelRate_d(df_input,a0, a1, hybrid=True)
    else:
        FC_t = fuelRate_d(df_input,a0, a1, hybrid=False)
    E_t = (FC_t * t)/3.78541 #The value 3.78541 represents the number of liters in one US gallon
    return E_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date'] = pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934  # Convert from mph to km/h
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame
df_conventional = df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional']
df_hybrid = df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid']
del df_trajectories

# read validation df
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['Qty']>0]
df_validation.sort_values(by=['Equipment ID','Transaction Date'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()

### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)

# Delete unnecessary dataframes
del df2, mydict


def process_dataframe(df, validation, a0, a1, hybrid):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_d(df_new,a0, a1, hybrid=hybrid)
    df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

    df_integrated = validation_new.copy()
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    def process_group(group):
        return pd.Series({'dist_sum': group['dist'].sum(), 'Energy_sum': group['Energy'].sum()})

    df_filtered = pd.DataFrame()
    for _, row in df_integrated.iterrows():
        vehicle, cur_time, prev_time = row['Vehicle'], row['ServiceDateTime'], row['ServiceDateTime_prev']
        group = df_new[(df_new['Vehicle'] == vehicle) & (df_new['ServiceDateTime'] > prev_time) & (df_new['ServiceDateTime'] < cur_time)]
        filtered_group = process_group(group)
        filtered_group['Vehicle'] = vehicle
        filtered_group['ServiceDateTime_cur'] = cur_time
        filtered_group['ServiceDateTime_prev'] = prev_time
        df_filtered = pd.concat([df_filtered, filtered_group.to_frame().T], ignore_index=True)
    df_integrated = df_integrated.merge(df_filtered, left_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'],
                                        right_on=['Vehicle', 'ServiceDateTime_cur', 'ServiceDateTime_prev'])

    # Drop rows with NaN values in 'Energy' or 'Qty' columns
    df_integrated.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
    df_integrated['Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Energy_sum'], where=df_integrated['Energy_sum'] != 0)
    df_integrated['Real_Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Qty'], where=df_integrated['Energy_sum'] != 0)
    #df_integrated.dropna(subset=['Fuel_economy'], inplace=True)
    df_integrated.dropna(subset=['Real_Fuel_economy'], inplace=True)
    #print(df_integrated)
    return df_integrated


# Calibrate parameters with Dask + Joblib for parallel processing
def calibrate_parameter(a0, a1, hybrid):
    
    start_time = time.time()

    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()

    validation.reset_index(drop=True, inplace=True)    
    
    #print("hybrid:",hybrid)
    #print("df columns:",df.columns)
    df_integrated = process_dataframe(df, validation, a0, a1, hybrid)
    df_integrated = df_integrated.loc[df_integrated['Energy_sum']!=0]
    train, test = train_test_split(df_integrated, test_size=0.2, random_state=42)
    
    # Training Data
    RMSE_Energy_train = np.sqrt(mean_squared_error(train['Qty'], train['Energy_sum']))
    MAPE_Energy_train = mean_absolute_percentage_error(train['Qty'] , train['Energy_sum'])
    RMSE_Economy_train = np.sqrt(mean_squared_error(train['Real_Fuel_economy'], train['Fuel_economy']))
    MAPE_Economy_train = mean_absolute_percentage_error(train['Real_Fuel_economy'] , train['Fuel_economy']) 
    
    # Testing Data
    RMSE_Energy_test = np.sqrt(mean_squared_error(test['Qty'], test['Energy_sum']))
    MAPE_Energy_test = mean_absolute_percentage_error(test['Qty'] , test['Energy_sum'])
    RMSE_Economy_test = np.sqrt(mean_squared_error(test['Real_Fuel_economy'], test['Fuel_economy']))
    MAPE_Economy_test = mean_absolute_percentage_error(test['Real_Fuel_economy'] , test['Fuel_economy'])
            
    results_df = pd.DataFrame({
        'parameter1_values': [a0], 
        'parameter2_values': [a1],
        'RMSE_Energy_train': [RMSE_Energy_train], 
        'MAPE_Energy_train': [MAPE_Energy_train], 
        'RMSE_Economy_train': [RMSE_Economy_train], 
        'MAPE_Economy_train': [MAPE_Economy_train],
        'RMSE_Energy_test': [RMSE_Energy_test], 
        'MAPE_Energy_test': [MAPE_Energy_test], 
        'RMSE_Economy_test': [RMSE_Economy_test], 
        'MAPE_Economy_test': [MAPE_Economy_test]
    })
    return results_df

# Configuration Section
#START1_VAL = 0.000001
START1_VAL = 0
#STOP1_VAL = 0.003
STEP_SIZE1 = 0.000001

#START2_VAL = 0.00001 
START2_VAL = 0
#STOP2_VAL = 0.0009
STEP_SIZE2 = 0.00001
N_POINTS = 100

# Initialize results dataframe to store results of all iterations
all_results_df = pd.DataFrame()

# The calibration process
# for hybrid_flag in [False, True]:
for hybrid_flag in [True]:
    for a0 in tqdm(np.linspace(START1_VAL, START1_VAL+(STEP_SIZE1 * (N_POINTS-1)), N_POINTS), desc="Calibrating a0"):
        for a1 in tqdm(np.linspace(START2_VAL, START2_VAL+(STEP_SIZE2 * (N_POINTS-1)), N_POINTS), desc="Calibrating a1", leave=False):
            current_result_df = calibrate_parameter(a0, a1, hybrid_flag)
            all_results_df = pd.concat([all_results_df, current_result_df], ignore_index=True)

# Write the CSV at the end of the process
all_results_df.to_csv('../../results/calibration_results_heb_oct2021-sep2022_10202023.csv', index=False)
