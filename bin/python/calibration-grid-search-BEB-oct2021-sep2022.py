# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:56:53 2023

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

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
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
b=p.beta
gamma=0.0411

# Define power function for electric vehicle
def power_e(df_input):
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
    P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input)
    E_t = P_t * t
    return E_t

# Read computed fuel rates
df_trajectories = pd.read_csv(r'../../data/tidy/Large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_beb=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()

# read validation df
df_validation = pd.read_excel(r'../../data/tidy/Jun2022-Sep2022-BEB-validation.xlsx')
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)


### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)


def process_dataframe(df, validation):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_e(df)
    df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

    df_integrated = validation_new.copy()
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    def process_group(group):
        return pd.Series({'dist_sum': group['dist'].sum(), 'Energy_sum': group['Energy'].sum()})

    df_filtered = pd.DataFrame()
    for _, row in tqdm(df_integrated.iterrows(), total=len(df_integrated)):
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
    df_integrated.dropna(subset=['Fuel_economy'], inplace=True)
    df_integrated.dropna(subset=['Real_Fuel_economy'], inplace=True)

    return df_integrated

def calibrate_parameter(args):
    start, stop, n_points = args
    start_time = time.time()
    parameter1_values = []
    RMSE_Energy_train = []
    MAPE_Energy_train = []
    RMSE_Energy_test = []
    MAPE_Energy_test = []
    #RMSE_Economy = []
    #MAPE_Economy = []

    df = df_beb
    validation = df_validation
    validation.reset_index(inplace=True)        
    decimal_places = 6  # Set the desired number of decimal places
    gamma_values = np.around(np.linspace(start, stop, n_points), decimals=decimal_places)

    for gamma in gamma_values:        
        df_integrated = process_dataframe(df, validation)
        df_integrated.dropna(subset=['Qty', 'Energy_sum'], inplace=True)
        #train, test = train_test_split(df_integrated, test_size=0.2, random_state=42)
        ### Assign each row to the train or test dataset.
        # Calculate cumulative count of each Vehicle's rows
        df_integrated['cumulative_count'] = df_integrated.groupby('Vehicle').cumcount()
        # Calculate total count of each Vehicle's rows
        vehicle_counts = df_integrated['Vehicle'].value_counts().rename('vehicle_counts')
        # Merge this information back into df_sorted
        df_integrated = df_integrated.merge(vehicle_counts, left_on='Vehicle', right_index=True)
        # Calculate the cumulative percentage for each Vehicle
        df_integrated['cumulative_percentage'] = df_integrated['cumulative_count'] / df_integrated['vehicle_counts']
        # Assign each row to train or test set
        df_integrated['dataset'] = np.where(df_integrated['cumulative_percentage'] < 0.8, 'train', 'test')
        df_train = df_integrated[df_integrated['dataset'] == 'train']
        df_test = df_integrated[df_integrated['dataset'] == 'test']
        # drop the auxiliary columns 
        df_train = df_train.drop(columns=['cumulative_count', 'vehicle_counts', 'cumulative_percentage', 'dataset'])
        df_test = df_test.drop(columns=['cumulative_count', 'vehicle_counts', 'cumulative_percentage', 'dataset'])
        df_integrated=df_integrated.reset_index()
        MSE_Energy_train_current = mean_squared_error(df_train['Qty'], df_train['Energy_sum'])
        RMSE_Energy_train_current = math.sqrt(MSE_Energy_train_current)
        MAPE_Energy_train_current = np.mean(np.abs((df_train['Qty'] - df_train['Energy_sum']) / df_train['Qty'])) * 100
        MSE_Energy_test_current = mean_squared_error(df_test['Qty'], df_test['Energy_sum'])
        RMSE_Energy_test_current = math.sqrt(MSE_Energy_test_current)
        MAPE_Energy_test_current = np.mean(np.abs((df_test['Qty'] - df_test['Energy_sum']) / df_test['Qty'])) * 100
        #RMSE_Economy_current = mean_squared_error(df_train['Real_Fuel_economy'], df_train['Fuel_economy'], squared=False)
        #MAPE_Economy_current = np.mean(np.abs((df_train['Real_Fuel_economy'] - df_train['Fuel_economy']) / df_train['Real_Fuel_economy'])) * 100
        parameter1_values.append(gamma)
        RMSE_Energy_train.append(RMSE_Energy_train_current)
        MAPE_Energy_train.append(MAPE_Energy_train_current)
        RMSE_Energy_train.append(RMSE_Energy_test_current)
        MAPE_Energy_train.append(MAPE_Energy_test_current)
        #RMSE_Economy.append(RMSE_Economy_current)
        #MAPE_Economy.append(MAPE_Economy_current)

    results = pd.DataFrame(list(zip(parameter1_values, RMSE_Energy_train, MAPE_Energy_train, RMSE_Energy_test, MAPE_Energy_test)),
                           columns=['parameter1_values', 'RMSE_Energy_train', 'MAPE_Energy_train', 'RMSE_Energy_test', 'MAPE_Energy_test'])
    results.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))

    
calibrate_parameter((1,5, 100))