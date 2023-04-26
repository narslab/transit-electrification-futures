# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import time


#import model


f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Read trajectories df
df = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_electric=df.loc[df['Powertrain'] == 'electric'].copy()

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

# Define power function for diesel vehicle
def power_d(df_input, hybrid=False):
    if hybrid == True:
       A_f_d=A_f_heb
    else:
       A_f_d=A_f_cdb        
    df = df_input
    v = df.Speed
    a = df.Acceleration
    gr = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
    return P_t


# Define fuel rate function for diesel vehicle
def regenerative_braking(df_input, hybrid=True):
	# Apply regenerative braking for HEBs 
    if hybrid == True:
        factor = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
        P_t = factor * power_d(df_input, hybrid=True)
    else:
        P_t = power_d(df_input, hybrid=False)
    return P_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    if hybrid == True:
        P_t = regenerative_braking(df_input, hybrid=True)
    else:
        P_t = regenerative_braking(df_input, hybrid=False)
    E_t = (P_t * t)/3.78541
    return E_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional'].copy()
df_hybrid=df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid'].copy()



# read validation df
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['gallons']>0]
df_validation.sort_values(by=['equipment_id','timestamp'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"timestamp": "ServiceDateTime","equipment_id":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()


# =============================================================================
# ### calibrating model
# def calibrate_parameter(hybrid=False):
#     if hybrid==True:
#         df=df_hybrid.copy()
#         validation=df_validation.copy()
#         df['Energy']=energyConsumption_d(df, hybrid=True)
#         df.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
#         df['ServiceDateTime']=pd.to_datetime(df['ServiceDateTime'])
#         df_integrated_hybrid = validation[(validation.Powertrain == 'hybrid')].copy()
#         df_integrated_hybrid.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
#         df_integrated_hybrid=df_integrated_hybrid.reset_index()
#         for i in df_integrated_hybrid.index:
#             if i==0:
#                 pass 
#             else:
#                 if df_integrated_hybrid['Vehicle'][i]==df_integrated_hybrid['Vehicle'][i-1]:
#                     df_filtered=df.loc[(df['Vehicle']==df_integrated_hybrid['Vehicle'][i])&(df_integrated_hybrid['ServiceDateTime'][i-1]<df['ServiceDateTime'])&(df['ServiceDateTime']<df_integrated_hybrid['ServiceDateTime'][i])]
#                     df_integrated_hybrid.loc[i,'dist']=df_filtered['dist'].sum()
#                     df_integrated_hybrid.loc[i,'Energy']=df_filtered['Energy'].sum()
#                 else:
#                     pass               
#         coefficients, residuals, _, _, _ = np.polyfit(df_integrated_hybrid['Energy'], df_integrated_hybrid['gallons'], deg=2, full=True)
#         a0_heb, a1_heb, a2_heb = coefficients
#     else:
#         df=df_conventional.copy()
#         validation=df_validation.copy()
#         df['Energy']=energyConsumption_d(df, hybrid=True)
#         df.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
#         df['ServiceDateTime']=pd.to_datetime(df['ServiceDateTime'])
#         df_integrated_hybrid = validation[(validation.Powertrain == 'conventional')].copy()
#         df_integrated_hybrid.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
#         df_integrated_hybrid=df_integrated_hybrid.reset_index()
#         for i in df_integrated_hybrid.index:
#             if i==0:
#                 pass 
#             else:
#                 if df_integrated_hybrid['Vehicle'][i]==df_integrated_hybrid['Vehicle'][i-1]:
#                     df_filtered=df.loc[(df['Vehicle']==df_integrated_hybrid['Vehicle'][i])&(df_integrated_hybrid['ServiceDateTime'][i-1]<df['ServiceDateTime'])&(df['ServiceDateTime']<df_integrated_hybrid['ServiceDateTime'][i])]
#                     df_integrated_hybrid.loc[i,'dist']=df_filtered['dist'].sum()
#                     df_integrated_hybrid.loc[i,'Energy']=df_filtered['Energy'].sum()
#                 else:
#                     pass               
#         coefficients, residuals, _, _, _ = np.polyfit(df_integrated_hybrid['Energy'], df_integrated_hybrid['gallons'], deg=2, full=True)
#         a0_cdb, a1_cdb, a2_cdb = coefficients
#         return coefficients
# =============================================================================
    


def calibrate_parameter(hybrid=False):
    # Create separate dataframes for hybrid and conventional vehicles
    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()
    
    # Calculate energy consumption for each trip
    df['Energy'] = energyConsumption_d(df, hybrid=True)
    
    # Sort dataframes by vehicle and service datetime
    df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    
    # Convert ServiceDateTime column to datetime data type
    df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])
    validation['ServiceDateTime'] = pd.to_datetime(validation['ServiceDateTime'])
    
    # Calculate total distance and energy consumption for each trip in validation dataframe
    df_integrated = pd.merge_asof(validation, df, on='ServiceDateTime', by='Vehicle', direction='nearest')
    df_integrated['dist'] = np.nan
    df_integrated['Energy'] = np.nan
    mask = df_integrated['Vehicle'].eq(df_integrated['Vehicle'].shift())
    for i, mask_value in enumerate(mask):
        if mask_value:
            df_filtered = df[(df['Vehicle'] == df_integrated.loc[i, 'Vehicle']) &
                             (df['ServiceDateTime'] >= df_integrated.loc[i-1, 'ServiceDateTime']) &
                             (df['ServiceDateTime'] <= df_integrated.loc[i, 'ServiceDateTime'])]
            df_integrated.loc[i, 'dist'] = df_filtered['dist'].sum()
            df_integrated.loc[i, 'Energy'] = df_filtered['Energy'].sum()

    # Calculate coefficients for polynomial fit
    coefficients, residuals, _, _, _ = np.polyfit(df_integrated['Energy'], df_integrated['gallons'], deg=2, full=True)
    
    # Return coefficients
    return coefficients



#hybrid
a0_heb, a1_heb, a2_heb = calibrate_parameter(hybrid=True)
print("a0_heb:", a0_heb, "a1_heb:", a1_heb, "a2_heb:", a2_heb)


#conventional
a0_cdb, a1_cdb, a2_cdb = calibrate_parameter(hybrid=False)
print("a0_cdb:", a0_cdb, "a1_cdb:", a1_cdb, "a2_cdb:", a2_heb)




